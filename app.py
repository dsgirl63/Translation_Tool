import os
import re
import json
import string
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import keras
from keras.layers import TextVectorization
from keras.models import load_model
from keras.src.legacy.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- 1. DEFINE & REGISTER CUSTOM TRANSFORMER LAYERS ---
@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "sequence_length": self.sequence_length, "vocab_size": self.vocab_size})
        return config

@keras.saving.register_keras_serializable()
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"), 
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "dense_dim": self.dense_dim, "num_heads": self.num_heads})
        return config

@keras.saving.register_keras_serializable()
class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(latent_dim, activation="relu"), 
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "latent_dim": self.latent_dim, "num_heads": self.num_heads})
        return config

# --- 2. SETUP PATHS & STANDARDIZATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

def load_json_data(filename):
    with open(os.path.join(BASE_DIR, filename), 'r', encoding='utf-8') as f:
        return json.load(f)

# Global variables for models (loaded once at startup)
transformer = None
model_fr = None
eng_vectorization = None
spa_vectorization = None
spa_index_lookup = None
english_tokenizer = None
french_tokenizer = None
fr_max_length = None

def load_models():
    """Load all models and tokenizers"""
    global transformer, model_fr, eng_vectorization, spa_vectorization
    global spa_index_lookup, english_tokenizer, french_tokenizer, fr_max_length
    
    print("Loading models...")
    
    # Load Spanish components
    eng_config = load_json_data('eng_vectorization_config.json')
    spa_config = load_json_data('spa_vectorization_config.json')
    eng_vocab = load_json_data('eng_vocab.json')
    spa_vocab = load_json_data('spa_vocab.json')
    
    eng_vectorization = TextVectorization(
        max_tokens=eng_config['max_tokens'],
        output_mode=eng_config['output_mode'],
        output_sequence_length=eng_config['output_sequence_length'],
        standardize=custom_standardization
    )
    eng_vectorization.set_vocabulary(eng_vocab)
    
    spa_vectorization = TextVectorization(
        max_tokens=spa_config['max_tokens'],
        output_mode=spa_config['output_mode'],
        output_sequence_length=spa_config['output_sequence_length'],
        standardize=custom_standardization
    )
    spa_vectorization.set_vocabulary(spa_vocab)
    
    # Load Spanish Model
    transformer = load_model(
        os.path.join(BASE_DIR, 'transformer_model.keras'),
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder
        }
    )
    
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    
    # Load French components
    model_fr = load_model(os.path.join(BASE_DIR, 'english_to_french_model.keras'))
    english_tokenizer = tokenizer_from_json(load_json_data('english_tokenizer.json'))
    french_tokenizer = tokenizer_from_json(load_json_data('french_tokenizer.json'))
    fr_max_length = load_json_data('sequence_length.json')
    
    print("Models loaded successfully!")

# --- 3. TRANSLATION LOGIC ---
def decode_sentence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(20):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item()
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower().translate(str.maketrans('', '', string.punctuation))
    sequences = english_tokenizer.texts_to_sequences([english_sentence])
    padded = pad_sequences(sequences, maxlen=fr_max_length, padding='post')
    predictions = model_fr.predict(padded, verbose=0)[0]
    french_indices = [np.argmax(word) for word in predictions]
    return french_tokenizer.sequences_to_texts([french_indices])[0]

def translate_to_spanish(english_sentence):
    result = decode_sentence(english_sentence)
    return result.replace("[start]", "").replace("[end]", "").strip()

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not language:
            return jsonify({'error': 'Language is required'}), 400
        
        if language not in ['French', 'Spanish']:
            return jsonify({'error': 'Invalid language. Must be French or Spanish'}), 400
        
        # Perform translation
        if language == 'French':
            translation = translate_to_french(text)
        else:
            translation = translate_to_spanish(text)
        
        return jsonify({
            'success': True,
            'translation': translation,
            'language': language
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': transformer is not None})

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
