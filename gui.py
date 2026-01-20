import os
import re
import json
import string
import numpy as np
import tkinter as tk
from tkinter import ttk

import tensorflow as tf
from tensorflow import argmax
import keras
from keras.layers import TextVectorization
from keras.models import load_model
from keras.src.legacy.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences

# --- 1. DEFINE & REGISTER CUSTOM TRANSFORMER LAYERS ---
# These are required to rebuild the Spanish Transformer model
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

# --- 3. LOAD SPANISH COMPONENTS ---
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

# Load Spanish Model with ALL Custom Objects
transformer = load_model(
    os.path.join(BASE_DIR, 'transformer_model.keras'),
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder
    }
)

spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

# --- 4. LOAD FRENCH COMPONENTS ---
# Load the .keras model file
model_fr = load_model(os.path.join(BASE_DIR, 'english_to_french_model.keras'))
english_tokenizer = tokenizer_from_json(load_json_data('english_tokenizer.json'))
french_tokenizer = tokenizer_from_json(load_json_data('french_tokenizer.json'))
fr_max_length = load_json_data('sequence_length.json')

# --- 5. TRANSLATION LOGIC ---
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
    predictions = model_fr.predict(padded)[0]
    french_indices = [np.argmax(word) for word in predictions]
    return french_tokenizer.sequences_to_texts([french_indices])[0]

def translate_to_spanish(english_sentence):
    result = decode_sentence(english_sentence)
    return result.replace("[start]", "").replace("[end]", "").strip()

# --- 6. GUI SETUP ---
# Modern Color Scheme
COLORS = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'bg_card': '#0f3460',
    'accent': '#e94560',
    'accent_hover': '#ff6b85',
    'text_primary': '#ffffff',
    'text_secondary': '#b8b8d4',
    'border': '#2a2a3e',
    'success': '#4ade80',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

def create_gradient_frame(parent, width, height):
    """Create a frame with gradient-like appearance"""
    frame = tk.Frame(parent, bg=COLORS['bg_card'], relief='flat', bd=0)
    return frame

def handle_translate():
    lang = language_var.get()
    text = text_input.get("1.0", "end-1c")
    
    if not text.strip():
        status_label.config(text="⚠️ Please enter text to translate", fg='#fbbf24')
        return
    
    if not lang:
        status_label.config(text="⚠️ Please select a target language", fg='#fbbf24')
        return
    
    # Update button to show loading state
    translate_btn.config(text="🔄 Translating...", state='disabled')
    status_label.config(text="⏳ Processing translation...", fg=COLORS['accent'])
    root.update()
    
    try:
        translation = translate_to_french(text) if lang == "French" else translate_to_spanish(text)
        translation_output.delete("1.0", "end")
        
        # Format output nicely
        lang_emoji = "🇫🇷" if lang == "French" else "🇪🇸"
        formatted_output = f"{lang_emoji} {lang} Translation:\n\n{translation}"
        translation_output.insert("end", formatted_output)
        
        status_label.config(text="✅ Translation completed successfully!", fg=COLORS['success'])
    except Exception as e:
        status_label.config(text=f"❌ Error: {str(e)}", fg='#ef4444')
        translation_output.delete("1.0", "end")
        translation_output.insert("end", f"Error occurred during translation.")
    finally:
        translate_btn.config(text="🚀 Translate", state='normal')

def on_language_change(event=None):
    """Update UI when language changes"""
    lang = language_var.get()
    if lang:
        lang_emoji = "🇫🇷" if lang == "French" else "🇪🇸"
        status_label.config(text=f"✓ {lang_emoji} {lang} selected", fg=COLORS['success'])

def clear_text():
    """Clear input and output"""
    text_input.delete("1.0", "end")
    translation_output.delete("1.0", "end")
    status_label.config(text="✨ Ready to translate", fg=COLORS['text_secondary'])

# Create main window
root = tk.Tk()
root.title("🌍 AI Language Translator")
root.geometry("900x750")
root.configure(bg=COLORS['bg_primary'])

# Configure style
style = ttk.Style()
style.theme_use('clam')

# Configure custom styles
style.configure('Title.TLabel', 
                background=COLORS['bg_primary'],
                foreground=COLORS['text_primary'],
                font=('Segoe UI', 28, 'bold'))

style.configure('Subtitle.TLabel',
                background=COLORS['bg_primary'],
                foreground=COLORS['text_secondary'],
                font=('Segoe UI', 11))

style.configure('Card.TFrame',
                background=COLORS['bg_card'],
                relief='flat')

style.configure('Card.TLabel',
                background=COLORS['bg_card'],
                foreground=COLORS['text_primary'],
                font=('Segoe UI', 10, 'bold'))

style.configure('Custom.TButton',
                font=('Segoe UI', 12, 'bold'),
                padding=15)

style.map('Custom.TButton',
          background=[('active', COLORS['accent_hover']),
                     ('!active', COLORS['accent'])])

style.configure('Custom.TCombobox',
                fieldbackground=COLORS['bg_secondary'],
                background=COLORS['bg_secondary'],
                foreground=COLORS['text_primary'],
                borderwidth=2,
                relief='flat',
                padding=10)

# Header Section
header_frame = tk.Frame(root, bg=COLORS['bg_primary'], height=90)
header_frame.pack(fill='x', padx=0, pady=0)
header_frame.pack_propagate(False)

title_label = tk.Label(header_frame,
                      text="🌍 AI Language Translator",
                      font=('Segoe UI', 24, 'bold'),
                      bg=COLORS['bg_primary'],
                      fg=COLORS['text_primary'])
title_label.pack(pady=(15, 3))

subtitle_label = tk.Label(header_frame,
                         text="Powered by Deep Learning • Translate between English, French & Spanish",
                         font=('Segoe UI', 9),
                         bg=COLORS['bg_primary'],
                         fg=COLORS['text_secondary'])
subtitle_label.pack(pady=(0, 10))

# Main container with padding
main_container = tk.Frame(root, bg=COLORS['bg_primary'])
main_container.pack(fill='both', expand=True, padx=30, pady=15)

# Input Card
input_card = tk.Frame(main_container, bg=COLORS['bg_card'], relief='flat', bd=0)
input_card.pack(fill='x', pady=(0, 12))

input_header = tk.Frame(input_card, bg=COLORS['bg_card'], height=35)
input_header.pack(fill='x', padx=15, pady=(12, 8))
input_header.pack_propagate(False)

tk.Label(input_header,
        text="📝 Enter Text to Translate",
        font=('Segoe UI', 13, 'bold'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary'],
        anchor='w').pack(side='left')

# Text input with styling
text_input_frame = tk.Frame(input_card, bg=COLORS['bg_card'])
text_input_frame.pack(fill='x', padx=15, pady=(0, 12))

text_input = tk.Text(text_input_frame,
                    height=5,
                    width=70,
                    font=('Segoe UI', 12),
                    bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'],
                    insertbackground=COLORS['accent'],
                    selectbackground=COLORS['accent'],
                    selectforeground=COLORS['text_primary'],
                    relief='flat',
                    bd=8,
                    wrap='word',
                    padx=12,
                    pady=12)
text_input.pack(fill='both', expand=True)

# Language Selection Card
lang_card = tk.Frame(main_container, bg=COLORS['bg_card'], relief='flat', bd=0)
lang_card.pack(fill='x', pady=(0, 12))

lang_header = tk.Frame(lang_card, bg=COLORS['bg_card'], height=35)
lang_header.pack(fill='x', padx=15, pady=(12, 8))
lang_header.pack_propagate(False)

tk.Label(lang_header,
        text="🌐 Select Target Language",
        font=('Segoe UI', 13, 'bold'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary'],
        anchor='w').pack(side='left')

language_var = tk.StringVar()
language_var.trace('w', lambda *args: on_language_change())

lang_select_frame = tk.Frame(lang_card, bg=COLORS['bg_card'])
lang_select_frame.pack(fill='x', padx=15, pady=(0, 12))

language_select = ttk.Combobox(lang_select_frame,
                              textvariable=language_var,
                              values=["French", "Spanish"],
                              state="readonly",
                              font=('Segoe UI', 12),
                              style='Custom.TCombobox',
                              width=68)
language_select.pack(fill='x', ipady=8)

# Button Frame
button_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
button_frame.pack(fill='x', pady=(0, 10))

translate_btn = tk.Button(button_frame,
                         text="🚀 Translate",
                         command=handle_translate,
                         font=('Segoe UI', 13, 'bold'),
                         bg=COLORS['accent'],
                         fg=COLORS['text_primary'],
                         activebackground=COLORS['accent_hover'],
                         activeforeground=COLORS['text_primary'],
                         relief='flat',
                         bd=0,
                         cursor='hand2',
                         padx=25,
                         pady=12)
translate_btn.pack(side='left', padx=(0, 8))

clear_btn = tk.Button(button_frame,
                     text="🗑️ Clear",
                     command=clear_text,
                     font=('Segoe UI', 12),
                     bg=COLORS['bg_secondary'],
                     fg=COLORS['text_secondary'],
                     activebackground=COLORS['border'],
                     activeforeground=COLORS['text_primary'],
                     relief='flat',
                     bd=0,
                     cursor='hand2',
                     padx=20,
                     pady=12)
clear_btn.pack(side='left')

# Status Label
status_label = tk.Label(main_container,
                       text="✨ Ready to translate",
                       font=('Segoe UI', 10),
                       bg=COLORS['bg_primary'],
                       fg=COLORS['text_secondary'],
                       anchor='w')
status_label.pack(fill='x', pady=(0, 15))

# Output Card
output_card = tk.Frame(main_container, bg=COLORS['bg_card'], relief='flat', bd=0)
output_card.pack(fill='both', expand=True)

output_header = tk.Frame(output_card, bg=COLORS['bg_card'], height=35)
output_header.pack(fill='x', padx=15, pady=(12, 8))
output_header.pack_propagate(False)

tk.Label(output_header,
        text="📄 Translation Result",
        font=('Segoe UI', 13, 'bold'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary'],
        anchor='w').pack(side='left')

# Translation output with styling and scrollbar
output_frame = tk.Frame(output_card, bg=COLORS['bg_card'])
output_frame.pack(fill='both', expand=True, padx=15, pady=(0, 12))

# Create scrollbar for output
scrollbar = tk.Scrollbar(output_frame, bg=COLORS['bg_secondary'], troughcolor=COLORS['bg_card'], width=12)
scrollbar.pack(side='right', fill='y')

translation_output = tk.Text(output_frame,
                            height=12,
                            width=70,
                            font=('Segoe UI', 12),
                            bg=COLORS['bg_secondary'],
                            fg=COLORS['text_primary'],
                            insertbackground=COLORS['accent'],
                            selectbackground=COLORS['accent'],
                            selectforeground=COLORS['text_primary'],
                            relief='flat',
                            bd=8,
                            wrap='word',
                            padx=12,
                            pady=12,
                            state='normal',
                            yscrollcommand=scrollbar.set)
translation_output.pack(side='left', fill='both', expand=True)

# Configure scrollbar
scrollbar.config(command=translation_output.yview)

# Footer
footer = tk.Label(root,
                 text="Made with ❤️ using TensorFlow & Keras",
                 font=('Segoe UI', 8),
                 bg=COLORS['bg_primary'],
                 fg=COLORS['text_secondary'])
footer.pack(side='bottom', pady=8)

# Center window on screen and ensure proper size
root.update_idletasks()
# Set explicit size to ensure all elements are visible
x = (root.winfo_screenwidth() // 2) - (900 // 2)
y = (root.winfo_screenheight() // 2) - (750 // 2)
root.geometry(f'900x750+{x}+{y}')

# Make window resizable for better user experience
root.resizable(True, True)
root.minsize(800, 650)  # Set minimum size

root.mainloop()