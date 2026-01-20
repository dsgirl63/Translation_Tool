// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const textInput = document.getElementById('textInput');
const languageSelect = document.getElementById('languageSelect');
const translateBtn = document.getElementById('translateBtn');
const clearBtn = document.getElementById('clearBtn');
const translationOutput = document.getElementById('translationOutput');
const statusMessage = document.getElementById('statusMessage');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
});

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        if (data.status === 'healthy') {
            updateStatus('✨ Ready to translate', 'success');
        } else {
            updateStatus('⚠️ Service unavailable', 'warning');
        }
    } catch (error) {
        updateStatus('⚠️ Unable to connect to server', 'warning');
        console.error('Health check failed:', error);
    }
}

// Setup Event Listeners
function setupEventListeners() {
    translateBtn.addEventListener('click', handleTranslate);
    clearBtn.addEventListener('click', handleClear);
    
    // Translate on Enter (Ctrl+Enter or Cmd+Enter)
    textInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            handleTranslate();
        }
    });
    
    // Update status when language changes
    languageSelect.addEventListener('change', (e) => {
        if (e.target.value) {
            const langEmoji = e.target.value === 'French' ? '🇫🇷' : '🇪🇸';
            updateStatus(`✓ ${langEmoji} ${e.target.value} selected`, 'success');
        }
    });
}

// Handle Translation
async function handleTranslate() {
    const text = textInput.value.trim();
    const language = languageSelect.value;
    
    // Validation
    if (!text) {
        updateStatus('⚠️ Please enter text to translate', 'warning');
        return;
    }
    
    if (!language) {
        updateStatus('⚠️ Please select a target language', 'warning');
        return;
    }
    
    // Update UI for loading state
    translateBtn.disabled = true;
    translateBtn.textContent = '🔄 Translating...';
    updateStatus('⏳ Processing translation...', 'loading');
    translationOutput.value = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Format output
            const langEmoji = language === 'French' ? '🇫🇷' : '🇪🇸';
            const formattedOutput = `${langEmoji} ${language} Translation:\n\n${data.translation}`;
            translationOutput.value = formattedOutput;
            updateStatus('✅ Translation completed successfully!', 'success');
        } else {
            throw new Error(data.error || 'Translation failed');
        }
    } catch (error) {
        console.error('Translation error:', error);
        translationOutput.value = `Error: ${error.message}`;
        updateStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        translateBtn.disabled = false;
        translateBtn.textContent = '🚀 Translate';
    }
}

// Handle Clear
function handleClear() {
    textInput.value = '';
    translationOutput.value = '';
    languageSelect.value = '';
    updateStatus('✨ Ready to translate', 'success');
    textInput.focus();
}

// Update Status Message
function updateStatus(message, type = '') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
}
