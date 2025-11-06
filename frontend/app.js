// API Configuration
const API_URL = 'http://localhost:5000';

// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('results');
const loadingSection = document.getElementById('loading');
const errorSection = document.getElementById('error');
const exampleChips = document.querySelectorAll('.chip');

// Event Listeners
analyzeBtn.addEventListener('click', analyzeSentiment);
clearBtn.addEventListener('click', clearAll);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeSentiment();
    }
});

// Example chips
exampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
        textInput.value = chip.getAttribute('data-text');
        analyzeSentiment();
    });
});

// Functions
async function analyzeSentiment() {
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    // Show loading
    hideAll();
    loadingSection.style.display = 'block';
    
    try {
        const response = await fetch(`${API_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error analyzing sentiment: ${error.message}. Make sure the backend server is running on ${API_URL}`);
    } finally {
        loadingSection.style.display = 'none';
    }
}

function displayResults(data) {
    hideAll();
    
    // Set sentiment
    const sentimentEl = document.getElementById('sentiment');
    sentimentEl.textContent = data.sentiment;
    sentimentEl.className = `sentiment-badge ${data.sentiment.toLowerCase()}`;
    
    // Set confidence
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    confidenceBar.style.width = `${data.confidence}%`;
    confidenceText.textContent = `${data.confidence}%`;
    
    // Set metrics
    document.getElementById('polarity').textContent = data.polarity;
    document.getElementById('subjectivity').textContent = data.subjectivity;
    
    // Set cleaned text
    document.getElementById('cleanedText').textContent = data.cleaned_text;
    
    // Show results
    resultsSection.style.display = 'block';
}

function showError(message) {
    hideAll();
    errorSection.textContent = message;
    errorSection.style.display = 'block';
}

function hideAll() {
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function clearAll() {
    textInput.value = '';
    hideAll();
    textInput.focus();
}

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        if (response.ok) {
            console.log('✅ Backend API is running');
        }
    } catch (error) {
        console.warn('⚠️ Backend API is not responding. Please start the Flask server.');
        showError('Backend server is not running. Please start it with: cd backend && python app.py');
    }
}

// Initialize
checkAPIHealth();
