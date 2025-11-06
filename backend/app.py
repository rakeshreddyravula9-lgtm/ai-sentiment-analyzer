from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
from textblob import TextBlob
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load ML model (we'll create a simple one for demonstration)
class SentimentAnalyzer:
    def __init__(self):
        self.model_loaded = False
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#', '', text)  # Remove hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = text.lower().strip()
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        cleaned_text = self.clean_text(text)
        blob = TextBlob(cleaned_text)
        
        # Get polarity (-1 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = "Positive"
            confidence = min((polarity + 1) / 2 * 100, 100)
        elif polarity < -0.1:
            sentiment = "Negative"
            confidence = min((abs(polarity) + 1) / 2 * 100, 100)
        else:
            sentiment = "Neutral"
            confidence = 50 + (1 - abs(polarity)) * 25
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "cleaned_text": cleaned_text
        }

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    return jsonify({
        "message": "AI Sentiment Analyzer API",
        "version": "1.0",
        "endpoints": {
            "/api/analyze": "POST - Analyze sentiment of text",
            "/api/batch": "POST - Batch analysis of multiple texts",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        if len(text.strip()) == 0:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Analyze sentiment
        result = analyzer.analyze_sentiment(text)
        result['original_text'] = text
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        results = []
        for text in texts:
            if len(text.strip()) > 0:
                result = analyzer.analyze_sentiment(text)
                result['original_text'] = text
                results.append(result)
        
        return jsonify({
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify({
        "model": "TextBlob Sentiment Analyzer",
        "version": "1.0",
        "features": [
            "Sentiment Classification (Positive/Negative/Neutral)",
            "Confidence Score",
            "Polarity Analysis",
            "Subjectivity Detection",
            "Batch Processing"
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
