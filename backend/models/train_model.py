"""
Advanced Sentiment Analysis Model
This script trains a machine learning model for sentiment analysis
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import joblib
import re
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def create_sample_data(self):
        """Create sample training data"""
        positive_samples = [
            "This is amazing! I love it so much!",
            "Excellent product, highly recommend!",
            "Best experience ever, very satisfied",
            "Wonderful service, will come back again",
            "Absolutely fantastic, exceeded expectations",
            "Great quality and fast delivery",
            "I'm very happy with this purchase",
            "Outstanding performance and value",
            "This made my day, thank you!",
            "Perfect! Exactly what I needed"
        ]
        
        negative_samples = [
            "This is terrible, very disappointed",
            "Worst product ever, don't buy",
            "Horrible experience, waste of money",
            "Poor quality and bad customer service",
            "I hate this, complete disaster",
            "Awful, nothing works as advertised",
            "Very unhappy and frustrated",
            "Terrible quality, broke immediately",
            "Disappointed and angry",
            "This is a scam, avoid at all costs"
        ]
        
        neutral_samples = [
            "It's okay, nothing special",
            "Average product, does the job",
            "Meh, could be better",
            "Standard quality, as expected",
            "It works, but not impressive",
            "Normal experience, nothing to note",
            "Fair price for what you get",
            "Acceptable, meets basic requirements",
            "Neither good nor bad",
            "It's fine, no complaints"
        ]
        
        # Create DataFrame
        data = {
            'text': positive_samples + negative_samples + neutral_samples,
            'sentiment': ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
        }
        
        return pd.DataFrame(data)
    
    def train(self, data=None):
        """Train the sentiment analysis model"""
        if data is None:
            print("Creating sample training data...")
            data = self.create_sample_data()
        
        print(f"Training with {len(data)} samples")
        
        # Clean texts
        data['cleaned_text'] = data['text'].apply(self.clean_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['cleaned_text'], 
            data['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train multiple models and choose the best
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        print("\nTraining models...")
        for name, model in models.items():
            model.fit(X_train_vec, y_train)
            score = model.score(X_test_vec, y_test)
            print(f"{name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
        
        # Predictions and metrics
        y_pred = self.model.predict(X_test_vec)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return best_score
    
    def predict(self, text):
        """Predict sentiment for new text"""
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        
        return {
            'sentiment': prediction,
            'confidence': max(probabilities) * 100,
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def save_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='models/sentiment_model.pkl',
                   vectorizer_path='models/vectorizer.pkl'):
        """Load trained model and vectorizer"""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully")

if __name__ == "__main__":
    print("=" * 50)
    print("Sentiment Analysis Model Training")
    print("=" * 50)
    
    trainer = SentimentModelTrainer()
    
    # Train model
    accuracy = trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Test predictions
    test_texts = [
        "This is absolutely wonderful!",
        "I hate this so much",
        "It's okay, nothing special"
    ]
    
    print("\n" + "=" * 50)
    print("Testing predictions:")
    print("=" * 50)
    for text in test_texts:
        result = trainer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}%")
