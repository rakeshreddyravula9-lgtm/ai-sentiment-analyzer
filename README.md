# 🤖 AI Sentiment Analyzer

A full-stack AI/ML web application that analyzes the sentiment of text using Natural Language Processing (NLP) and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time Sentiment Analysis**: Analyze text sentiment instantly
- **Confidence Scoring**: Get confidence levels for predictions
- **Polarity & Subjectivity Metrics**: Detailed emotional analysis
- **Clean UI**: Modern, responsive web interface
- **RESTful API**: Easy-to-use Flask backend
- **Batch Processing**: Analyze multiple texts at once
- **Text Preprocessing**: Automatic cleaning and normalization

## 🎯 Sentiment Categories

- **Positive** 😊 - Optimistic, happy, satisfied content
- **Negative** 😞 - Pessimistic, unhappy, dissatisfied content  
- **Neutral** 😐 - Balanced, factual, objective content

## 🛠️ Tech Stack

### Backend
- **Flask** - Python web framework
- **TextBlob** - NLP library for sentiment analysis
- **Scikit-learn** - Machine learning library
- **NLTK** - Natural language toolkit
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with modern gradients and animations
- **Vanilla JavaScript** - Interactive functionality
- **Fetch API** - HTTP requests

### Machine Learning
- **TF-IDF Vectorization** - Text feature extraction
- **Logistic Regression** - Classification model
- **Naive Bayes** - Alternative classification
- **Random Forest** - Ensemble learning

## 📁 Project Structure

```
ai-sentiment-analyzer/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── models/
│   │   ├── train_model.py     # ML model training script
│   │   ├── sentiment_model.pkl
│   │   └── vectorizer.pkl
│   └── utils/
├── frontend/
│   ├── index.html             # Main HTML page
│   ├── style.css              # Styling
│   └── app.js                 # JavaScript logic
├── notebooks/
│   └── sentiment_analysis.ipynb
├── requirements.txt           # Python dependencies
├── README.md
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rakeshreddyravula9-lgtm/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (optional, for enhanced preprocessing)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Running the Application

1. **Start the Flask backend**
```bash
cd backend
python app.py
```
The API will be available at `http://localhost:5000`

2. **Open the frontend**

Simply open `frontend/index.html` in your web browser, or serve it using:
```bash
cd frontend
python -m http.server 8000
```
Then visit `http://localhost:8000`

## 📖 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Analyze Single Text
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "I absolutely love this product!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 87.5,
  "polarity": 0.625,
  "subjectivity": 0.8,
  "cleaned_text": "absolutely love product",
  "original_text": "I absolutely love this product!",
  "timestamp": "2025-11-06T10:30:00"
}
```

#### 2. Batch Analysis
```http
POST /api/batch
Content-Type: application/json

{
  "texts": [
    "This is amazing!",
    "I hate this",
    "It's okay"
  ]
}
```

#### 3. Health Check
```http
GET /api/health
```

#### 4. Model Stats
```http
GET /api/stats
```

## 🧪 Training Custom Model

To train your own sentiment analysis model:

```bash
cd backend/models
python train_model.py
```

This will:
1. Create sample training data
2. Train multiple ML models
3. Select the best performing model
4. Save the model and vectorizer

## 💡 Usage Examples

### Example 1: Positive Sentiment
**Input:** "This product is absolutely amazing! I love it so much!"

**Output:**
- Sentiment: Positive
- Confidence: 92.3%
- Polarity: 0.65

### Example 2: Negative Sentiment
**Input:** "This is the worst experience ever. Very disappointed."

**Output:**
- Sentiment: Negative
- Confidence: 88.7%
- Polarity: -0.75

### Example 3: Neutral Sentiment
**Input:** "It's okay, nothing special really."

**Output:**
- Sentiment: Neutral
- Confidence: 65.2%
- Polarity: 0.05

## 🎨 Screenshots

*(Add screenshots of your application here)*

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
```

### CORS Settings

To allow frontend from different origins, modify `app.py`:

```python
CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"])
```

## 📊 Model Performance

- **Accuracy**: ~85-90% on test data
- **Precision**: High for positive/negative sentiments
- **Recall**: Balanced across all categories
- **F1-Score**: Consistent performance

## 🚢 Deployment

### Deploy to Heroku

1. Create `Procfile`:
```
web: gunicorn backend.app:app
```

2. Deploy:
```bash
heroku create ai-sentiment-analyzer
git push heroku main
```

### Deploy to AWS/GCP

Use Docker for containerized deployment:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "backend/app.py"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rakesh Reddy Ravula**
- GitHub: [@rakeshreddyravula9-lgtm](https://github.com/rakeshreddyravula9-lgtm)
- Email: rakeshreddyravula9@gmail.com

## 🙏 Acknowledgments

- TextBlob for sentiment analysis
- Flask for the awesome web framework
- Scikit-learn for machine learning tools
- The open-source community

## 📈 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement LSTM/Transformer models
- [ ] Add sentiment trends visualization
- [ ] Create mobile app version
- [ ] Add user authentication
- [ ] Store analysis history
- [ ] Add export functionality (CSV, JSON)
- [ ] Integrate with social media APIs

## 🐛 Known Issues

None at the moment. Please report any issues in the [Issues](https://github.com/rakeshreddyravula9-lgtm/ai-sentiment-analyzer/issues) section.

## 📞 Support

For support, email rakeshreddyravula9@gmail.com or open an issue on GitHub.

---

⭐️ If you find this project helpful, please give it a star!
