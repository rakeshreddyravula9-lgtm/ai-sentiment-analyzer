# Quick Start Guide

## 🚀 Push to GitHub

Since the local repository is already initialized and committed, follow these steps:

### Option 1: Using the GitHub Website

1. **Create a new repository on GitHub:**
   - Go to: https://github.com/new
   - Repository name: `ai-sentiment-analyzer`
   - Description: `AI/ML full-stack sentiment analysis web application`
   - Visibility: **Public**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Push your code:**
   ```bash
   cd /home/rakesh/projects/ai-sentiment-analyzer
   git push -u origin main
   ```

### Option 2: Using GitHub CLI (requires gh installation)

```bash
# Install GitHub CLI
sudo apt install gh

# Authenticate
gh auth login

# Create and push repository
gh repo create ai-sentiment-analyzer --public --source=. --remote=origin --push
```

## 🏃 Running the Application

### 1. Install Dependencies

```bash
# Create virtual environment (if not already created)
cd /home/rakesh/projects/ai-sentiment-analyzer
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
cd backend
python app.py
```

Backend will run on: http://localhost:5000

### 3. Open Frontend

Open `frontend/index.html` in your web browser

Or serve it with:
```bash
cd frontend
python -m http.server 8000
```

Then visit: http://localhost:8000

## 🧪 Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Analyze sentiment
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## 📊 Train Custom Model (Optional)

```bash
cd backend/models
python train_model.py
```

## ✅ You're Done!

Your repository should now be live at:
https://github.com/rakeshreddyravula9-lgtm/ai-sentiment-analyzer

Add this link to your resume and LinkedIn! 🎉
