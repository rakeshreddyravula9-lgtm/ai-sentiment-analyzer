#!/bin/bash

# Setup script for AI Sentiment Analyzer
echo "=================================="
echo "AI Sentiment Analyzer Setup"
echo "=================================="

# Step 1: Create GitHub repository
echo ""
echo "Step 1: Create GitHub Repository"
echo "--------------------------------"
echo "Please go to: https://github.com/new"
echo "Repository name: ai-sentiment-analyzer"
echo "Description: AI/ML full-stack sentiment analysis web application"
echo "Visibility: Public"
echo "DO NOT initialize with README, .gitignore, or license"
echo ""
read -p "Press Enter after creating the repository on GitHub..."

# Step 2: Push code to GitHub
echo ""
echo "Step 2: Pushing code to GitHub..."
echo "--------------------------------"
git push -u origin main

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. View your repository at: https://github.com/rakeshreddyravula9-lgtm/ai-sentiment-analyzer"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Run the backend: cd backend && python app.py"
echo "4. Open frontend: Open frontend/index.html in your browser"
echo ""
