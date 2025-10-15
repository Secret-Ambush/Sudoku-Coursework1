#!/bin/bash

# Deployment script for Sudoku Solver
echo "🚀 Deploying Sudoku Solver App..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Sudoku Solver with React + Python"
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No remote origin found. Please add your GitHub repository:"
    echo "git remote add origin https://github.com/yourusername/your-repo-name.git"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Deploy: Add deployment configuration"
git push origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "🎯 Next steps:"
echo "1. Deploy backend to Railway: https://railway.app"
echo "2. Deploy frontend to Vercel: https://vercel.com"
echo "3. Set environment variable REACT_APP_API_URL in Vercel"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
