#!/bin/bash

# Setup script for Sudoku React + Python Flask app

echo "🧩 Setting up Sudoku Solver App..."

# Check if we're in the right directory
if [ ! -f "sudoku_generator.py" ]; then
    echo "❌ Please run this script from the Sudoku project root directory"
    exit 1
fi

# Setup Python backend
echo "📦 Setting up Python backend..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python backend setup complete!"

# Setup React frontend
echo "📦 Setting up React frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

echo "✅ React frontend setup complete!"

echo ""
echo "🚀 Setup complete! To run the application:"
echo ""
echo "1. Start the Python backend:"
echo "   cd backend && source venv/bin/activate && python app.py"
echo ""
echo "2. In a new terminal, start the React frontend:"
echo "   cd frontend && npm start"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "The backend will run on http://localhost:5000"
echo "The frontend will run on http://localhost:3000"
