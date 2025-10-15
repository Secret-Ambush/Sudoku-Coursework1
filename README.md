# 🧩 Sudoku Solver - React + Python Flask

Application for experimenting with classic easy, medium and hard 9x9 Sudoku grids.

## Features

- **Interactive Sudoku Grid**: Click and type directly into cells, just like the image you wanted!
- **AI-Powered Solving**: Uses Arc Consistency and Backtracking with Pruning algorithms 
- **Puzzle Generation**: Generate puzzles of different difficulties (Easy, Medium, Hard)
- **Real-time Validation**: Check your solution as you play

## Getting Started

1. Run the setup script:
   ```bash
   ./setup.sh
   ```

2. Start the backend (Terminal 1):
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   ```

3. Start the frontend (Terminal 2):
   ```bash
   cd frontend
   npm start
   ```
