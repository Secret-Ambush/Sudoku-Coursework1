# 🧩 Sudoku Solver

Interactive web app for experimenting with 9x9 Sudoku puzzles. The project couples a React front end with a Python Flask API so you can play, validate, generate, and automatically solve puzzles while comparing multiple AI strategies.

## Key Features
- Play through easy, medium, or hard classic Sudoku boards with instant validation feedback.
- Generate fresh puzzles or upload your own grid via CSV.
- Solve puzzles with either Arc Consistency (AC-3) or Backtracking with Pruning and compare runtime statistics.
- Inspect algorithm metrics such as assignments explored, backtracks, and total solve time.

## Live Deployment
- UI: https://sudoku-coursework1.vercel.app/
- API: Hosted on Railway, serving the Flask backend.

## Project Structure
- `frontend/`: React app bootstrapped with Create React App.
- `backend/`: Flask API exposing puzzle generation, validation, and solver endpoints.
  - `ArcConsistency_Implementation/`: AC-3 solver.
  - `Pruning_Implementation/`: Backtracking with pruning solver.
  - `sudoku_generator.py`: Difficulty-aware puzzle generator.

## Getting Started Locally

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ with `pip`

### 1. Start the backend API
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
The API listens on `http://localhost:8000` by default (configurable via the `PORT` environment variable).

### 2. Start the frontend
```bash
cd frontend
npm install
npm start
```
The React dev server runs on `http://localhost:3000` and proxies API calls to the Flask service.

## API Overview
- `POST /api/solve` – Accepts a 9x9 grid and returns solutions plus performance statistics for both algorithms.
- `POST /api/generate` – Creates a puzzle/solution pair for the requested difficulty (`Easy`, `Medium`, or `Hard`).
- `POST /api/validate` – Confirms whether a partially filled grid complies with Sudoku rules.
- `POST /api/upload` – Parses an uploaded CSV file into a grid and validates its structure.
- `GET /api/health` – Lightweight service health check.

All grid payloads are represented as 9 arrays of 9 strings, with empty cells denoted by `""`.

## Deployment Notes
- The production UI is deployed on Vercel.
- The Flask API is containerised and deployed to Railway, exposing the same routes used in development.

## Interface
<img width="714" height="689" alt="Screenshot 2025-10-22 at 3 37 29 pm" src="https://github.com/user-attachments/assets/53a3692e-c741-41d0-8da4-8c32671c9e9c" />
<img width="709" height="759" alt="Screenshot 2025-10-22 at 3 37 03 pm" src="https://github.com/user-attachments/assets/eedafa0f-b3e3-4f80-b70a-262fe274583e" />
