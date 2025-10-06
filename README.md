# Sudoku Playground UI

Prototype Streamlit app for experimenting with classic and killer Sudoku inputs.

## Features

- Upload 9x9 puzzles from `.csv` or `.txt` files (zero/blank denotes empty cells).
- Type directly into a stylised 9x9 grid with subgrid borders.
- Generate classic puzzles by calling a public API (with automatic sample fallback if offline) and preview a killer Sudoku scaffold.
- Preview panel renders the active grid with bold 3×3 blocks.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Puzzle API

The classic generator calls `https://sugoku.onrender.com/board` (and falls back to `https://sugoku.herokuapp.com/board`). If both endpoints fail, the app swaps in a bundled sample puzzle and shows the failure details so you can retry later.

The solver and additional puzzle APIs will be integrated in upcoming iterations.
