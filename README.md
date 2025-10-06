# Sudoku Playground UI

Prototype Streamlit app for experimenting with classic and killer Sudoku inputs.

## Features

- Upload 9x9 puzzles from `.csv` or `.txt` files (zero/blank denotes empty cells).
- Type directly into a stylised 9x9 grid with subgrid borders.
- Generate classic puzzles locally, tuned for easy/medium/hard clue counts, and preview a killer Sudoku scaffold.
- Preview panel renders the active grid with bold 3×3 blocks.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Puzzle Generation

`sudoku_generator.py` contains the backtracking-based generator used by the app. Tweak the clue targets there if you want to calibrate difficulty further.
