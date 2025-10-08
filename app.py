"""Streamlit UI scaffold for Sudoku inputs."""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from sudoku_generator import generate_sudoku

Grid = List[List[str]]


st.set_page_config(page_title="Sudoku Playground", layout="wide", page_icon="🎯")


# --- CSS helpers -----------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --sudoku-border-regular: 1px solid #7f8c8d;
        --sudoku-border-strong: 2px solid #2c3e50;
        --sudoku-cell-size: 48px;
    }
    .sudoku-grid {
        display: grid;
        grid-template-columns: repeat(9, var(--sudoku-cell-size));
        grid-auto-rows: var(--sudoku-cell-size);
        gap: 0;
        margin: 0 auto;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border: var(--sudoku-border-strong);
        border-left: none;
        border-top: none;
    }
    .sudoku-cell {
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: "DM Mono", "Fira Code", monospace;
        font-size: 1.25rem;
        font-weight: 500;
        color: #2c3e50;
        background: #fafafa;
    }
    .sudoku-cell:nth-child(odd) {
        background: #fefefe;
    }
    .sudoku-cell:nth-child(even) {
        background: #f6f8fa;
    }
    .sudoku-cell::selection {
        background: #1abc9c;
        color: white;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        text-align: center;
        font-family: "DM Mono", "Fira Code", monospace;
        font-size: 1rem;
        border-radius: 6px;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Utility functions -----------------------------------------------------
def create_empty_grid() -> Grid:
    """Return a blank 9x9 grid."""

    return [["" for _ in range(9)] for _ in range(9)]


def normalise_cell(raw_value: str) -> str:
    """Normalise cell content coming from uploads/forms."""

    if raw_value is None:
        return ""
    text = str(raw_value).strip()
    if text.lower() in {"", "nan", "none", "0", "0.0"}:
        return ""
    if not re.fullmatch(r"[1-9]", text):
        raise ValueError("Cells must contain digits 1-9 or be left blank")
    return text


def render_grid(grid: Grid) -> None:
    """Render a Sudoku grid using HTML/CSS for a premium look."""

    cells = []
    for row_idx, row in enumerate(grid):
        for col_idx, value in enumerate(row):
            borders = []
            # Horizontal borders
            if row_idx % 3 == 0:
                borders.append("border-top: var(--sudoku-border-strong)")
            else:
                borders.append("border-top: var(--sudoku-border-regular)")
            if row_idx == 8:
                borders.append("border-bottom: var(--sudoku-border-strong)")
            elif (row_idx + 1) % 3 == 0:
                borders.append("border-bottom: var(--sudoku-border-strong)")
            else:
                borders.append("border-bottom: var(--sudoku-border-regular)")

            # Vertical borders
            if col_idx % 3 == 0:
                borders.append("border-left: var(--sudoku-border-strong)")
            else:
                borders.append("border-left: var(--sudoku-border-regular)")
            if col_idx == 8:
                borders.append("border-right: var(--sudoku-border-strong)")
            elif (col_idx + 1) % 3 == 0:
                borders.append("border-right: var(--sudoku-border-strong)")
            else:
                borders.append("border-right: var(--sudoku-border-regular)")

            display_value = value if value else "&nbsp;"
            cell_style = "; ".join(borders)
            cells.append(
                f'<div class="sudoku-cell" style="{cell_style}">{display_value}</div>'
            )

    st.markdown(
        f'<div class="sudoku-grid">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def parse_uploaded_file(upload) -> Grid:
    """Parse CSV/TXT uploads into a Sudoku grid."""

    suffix = Path(upload.name).suffix.lower()
    content = upload.getvalue().decode("utf-8")

    if suffix == ".csv":
        df = pd.read_csv(StringIO(content), header=None, nrows=9)
    else:
        df = pd.read_csv(
            StringIO(content),
            header=None,
            nrows=9,
            sep=r"[\s,;]+",
            engine="python",
        )

    if df.shape[0] < 9 or df.shape[1] < 9:
        raise ValueError("Expected a 9x9 grid. Check the uploaded file content.")

    grid = create_empty_grid()
    for i in range(9):
        for j in range(9):
            grid[i][j] = normalise_cell(df.iat[i, j])
    return grid


def render_manual_input(default: Optional[Grid] = None) -> Grid:
    """Render a 9x9 manual entry form and return the captured grid."""

    captured: Grid = []
    default = default or create_empty_grid()
    for row_idx in range(9):
        cols = st.columns(9, gap="small")
        row: List[str] = []
        for col_idx in range(9):
            default_value = default[row_idx][col_idx]
            cell_value = cols[col_idx].text_input(
                label=f"r{row_idx + 1}c{col_idx + 1}",
                value=default_value,
                max_chars=1,
                key=f"manual_{row_idx}_{col_idx}",
            )
            row.append(cell_value)
        captured.append(row)
    return captured


CLASSIC_GENERATOR_SOURCE = "Generated • Classic • {difficulty} • Local generator"


# --- Session state bootstrap ----------------------------------------------
def ensure_session_state() -> None:
    if "current_puzzle" not in st.session_state:
        st.session_state["current_puzzle"] = None
    if "puzzle_source" not in st.session_state:
        st.session_state["puzzle_source"] = None


ensure_session_state()


# --- UI -------------------------------------------------------------------
st.title("🧩 Sudoku Playground")
st.caption("Prototype UI for classic Sudoku inputs")

upload_tab, manual_tab, generate_tab = st.tabs([
    "Upload a puzzle",
    "Fill in the grid",
    "Generate for me",
])


with upload_tab:
    st.subheader("Upload a CSV or TXT")
    st.write(
        "Use zeros or blanks for empty cells. CSV should not include headers; TXT can be whitespace or comma separated."
    )
    uploaded_file = st.file_uploader("Choose a puzzle file", type=["csv", "txt"])
    if uploaded_file:
        try:
            puzzle_grid = parse_uploaded_file(uploaded_file)
        except ValueError as exc:
            st.error(f"{exc}")
        else:
            st.session_state["current_puzzle"] = puzzle_grid
            st.session_state["puzzle_source"] = f"Uploaded • {uploaded_file.name}"
            st.success("Nice! Scroll down to preview your puzzle.")


with manual_tab:
    st.subheader("Click into the cells to type numbers")
    st.write(
        "Leave cells blank for unknown digits. The form stores your input when you hit the button."
    )
    with st.form("manual-entry-form"):
        entered_grid = render_manual_input()
        submitted = st.form_submit_button("Use this puzzle")
    if submitted:
        try:
            sanitized_grid: Grid = []
            for row in entered_grid:
                sanitized_row = [normalise_cell(cell) if cell else "" for cell in row]
                sanitized_grid.append(sanitized_row)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["current_puzzle"] = sanitized_grid
            st.session_state["puzzle_source"] = "Manual entry"
            st.success("Got it! Puzzle updated below.")


with generate_tab:
    st.subheader("Grab a generated puzzle")
    generator_col, preview_col = st.columns([1.3, 1])

    with generator_col:
        difficulty = st.select_slider(
            "Select difficulty",
            options=["Easy", "Medium", "Hard"],
            value="Easy",
        )
        generate_button = st.button("✨ Generate", type="primary")

    with preview_col:
        st.markdown("**Latest selection**")
        st.markdown(
            f"- Style: Classic Sudoku\n- Difficulty: {difficulty}"
        )

    if generate_button:
        with st.spinner("Generating puzzle..."):
            chosen_grid = generate_sudoku(difficulty)

        st.session_state["current_puzzle"] = chosen_grid
        st.session_state["puzzle_source"] = CLASSIC_GENERATOR_SOURCE.format(
            difficulty=difficulty
        )
        st.success("Classic puzzle generated locally! Preview below.")


# --- Preview & Meta -------------------------------------------------------
st.divider()

current = st.session_state.get("current_puzzle")
source = st.session_state.get("puzzle_source")

if current:
    st.subheader("Puzzle preview")
    if source:
        st.caption(source)
    render_grid(current)
else:
    st.info("Upload, enter, or generate a puzzle to see it rendered here.")


st.divider()

st.markdown(
    "Need a solver or API integration next? Let's tackle that after we lock in the UI experience."
)
