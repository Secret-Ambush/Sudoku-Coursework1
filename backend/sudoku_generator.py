"""Sudoku puzzle generator with basic difficulty tuning."""
from __future__ import annotations

import random
from typing import List

Grid = List[List[str]]
Board = List[List[int]]

_DIFFICULTY_TARGETS = {"Easy": 40, "Medium": 32, "Hard": 28}


def _is_valid_move(board: Board, row: int, col: int, num: int) -> bool:
    """Return True if num can be placed safely at board[row][col]."""

    if any(board[row][c] == num for c in range(9)):
        return False
    if any(board[r][col] == num for r in range(9)):
        return False
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True


def _generate_solved_board() -> Board:
    """Generate a fully solved Sudoku board using backtracking."""

    board: Board = [[0 for _ in range(9)] for _ in range(9)]

    def fill(cell_index: int = 0) -> bool:
        if cell_index == 81:
            return True
        row, col = divmod(cell_index, 9)
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        for number in numbers:
            if _is_valid_move(board, row, col, number):
                board[row][col] = number
                if fill(cell_index + 1):
                    return True
                board[row][col] = 0
        return False

    fill()
    return board


def _count_solutions(board: Board, limit: int = 2) -> int:
    """Count valid solutions for board up to the provided limit."""

    solutions = 0

    def backtrack() -> bool:
        nonlocal solutions
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for number in range(1, 10):
                        if _is_valid_move(board, row, col, number):
                            board[row][col] = number
                            finished = backtrack()
                            board[row][col] = 0
                            if finished:
                                return True
                    return False
        solutions += 1
        return solutions >= limit

    backtrack()
    return solutions


def generate_sudoku(difficulty: str = "Easy") -> tuple[Grid, Grid]:
    """Return a Sudoku puzzle and its solution tuned to the given difficulty."""

    label = difficulty.title()
    target_clues = _DIFFICULTY_TARGETS.get(label, _DIFFICULTY_TARGETS["Easy"])

    solved = _generate_solved_board()
    puzzle: Board = [row[:] for row in solved]

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row in range(9):
        for col in range(9):
            mirror = (8 - row, 8 - col)
            if (row, col) <= mirror:
                pairs.append(((row, col), mirror))

    random.shuffle(pairs)

    def clues_count() -> int:
        return sum(1 for row in puzzle for value in row if value != 0)

    for (r1, c1), (r2, c2) in pairs:
        if puzzle[r1][c1] == 0 and puzzle[r2][c2] == 0:
            continue

        removed_cells = {(r1, c1)}
        if (r1, c1) != (r2, c2):
            removed_cells.add((r2, c2))

        if clues_count() - len(removed_cells) < target_clues:
            continue

        backups = {coords: puzzle[coords[0]][coords[1]] for coords in removed_cells}
        for rr, cc in removed_cells:
            puzzle[rr][cc] = 0

        puzzle_copy: Board = [row[:] for row in puzzle]
        if _count_solutions(puzzle_copy) != 1:
            for coords, value in backups.items():
                puzzle[coords[0]][coords[1]] = value

        if clues_count() <= target_clues:
            break

    grid: Grid = [[str(value) if value else "" for value in row] for row in puzzle]
    solution: Grid = [[str(value) for value in row] for row in solved]
    return grid, solution
