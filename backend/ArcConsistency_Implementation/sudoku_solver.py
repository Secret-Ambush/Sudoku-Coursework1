"""
Sudoku Solver using Backtracking with Maintain Arc Consistency (MAC-3)
Refactored for efficiency — same API and outputs as original version.
"""

from __future__ import annotations
import copy
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import time


class SudokuSolver:
    def __init__(self, grid: List[List[str]]):
        self.grid = self._normalize_grid(grid)
        self.size = 9
        self.box_size = 3

        self.board = self._grid_to_board(self.grid)
        self.given_cells = {(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] != 0}

        self.domains = self._initialize_domains()
        self.neighbours = self._build_neighbour_map()

        self.assignments = 0
        self.backtracks = 0
        self.start_time = None

    # ------------------------------
    # Setup and utility functions
    # ------------------------------

    def _normalize_grid(self, grid: List[List[str]]) -> List[List[str]]:
        normalized = []
        for row in grid:
            normalized.append(['' if cell in ['', '0', '0.0', None] else str(cell).strip() for cell in row])
        return normalized

    def _grid_to_board(self, grid: List[List[str]]) -> List[List[int]]:
        return [[int(cell) if cell != '' else 0 for cell in row] for row in grid]

    def _board_to_grid(self, board: List[List[int]]) -> List[List[str]]:
        return [[str(cell) if cell != 0 else '' for cell in row] for row in board]

    def _initialize_domains(self) -> Dict[Tuple[int, int], Set[int]]:
        domains = {}
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {self.board[r][c]}
        return domains

    def _build_neighbour_map(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Precompute all neighbours for each cell (row, column, box)."""
        neighbours = defaultdict(set)
        for r in range(self.size):
            for c in range(self.size):
                # Same row and column
                for i in range(self.size):
                    if i != c:
                        neighbours[(r, c)].add((r, i))
                    if i != r:
                        neighbours[(r, c)].add((i, c))
                # Same box
                box_r, box_c = r // 3, c // 3
                for rr in range(box_r * 3, (box_r + 1) * 3):
                    for cc in range(box_c * 3, (box_c + 1) * 3):
                        if (rr, cc) != (r, c):
                            neighbours[(r, c)].add((rr, cc))
        return dict(neighbours)

    # ------------------------------
    # Constraint utilities
    # ------------------------------

    def _revise(self, xi: Tuple[int, int], xj: Tuple[int, int],
                domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        """Revise domain of xi to ensure arc consistency wrt xj."""
        revised = False
        to_remove = set()
        for x in domains[xi]:
            # x is consistent if there exists some y in domain[xj] != x
            if not any(x != y for y in domains[xj]):
                to_remove.add(x)
        if to_remove:
            domains[xi] -= to_remove
            revised = True
        return revised

    def _ac3(self, queue: deque, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        """Perform AC-3 on the current queue only (local MAC propagation)."""
        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj, domains):
                if not domains[xi]:
                    return False
                for xk in self.neighbours[xi] - {xj}:
                    queue.append((xk, xi))
        return True

    # ------------------------------
    # Heuristics
    # ------------------------------

    def _select_unassigned_variable(self, domains: Dict[Tuple[int, int], Set[int]]) -> Optional[Tuple[int, int]]:
        unassigned = [(cell, len(dom)) for cell, dom in domains.items() if len(dom) > 1]
        if not unassigned:
            return None
        return min(unassigned, key=lambda x: x[1])[0]

    def _order_domain_values(self, var: Tuple[int, int],
                             domains: Dict[Tuple[int, int], Set[int]]) -> List[int]:
        """Least Constraining Value heuristic."""
        def count_constraints(val):
            count = 0
            for n in self.neighbours[var]:
                if val in domains[n]:
                    count += 1
            return count
        return sorted(domains[var], key=count_constraints)

    def _is_complete(self, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        return all(len(dom) == 1 for dom in domains.values())

    # ------------------------------
    # Backtracking with MAC
    # ------------------------------

    def _backtrack(self, domains: Dict[Tuple[int, int], Set[int]]) -> Optional[Dict[Tuple[int, int], Set[int]]]:
        if self._is_complete(domains):
            return domains

        var = self._select_unassigned_variable(domains)
        if var is None:
            return None

        for val in self._order_domain_values(var, domains):
            self.assignments += 1
            new_domains = {k: set(v) for k, v in domains.items()}  # shallow copy only
            new_domains[var] = {val}

            # Build local queue: all arcs (neighbour, var)
            queue = deque((n, var) for n in self.neighbours[var])
            if self._ac3(queue, new_domains):
                result = self._backtrack(new_domains)
                if result is not None:
                    return result
            self.backtracks += 1

        return None

    # ------------------------------
    # Solver entry points
    # ------------------------------

    def solve(self) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
        self.start_time = time.time()
        self.assignments = 0
        self.backtracks = 0

        # Initial MAC propagation
        queue = deque()
        for xi in self.domains:
            for xj in self.neighbours[xi]:
                queue.append((xi, xj))
        if not self._ac3(queue, self.domains):
            return False, self.grid, {
                "assignments": self.assignments,
                "backtracks": self.backtracks,
                "time": time.time() - self.start_time,
                "solvable": False
            }

        result = self._backtrack(self.domains)
        end = time.time()

        if result is None:
            return False, self.grid, {
                "assignments": self.assignments,
                "backtracks": self.backtracks,
                "time": end - self.start_time,
                "solvable": False
            }

        solution_board = [[list(result[(r, c)])[0] for c in range(self.size)] for r in range(self.size)]
        solution_grid = self._board_to_grid(solution_board)

        stats = {
            "assignments": self.assignments,
            "backtracks": self.backtracks,
            "time": end - self.start_time,
            "solvable": True
        }
        return True, solution_grid, stats

    # ------------------------------
    # Helpers for validation/printing
    # ------------------------------

    def is_valid_sudoku(self, grid: Optional[List[List[str]]] = None) -> bool:
        if grid is None:
            grid = self.grid
        board = self._grid_to_board(grid)
        for r in range(self.size):
            vals = [v for v in board[r] if v != 0]
            if len(vals) != len(set(vals)):
                return False
        for c in range(self.size):
            vals = [board[r][c] for r in range(self.size) if board[r][c] != 0]
            if len(vals) != len(set(vals)):
                return False
        for br in range(0, self.size, 3):
            for bc in range(0, self.size, 3):
                vals = []
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        if board[r][c] != 0:
                            vals.append(board[r][c])
                if len(vals) != len(set(vals)):
                    return False
        return True

    def print_grid(self, grid: Optional[List[List[str]]] = None, title: str = "Sudoku Grid"):
        if grid is None:
            grid = self.grid
        print(f"\n{title}:")
        print("+" + "-" * 25 + "+")
        for i, row in enumerate(grid):
            print("|", end="")
            for j, cell in enumerate(row):
                if j % 3 == 0 and j > 0:
                    print(" |", end="")
                print(f" {cell if cell else '.'}", end="")
            print(" |")
            if (i + 1) % 3 == 0 and i < 8:
                print("|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7 + "|")
        print("+" + "-" * 25 + "+")


def solve_sudoku(grid: List[List[str]]) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
    solver = SudokuSolver(grid)
    return solver.solve()
