#!/usr/bin/env python3
"""
CLI Demo for Pruning-Based Sudoku Solver

This script demonstrates the pruning-based Sudoku solver (from app1.py) with
performance tracking including time, number of backtracks, and assignments.

The solver uses basic backtracking with pruning techniques:
- Optimized cell finding
- Efficient validity checking
- No arc consistency (for comparison with AC-3 implementation)
"""

import time
import copy
from typing import List, Tuple, Optional, Dict, Any


class PruningSudokuSolver:
    """
    A Sudoku solver using backtracking with pruning techniques.
    
    This solver implements the same algorithm as app1.py but without pygame
    for CLI testing and performance analysis.
    """
    
    def __init__(self, grid: List[List[str]]):
        """
        Initialize the solver with a Sudoku grid.
        
        Args:
            grid: 9x9 grid where empty cells are represented as empty strings or '0'
        """
        self.grid = self._normalize_grid(grid)
        self.size = 9
        
        # Convert grid to integer representation for easier processing
        self.board = self._grid_to_board(self.grid)
        
        # Track original given cells
        self.given_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    self.given_cells.add((row, col))
        
        # Track statistics
        self.assignments = 0
        self.backtracks = 0
        self.start_time = None
        
    def _normalize_grid(self, grid: List[List[str]]) -> List[List[str]]:
        """Normalize grid input to handle various formats."""
        normalized = []
        for row in grid:
            normalized_row = []
            for cell in row:
                if cell in ['', '0', '0.0', None]:
                    normalized_row.append('')
                else:
                    normalized_row.append(str(cell).strip())
            normalized.append(normalized_row)
        return normalized
    
    def _grid_to_board(self, grid: List[List[str]]) -> List[List[int]]:
        """Convert string grid to integer board."""
        board = []
        for row in grid:
            board_row = []
            for cell in row:
                if cell == '':
                    board_row.append(0)
                else:
                    board_row.append(int(cell))
            board.append(board_row)
        return board
    
    def _board_to_grid(self, board: List[List[int]]) -> List[List[str]]:
        """Convert integer board back to string grid."""
        grid = []
        for row in board:
            grid_row = []
            for cell in row:
                if cell == 0:
                    grid_row.append('')
                else:
                    grid_row.append(str(cell))
            grid.append(grid_row)
        return grid
    
    def find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Find the next empty cell efficiently in a single pass."""
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    return (r, c)
        return None
    
    def is_valid_optimized(self, board: List[List[int]], row: int, col: int, num: int) -> bool:
        """Optimized validity check using efficient lookups."""
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve_sudoku_optimized(self) -> bool:
        """Optimized solve using backtracking with pruning."""
        # Find next empty cell
        empty = self.find_empty_cell()
        if not empty:
            return True  # Solved!
        
        row, col = empty
        
        for num in range(1, 10):
            if self.is_valid_optimized(self.board, row, col, num):
                self.board[row][col] = num
                self.assignments += 1
                
                if self.solve_sudoku_optimized():
                    return True
                
                # Backtrack
                self.board[row][col] = 0
                self.backtracks += 1
        
        return False
    
    def solve(self) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
        """
        Solve the Sudoku puzzle.
        
        Returns:
            Tuple of (success, solution_grid, statistics)
        """
        self.start_time = time.time()
        self.assignments = 0
        self.backtracks = 0
        
        # Solve using backtracking
        success = self.solve_sudoku_optimized()
        
        end_time = time.time()
        
        if success:
            solution_grid = self._board_to_grid(self.board)
        else:
            solution_grid = self.grid
        
        stats = {
            'assignments': self.assignments,
            'backtracks': self.backtracks,
            'time': end_time - self.start_time,
            'solvable': success
        }
        
        return success, solution_grid, stats
    
    def is_valid_sudoku(self, grid: Optional[List[List[str]]] = None) -> bool:
        """
        Validate if a Sudoku grid is valid (no conflicts).
        
        Args:
            grid: Grid to validate (uses current grid if None)
            
        Returns:
            True if valid, False otherwise
        """
        if grid is None:
            grid = self.grid
        
        board = self._grid_to_board(grid)
        
        # Check rows
        for row in board:
            values = [v for v in row if v != 0]
            if len(values) != len(set(values)):
                return False
        
        # Check columns
        for col in range(self.size):
            values = [board[row][col] for row in range(self.size) if board[row][col] != 0]
            if len(values) != len(set(values)):
                return False
        
        # Check boxes
        for box_row in range(0, self.size, 3):
            for box_col in range(0, self.size, 3):
                values = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        if board[r][c] != 0:
                            values.append(board[r][c])
                if len(values) != len(set(values)):
                    return False
        
        return True
    
    def print_grid(self, grid: Optional[List[List[str]]] = None, title: str = "Sudoku Grid"):
        """Print a Sudoku grid in a formatted way."""
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


def solve_sudoku_pruning(grid: List[List[str]]) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
    """
    Convenience function to solve a Sudoku grid using pruning-based backtracking.
    
    Args:
        grid: 9x9 Sudoku grid with empty cells as empty strings or '0'
        
    Returns:
        Tuple of (success, solution_grid, statistics)
    """
    solver = PruningSudokuSolver(grid)
    return solver.solve()


def create_test_puzzles():    
    # Easy puzzle
    easy_puzzle = [
        ["5", "3", "", "", "7", "", "", "", ""],
        ["6", "", "", "1", "9", "5", "", "", ""],
        ["", "9", "8", "", "", "", "", "6", ""],
        ["8", "", "", "", "6", "", "", "", "3"],
        ["4", "", "", "8", "", "3", "", "", "1"],
        ["7", "", "", "", "2", "", "", "", "6"],
        ["", "6", "", "", "", "", "2", "8", ""],
        ["", "", "", "4", "1", "9", "", "", "5"],
        ["", "", "", "", "8", "", "", "7", "9"]
    ]

    # Medium puzzle
    medium_puzzle = [
        ["", "6", "", "1", "", "4", "", "5", ""],
        ["", "", "8", "3", "", "5", "6", "", ""],
        ["2", "", "", "", "", "", "", "", "1"],
        ["8", "", "", "4", "", "7", "", "", "6"],
        ["", "", "6", "", "8", "", "3", "", ""],
        ["7", "", "", "9", "", "1", "", "", "4"],
        ["5", "", "", "", "", "", "", "", "2"],
        ["", "", "7", "2", "", "6", "9", "", ""],
        ["", "4", "", "5", "", "8", "", "7", ""]
    ]

    # Hard puzzle
    hard_puzzle = [
        ["", "", "5", "3", "", "", "", "", ""],
        ["8", "", "", "", "", "", "", "2", ""],
        ["", "7", "", "", "1", "", "5", "", ""],
        ["4", "", "", "", "", "5", "3", "", ""],
        ["", "1", "", "", "7", "", "", "", "6"],
        ["", "", "3", "2", "", "", "", "8", ""],
        ["", "6", "", "5", "", "", "", "", "9"],
        ["", "", "4", "", "", "", "", "3", ""],
        ["", "", "", "", "", "9", "7", "", ""]
    ]

    return {
        "Easy": easy_puzzle,
        "Medium": medium_puzzle,
        "Hard": hard_puzzle,
    }


def test_puzzle_solving():
    """Test the pruning-based solver on various puzzles."""
    print("=" * 60)
    print("PRUNING-BASED SUDOKU SOLVER DEMONSTRATION")
    print("=" * 60)
    
    puzzles = create_test_puzzles()
    
    for difficulty, puzzle in puzzles.items():
        print(f"\n{'='*20} {difficulty.upper()} PUZZLE {'='*20}")
        
        solver = PruningSudokuSolver(puzzle)
        
        # Display original puzzle
        solver.print_grid(puzzle, f"Original {difficulty} Puzzle")
        
        # Validate puzzle
        is_valid = solver.is_valid_sudoku()
        print(f"\nPuzzle validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
        
        if not is_valid:
            print("Skipping invalid puzzle...")
            continue
        
        # Solve puzzle
        print(f"\nSolving {difficulty.lower()} puzzle...")
        start_time = time.time()
        success, solution, stats = solver.solve()
        end_time = time.time()
        
        if success:
            solver.print_grid(solution, f"Solution for {difficulty} Puzzle")
            
            # Validate solution
            solution_valid = solver.is_valid_sudoku(solution)
            print(f"\nSolution validation: {'✓ Valid' if solution_valid else '✗ Invalid'}")
            
            # Display statistics
            print(f"\nSolving Statistics:")
            print(f"  • Total time: {stats['time']:.4f} seconds")
            print(f"  • Assignments made: {stats['assignments']}")
            print(f"  • Backtracks: {stats['backtracks']}")
            print(f"  • Success rate: {stats['assignments']/(stats['assignments']+stats['backtracks'])*100:.1f}%")
            
        else:
            print(f"✗ No solution found for {difficulty.lower()} puzzle!")
            print(f"Statistics:")
            print(f"  • Time: {stats['time']:.4f} seconds")
            print(f"  • Assignments: {stats['assignments']}")
            print(f"  • Backtracks: {stats['backtracks']}")


def performance_comparison():
    """Compare performance on different puzzle types."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON - PRUNING-BASED SOLVER")
    print(f"{'='*60}")
    
    puzzles = create_test_puzzles()
    results = []
    
    for difficulty, puzzle in puzzles.items():
        if difficulty == "Invalid":
            continue
            
        solver = PruningSudokuSolver(puzzle)
        
        if not solver.is_valid_sudoku():
            continue
        
        start_time = time.time()
        success, solution, stats = solver.solve()
        end_time = time.time()
        
        if success:
            results.append({
                'difficulty': difficulty,
                'time': stats['time'],
                'assignments': stats['assignments'],
                'backtracks': stats['backtracks'],
                'efficiency': stats['assignments'] / (stats['assignments'] + stats['backtracks'])
            })
    
    # Sort by time
    results.sort(key=lambda x: x['time'])
    
    print(f"{'Difficulty':<12} {'Time (s)':<10} {'Assignments':<12} {'Backtracks':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['difficulty']:<12} {result['time']:<10.4f} {result['assignments']:<12} {result['backtracks']:<10} {result['efficiency']:<10.2%}")


def compare_with_arc_consistency():
    """Compare pruning-based solver with arc consistency solver."""
    print(f"\n{'='*60}")
    print("COMPARISON: PRUNING vs ARC CONSISTENCY")
    print(f"{'='*60}")
    
    # Test puzzle
    test_puzzle = [
        ["5", "3", "", "", "7", "", "", "", ""],
        ["6", "", "", "1", "9", "5", "", "", ""],
        ["", "9", "8", "", "", "", "", "6", ""],
        ["8", "", "", "", "6", "", "", "", "3"],
        ["4", "", "", "8", "", "3", "", "", "1"],
        ["7", "", "", "", "2", "", "", "", "6"],
        ["", "6", "", "", "", "", "2", "8", ""],
        ["", "", "", "4", "1", "9", "", "", "5"],
        ["", "", "", "", "8", "", "", "7", "9"]
    ]
    
    print("Test Puzzle:")
    solver = PruningSudokuSolver(test_puzzle)
    solver.print_grid(test_puzzle, "Comparison Test Puzzle")
    
    # Test pruning-based solver
    print(f"\n{'='*30} PRUNING-BASED SOLVER {'='*30}")
    success_pruning, solution_pruning, stats_pruning = solver.solve()
    
    if success_pruning:
        print(f"✓ Solved successfully!")
        print(f"Time: {stats_pruning['time']:.4f} seconds")
        print(f"Assignments: {stats_pruning['assignments']}")
        print(f"Backtracks: {stats_pruning['backtracks']}")
        print(f"Efficiency: {stats_pruning['assignments']/(stats_pruning['assignments']+stats_pruning['backtracks'])*100:.1f}%")
    else:
        print("✗ Failed to solve")
    
    # Test arc consistency solver (if available)
    print(f"\n{'='*30} ARC CONSISTENCY SOLVER {'='*30}")
    import sys
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
    from ArcConsistency_Implementation.sudoku_solver import solve_sudoku
    
    success_ac, solution_ac, stats_ac = solve_sudoku(test_puzzle)
    
    if success_ac:
        print(f"✓ Solved successfully!")
        print(f"Time: {stats_ac['time']:.4f} seconds")
        print(f"Assignments: {stats_ac['assignments']}")
        print(f"Backtracks: {stats_ac['backtracks']}")
        denom = (stats_ac['assignments']+stats_ac['backtracks'])
        if denom!=0:
            print(f"Efficiency: {stats_ac['assignments']/denom*100:.1f}%")
        else:
            print(f"Efficiency: undefined")
    else:
        print("✗ Failed to solve")
    
    # Comparison summary
    print(f"\n{'='*30} COMPARISON SUMMARY {'='*30}")
    if success_pruning and success_ac:
        print(f"Both solvers found solutions!")
        print(f"Pruning solver: {stats_pruning['time']:.4f}s, {stats_pruning['assignments']} assignments")
        print(f"AC-3 solver:   {stats_ac['time']:.4f}s, {stats_ac['assignments']} assignments")
        
        if stats_pruning['time'] < stats_ac['time']:
            print(f"Pruning solver was {stats_ac['time']/stats_pruning['time']:.2f}x faster!")
        else:
            print(f"AC-3 solver was {stats_pruning['time']/stats_ac['time']:.2f}x faster!")

if __name__ == "__main__":
    try:
        # Run all demonstrations
        test_puzzle_solving()
        performance_comparison()
        compare_with_arc_consistency()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
