#!/usr/bin/env python3
"""
Simple usage example for the Sudoku Solver with Arc Consistency

This file demonstrates how to use the SudokuSolver class to solve Sudoku puzzles.
"""

from sudoku_solver import SudokuSolver, solve_sudoku


def main():
    """Simple example of using the Sudoku solver."""
    
    print("Sudoku Solver - Simple Usage Example")
    print("=" * 40)
    
    # Example puzzle (medium difficulty)
    puzzle = [
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
    
    # Method 1: Using the SudokuSolver class
    print("\nMethod 1: Using SudokuSolver class")
    print("-" * 30)
    
    solver = SudokuSolver(puzzle)
    
    # Display original puzzle
    solver.print_grid(puzzle, "Original Puzzle")
    
    # Solve the puzzle
    success, solution, stats = solver.solve()
    
    if success:
        solver.print_grid(solution, "Solution")
        print(f"\nSolved successfully!")
        print(f"Time: {stats['time']:.4f} seconds")
        print(f"Assignments: {stats['assignments']}")
        print(f"Backtracks: {stats['backtracks']}")
    else:
        print("No solution found!")
    
    # Method 2: Using the convenience function
    print("\n\nMethod 2: Using convenience function")
    print("-" * 30)
    
    success, solution, stats = solve_sudoku(puzzle)
    
    if success:
        print("✓ Puzzle solved successfully!")
        print(f"Statistics: {stats}")
    else:
        print("✗ Failed to solve puzzle")
    
    # Method 3: Validate a puzzle
    print("\n\nMethod 3: Puzzle validation")
    print("-" * 30)
    
    # Valid puzzle
    valid_puzzle = [
        ["5", "3", "4", "6", "7", "8", "9", "1", "2"],
        ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
        ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
        ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
        ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
        ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
        ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
        ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
        ["3", "4", "5", "2", "8", "6", "1", "7", "9"]
    ]
    
    solver_valid = SudokuSolver(valid_puzzle)
    is_valid = solver_valid.is_valid_sudoku()
    print(f"Valid puzzle check: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Invalid puzzle (conflicting)
    invalid_puzzle = [
        ["1", "1", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""]
    ]
    
    solver_invalid = SudokuSolver(invalid_puzzle)
    is_invalid = solver_invalid.is_valid_sudoku()
    print(f"Invalid puzzle check: {'✓ Valid' if is_invalid else '✗ Invalid'}")
    
    print(f"\n{'='*40}")
    print("Key Features of this Sudoku Solver:")
    print("• Backtracking with intelligent variable ordering (MRV)")
    print("• Value ordering using Least Constraining Value (LCV)")
    print("• Arc consistency (AC-3) for constraint propagation")
    print("• Forward checking to reduce search space")
    print("• Comprehensive validation and error handling")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
