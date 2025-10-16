#!/usr/bin/env python3
"""
Simple usage example for the Pruning-Based Sudoku Solver

This file demonstrates how to use the PruningSudokuSolver class to solve Sudoku puzzles
using the same algorithm as app1.py but without pygame for CLI testing.
"""

from pruning_demo import PruningSudokuSolver, solve_sudoku_pruning


def main():
    """Simple example of using the pruning-based Sudoku solver."""
    
    print("Pruning-Based Sudoku Solver - Simple Usage Example")
    print("=" * 50)
    
    # Example puzzle (same as app1.py)
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
    
    # Method 1: Using the PruningSudokuSolver class
    print("\nMethod 1: Using PruningSudokuSolver class")
    print("-" * 40)
    
    solver = PruningSudokuSolver(puzzle)
    
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
        print(f"Efficiency: {stats['assignments']/(stats['assignments']+stats['backtracks'])*100:.1f}%")
    else:
        print("No solution found!")
    
    # Method 2: Using the convenience function
    print("\n\nMethod 2: Using convenience function")
    print("-" * 40)
    
    success, solution, stats = solve_sudoku_pruning(puzzle)
    
    if success:
        print("✓ Puzzle solved successfully!")
        print(f"Statistics: {stats}")
    else:
        print("✗ Failed to solve puzzle")
    
    # Method 3: Validate a puzzle
    print("\n\nMethod 3: Puzzle validation")
    print("-" * 40)
    
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
    
    solver_valid = PruningSudokuSolver(valid_puzzle)
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
    
    solver_invalid = PruningSudokuSolver(invalid_puzzle)
    is_invalid = solver_invalid.is_valid_sudoku()
    print(f"Invalid puzzle check: {'✓ Valid' if is_invalid else '✗ Invalid'}")
    
    # Method 4: Performance comparison
    print("\n\nMethod 4: Performance comparison")
    print("-" * 40)
    
    test_puzzles = {
        "Easy": puzzle,
        "Medium": [
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "3", "", "8", "5"],
            ["", "", "", "", "1", "", "", "", ""],
            ["", "", "", "", "8", "", "", "", ""],
            ["", "", "", "", "", "1", "", "", ""],
            ["", "", "", "", "2", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""]
        ]
    }
    
    print(f"{'Puzzle':<10} {'Time (s)':<10} {'Assignments':<12} {'Backtracks':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    for name, test_puzzle in test_puzzles.items():
        solver_test = PruningSudokuSolver(test_puzzle)
        success, solution, stats = solver_test.solve()
        
        if success:
            efficiency = stats['assignments'] / (stats['assignments'] + stats['backtracks']) * 100
            print(f"{name:<10} {stats['time']:<10.4f} {stats['assignments']:<12} {stats['backtracks']:<10} {efficiency:<10.1f}%")
    
    print(f"\n{'='*50}")
    print("Key Features of this Pruning-Based Solver:")
    print("• Backtracking with optimized cell finding")
    print("• Efficient validity checking using Python's 'in' operator")
    print("• No constraint propagation (pure backtracking)")
    print("• Performance optimized for speed")
    print("• Same algorithm as app1.py but without pygame")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
