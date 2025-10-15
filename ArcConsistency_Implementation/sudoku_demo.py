#!/usr/bin/env python3
"""
Demo script for Sudoku Solver with Arc Consistency

This script demonstrates the capabilities of the Sudoku solver by:
1. Testing various difficulty levels
2. Comparing performance with and without arc consistency
3. Showing step-by-step solving process
4. Validating solutions
"""

import time
from sudoku_solver import SudokuSolver, solve_sudoku


def create_test_puzzles():
    """Create various test puzzles of different difficulties."""
    
    # Easy puzzle (many given numbers)
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
    
    # Hard puzzle (few given numbers)
    hard_puzzle = [
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""]
    ]
    
    # Very hard puzzle (minimal clues)
    very_hard_puzzle = [
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""]
    ]
    
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
    
    return {
        "Easy": easy_puzzle,
        "Medium": medium_puzzle,
        "Hard": hard_puzzle,
        "Very Hard": very_hard_puzzle,
        "Invalid": invalid_puzzle
    }


def test_puzzle_solving():
    """Test the solver on various puzzles."""
    print("=" * 60)
    print("SUDOKU SOLVER DEMONSTRATION")
    print("=" * 60)
    
    puzzles = create_test_puzzles()
    
    for difficulty, puzzle in puzzles.items():
        print(f"\n{'='*20} {difficulty.upper()} PUZZLE {'='*20}")
        
        solver = SudokuSolver(puzzle)
        
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


def demonstrate_arc_consistency():
    """Demonstrate the power of arc consistency."""
    print(f"\n{'='*60}")
    print("ARC CONSISTENCY DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create a puzzle where arc consistency helps significantly
    puzzle = [
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
    
    solver = SudokuSolver(puzzle)
    
    print("Initial domains after arc consistency:")
    print("-" * 40)
    
    for row in range(9):
        for col in range(9):
            if len(solver.domains[(row, col)]) > 1:
                print(f"Cell ({row},{col}): {sorted(solver.domains[(row, col)])}")
    
    print(f"\nTotal empty cells: {sum(1 for domain in solver.domains.values() if len(domain) > 1)}")
    
    # Solve and show how arc consistency reduces search space
    success, solution, stats = solver.solve()
    
    if success:
        print(f"\nArc consistency reduced search space significantly!")
        print(f"Final statistics:")
        print(f"  • Assignments: {stats['assignments']}")
        print(f"  • Backtracks: {stats['backtracks']}")
        print(f"  • Time: {stats['time']:.4f} seconds")


def performance_comparison():
    """Compare performance on different puzzle types."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    puzzles = create_test_puzzles()
    results = []
    
    for difficulty, puzzle in puzzles.items():
        if difficulty == "Invalid":
            continue
            
        solver = SudokuSolver(puzzle)
        
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


def interactive_demo():
    """Interactive demo where user can input their own puzzle."""
    print(f"\n{'='*60}")
    print("INTERACTIVE DEMO")
    print(f"{'='*60}")
    
    print("Enter your Sudoku puzzle (9x9 grid):")
    print("Use numbers 1-9 for given cells, empty string or 0 for empty cells")
    print("Enter each row separated by commas, or press Enter for empty row")
    
    puzzle = []
    for i in range(9):
        while True:
            try:
                row_input = input(f"Row {i+1}: ").strip()
                if not row_input:
                    row = [""] * 9
                else:
                    row = [cell.strip() for cell in row_input.split(",")]
                    if len(row) != 9:
                        print("Please enter exactly 9 cells per row")
                        continue
                puzzle.append(row)
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                return
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again")
    
    solver = SudokuSolver(puzzle)
    
    print(f"\nYour puzzle:")
    solver.print_grid(puzzle, "Your Puzzle")
    
    if not solver.is_valid_sudoku():
        print("✗ Invalid puzzle! Please check for conflicts.")
        return
    
    print("\nSolving...")
    success, solution, stats = solver.solve()
    
    if success:
        solver.print_grid(solution, "Solution")
        print(f"\nSolved in {stats['time']:.4f} seconds!")
        print(f"Made {stats['assignments']} assignments with {stats['backtracks']} backtracks")
    else:
        print("✗ No solution found!")


if __name__ == "__main__":
    try:
        # Run all demonstrations
        test_puzzle_solving()
        demonstrate_arc_consistency()
        performance_comparison()
        
        # Ask if user wants interactive demo
        print(f"\n{'='*60}")
        response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo()
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETE!")
        print("The Sudoku solver uses:")
        print("• Backtracking with intelligent variable ordering (MRV)")
        print("• Value ordering using Least Constraining Value (LCV)")
        print("• Arc consistency (AC-3) for constraint propagation")
        print("• Forward checking to reduce search space")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
