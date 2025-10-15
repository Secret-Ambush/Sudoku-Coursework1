import time

# ---- Sudoku board ----
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Statistics tracking
stats = {
    'backtracks': 0,
    'assignments': 0,
    'validity_checks': 0,
    'start_time': 0,
    'end_time': 0
}

def find_empty_cell():
    """Find the next empty cell"""
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def is_valid(board, row, col, num):
    """Check if placing num at (row, col) is valid"""
    stats['validity_checks'] += 1
    
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

def solve_sudoku():
    """Solve sudoku using backtracking with statistics"""
    # Find next empty cell
    empty = find_empty_cell()
    if not empty:
        return True  # Solved!
    
    row, col = empty
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            # Make assignment
            board[row][col] = num
            stats['assignments'] += 1
            
            if solve_sudoku():
                return True
            
            # Backtrack
            board[row][col] = 0
            stats['backtracks'] += 1
    
    return False

def print_board(board):
    """Pretty print the board"""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(board[i][j] if board[i][j] != 0 else ".", end=" ")
        print()

def print_statistics():
    """Print performance statistics"""
    elapsed_time = stats['end_time'] - stats['start_time']
    
    print("\n" + "="*50)
    print("SUDOKU SOLVER STATISTICS")
    print("="*50)
    print(f"Total Assignments:    {stats['assignments']}")
    print(f"Total Backtracks:     {stats['backtracks']}")
    print(f"Validity Checks:      {stats['validity_checks']}")
    print(f"Completion Time:      {elapsed_time:.6f} seconds")
    print(f"Time (milliseconds):  {elapsed_time * 1000:.2f} ms")
    print("="*50)
    
    # Calculate efficiency metrics
    if stats['assignments'] > 0:
        backtrack_ratio = stats['backtracks'] / stats['assignments']
        print(f"\nBacktrack Ratio:      {backtrack_ratio:.2f}")
        print(f"Checks per Assignment: {stats['validity_checks'] / stats['assignments']:.2f}")

def main():
    print("Original Sudoku Board:")
    print_board(board)
    
    print("\nSolving...")
    stats['start_time'] = time.time()
    
    if solve_sudoku():
        stats['end_time'] = time.time()
        print("\nSolved Sudoku Board:")
        print_board(board)
        print_statistics()
    else:
        stats['end_time'] = time.time()
        print("No solution exists!")
        print_statistics()

if __name__ == "__main__":
    main()