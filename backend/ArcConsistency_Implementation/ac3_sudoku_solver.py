"""
Sudoku Solver using Backtracking with Arc Consistency

- Backtracking with intelligent variable ordering (MRV heuristic)
- Value ordering using Least Constraining Value (LCV) heuristic
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Set, Tuple, Optional, Any
import time


class SudokuSolver:
    def __init__(self, grid: List[List[str]]):
        """
        Initialize the solver with a 9x9 Sudoku grid.
        """
        self.grid = self._normalize_grid(grid)
        self.size = 9
        self.box_size = 3
        
        # Convert grid to integer representation for easier processing
        self.board = self._grid_to_board(self.grid)
        
        # Track original given cells
        self.given_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    self.given_cells.add((row, col))
        
        # Initialize domains for each cell
        self.domains = self._initialize_domains()
        
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
    
    def _initialize_domains(self) -> Dict[Tuple[int, int], Set[int]]:
        """Initialize domains for all empty cells."""
        domains = {}
        
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:  # Empty cell
                    domains[(row, col)] = set(range(1, 10))
                else:  # Given cell
                    domains[(row, col)] = {self.board[row][col]}
        
        return domains
    
    def _get_constraints(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        List of constraint pairs (variable1, variable2) where variable1 != variable2
        """
        constraints = []
        
        # Row constraints
        for row in range(self.size):
            cells_in_row = [(row, col) for col in range(self.size)]
            for i in range(len(cells_in_row)):
                for j in range(i + 1, len(cells_in_row)):
                    constraints.append((cells_in_row[i], cells_in_row[j]))
        
        # Column constraints
        for col in range(self.size):
            cells_in_col = [(row, col) for row in range(self.size)]
            for i in range(len(cells_in_col)):
                for j in range(i + 1, len(cells_in_col)):
                    constraints.append((cells_in_col[i], cells_in_col[j]))
        
        # Box constraints
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                cells_in_box = []
                for r in range(box_row, box_row + self.box_size):
                    for c in range(box_col, box_col + self.box_size):
                        cells_in_box.append((r, c))
                
                for i in range(len(cells_in_box)):
                    for j in range(i + 1, len(cells_in_box)):
                        constraints.append((cells_in_box[i], cells_in_box[j]))
        
        return constraints
    
    def _is_consistent(self, var1: Tuple[int, int], val1: int, 
                      var2: Tuple[int, int], val2: int) -> bool:
        """
        Check if two assignments are consistent with Sudoku constraints.
        
        Args:
            var1, var2: Cell coordinates (row, col)
            val1, val2: Values to assign
            
        Returns:
            True if the assignments are consistent
        """
        # Same cell constraint
        if var1 == var2:
            return val1 == val2
        
        # If same value, check if they're in the same constraint group
        if val1 == val2:
            row1, col1 = var1
            row2, col2 = var2
            
            # Same row
            if row1 == row2:
                return False
            
            # Same column
            if col1 == col2:
                return False
            
            # Same box
            box1_row, box1_col = row1 // self.box_size, col1 // self.box_size
            box2_row, box2_col = row2 // self.box_size, col2 // self.box_size
            
            if box1_row == box2_row and box1_col == box2_col:
                return False
        
        return True
    
    def _revise(self, var1: Tuple[int, int], var2: Tuple[int, int], 
                domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        """
        Revise domains based on arc consistency.
        
        Args:
            var1, var2: Variables to check
            domains: Current domain dictionary
            
        Returns:
            True if any domain was revised
        """
        revised = False
        
        # If var1 is assigned (domain has only one value)
        if len(domains[var1]) == 1:
            val1 = list(domains[var1])[0]
            
            # Remove inconsistent values from var2's domain
            values_to_remove = set()
            for val2 in domains[var2]:
                if not self._is_consistent(var1, val1, var2, val2):
                    values_to_remove.add(val2)
            
            if values_to_remove:
                domains[var2] -= values_to_remove
                revised = True
        
        return revised
    
    def _arc_consistency(self, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        """
        Apply AC-3 algorithm for arc consistency.
        
        Args:
            domains: Domain dictionary to make arc consistent
            
        Returns:
            True if domains are arc consistent, False if domain becomes empty
        """
        constraints = self._get_constraints()
        queue = deque(constraints)
        
        while queue:
            var1, var2 = queue.popleft()
            
            if self._revise(var1, var2, domains):
                # If domain becomes empty, problem is unsolvable
                if not domains[var2]:
                    return False
                
                # Add all constraints involving var2 back to queue
                for constraint in constraints:
                    if constraint[1] == var2 and constraint[0] != var1:
                        queue.append(constraint)
        
        return True
    
    def _select_unassigned_variable(self, domains: Dict[Tuple[int, int], Set[int]]) -> Optional[Tuple[int, int]]:
        """
        Select unassigned variable using Minimum Remaining Values (MRV) heuristic.
        
        Args:
            domains: Current domain dictionary
            
        Returns:
            Coordinates of selected variable, or None if all variables assigned
        """
        unassigned = []
        
        for (row, col), domain in domains.items():
            if len(domain) > 1:  # Not assigned yet
                unassigned.append(((row, col), len(domain)))
        
        if not unassigned:
            return None
        
        # Sort by domain size (MRV heuristic)
        unassigned.sort(key=lambda x: x[1])
        return unassigned[0][0]
    
    def _order_domain_values(self, var: Tuple[int, int], 
                            domains: Dict[Tuple[int, int], Set[int]]) -> List[int]:
        """
        Order domain values using Least Constraining Value (LCV) heuristic.
        
        Args:
            var: Variable to order values for
            domains: Current domain dictionary
            
        Returns:
            List of values ordered by how constraining they are
        """
        row, col = var
        values = list(domains[var])
        
        def count_constraints(value):
            """Count how many constraints this value would impose."""
            constraints = 0
            
            # Count constraints in same row
            for c in range(self.size):
                if c != col and (row, c) in domains:
                    if value in domains[(row, c)]:
                        constraints += 1
            
            # Count constraints in same column
            for r in range(self.size):
                if r != row and (r, col) in domains:
                    if value in domains[(r, col)]:
                        constraints += 1
            
            # Count constraints in same box
            box_row, box_col = row // self.box_size, col // self.box_size
            for r in range(box_row * self.box_size, (box_row + 1) * self.box_size):
                for c in range(box_col * self.box_size, (box_col + 1) * self.box_size):
                    if (r, c) != (row, col) and (r, c) in domains:
                        if value in domains[(r, c)]:
                            constraints += 1
            
            return constraints
        
        # Sort by constraint count (ascending = least constraining first)
        values.sort(key=count_constraints)
        return values
    
    def _is_complete(self, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        """Check if all variables are assigned."""
        return all(len(domain) == 1 for domain in domains.values())
    
    def _backtrack(self, domains: Dict[Tuple[int, int], Set[int]]) -> Optional[Dict[Tuple[int, int], Set[int]]]:
        """
        Main backtracking algorithm with arc consistency.
        
        Args:
            domains: Current domain dictionary
            
        Returns:
            Solution domains if found, None otherwise
        """
        # Check if complete
        if self._is_complete(domains):
            return domains
        
        # Select variable using MRV heuristic
        var = self._select_unassigned_variable(domains)
        if var is None:
            return None
        
        # Order values using LCV heuristic
        values = self._order_domain_values(var, domains)
        
        for value in values:
            self.assignments += 1
            
            # Create a copy of domains for this branch
            new_domains = copy.deepcopy(domains)
            new_domains[var] = {value}
            
            # Apply arc consistency
            if self._arc_consistency(new_domains):
                # Recursive call
                result = self._backtrack(new_domains)
                if result is not None:
                    return result
            
            self.backtracks += 1
        
        return None
    
    def solve(self) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
        """
        Solve the Sudoku puzzle.
        
        Returns:
            Tuple of (success, solution_grid, statistics)
        """
        self.start_time = time.time()
        self.assignments = 0
        self.backtracks = 0
        
        # Apply initial arc consistency
        if not self._arc_consistency(self.domains):
            end_time = time.time()
            stats = {
                'assignments': self.assignments,
                'backtracks': self.backtracks,
                'time': end_time - self.start_time,
                'solvable': False
            }
            return False, self.grid, stats
        
        # Solve using backtracking
        solution_domains = self._backtrack(self.domains)
        
        end_time = time.time()
        
        if solution_domains is None:
            stats = {
                'assignments': self.assignments,
                'backtracks': self.backtracks,
                'time': end_time - self.start_time,
                'solvable': False
            }
            return False, self.grid, stats
        
        # Convert solution back to grid format
        solution_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for (row, col), domain in solution_domains.items():
            solution_board[row][col] = list(domain)[0]
        
        solution_grid = self._board_to_grid(solution_board)
        
        stats = {
            'assignments': self.assignments,
            'backtracks': self.backtracks,
            'time': end_time - self.start_time,
            'solvable': True
        }
        
        return True, solution_grid, stats
    
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
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                values = []
                for r in range(box_row, box_row + self.box_size):
                    for c in range(box_col, box_col + self.box_size):
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


def solve_sudoku(grid: List[List[str]]) -> Tuple[bool, List[List[str]], Dict[str, Any]]:
    """
    Convenience function to solve a Sudoku grid.
    
    Args:
        grid: 9x9 Sudoku grid with empty cells as empty strings or '0'
        
    Returns:
        Tuple of (success, solution_grid, statistics)
    """
    solver = SudokuSolver(grid)
    return solver.solve()


if __name__ == "__main__":
    # Example usage and testing
    print("Sudoku Solver with Arc Consistency")
    print("=" * 40)
    
    # Test puzzle (solvable)
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
    
    solver = SudokuSolver(test_puzzle)
    
    print("Original puzzle:")
    solver.print_grid(test_puzzle, "Original Puzzle")
    
    print(f"\nPuzzle is valid: {solver.is_valid_sudoku()}")
    
    print("\nSolving...")
    success, solution, stats = solver.solve()
    
    if success:
        solver.print_grid(solution, "Solution")
        print(f"\nStatistics:")
        print(f"  Assignments: {stats['assignments']}")
        print(f"  Backtracks: {stats['backtracks']}")
        print(f"  Time: {stats['time']:.4f} seconds")
        print(f"  Solvable: {stats['solvable']}")
    else:
        print("No solution found!")
        print(f"Statistics:")
        print(f"  Assignments: {stats['assignments']}")
        print(f"  Backtracks: {stats['backtracks']}")
        print(f"  Time: {stats['time']:.4f} seconds")
        print(f"  Solvable: {stats['solvable']}")
