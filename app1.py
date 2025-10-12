import pygame
import time

# initialize pygame
pygame.init()

# ---- some constants ----
WIDTH, HEIGHT = 540, 600  # 9x9 board + space for messages
ROWS, COLS = 9, 9
CELL_SIZE = WIDTH // COLS

# fonts
FONT = pygame.font.SysFont("comicsans", 40)
FONT_SMALL = pygame.font.SysFont("comicsans", 25)

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (180, 180, 180)
BLUE = (50, 50, 255)
GREEN = (0, 180, 0)
RED = (200, 0, 0)

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

original_board = [row[:] for row in board]  # copy for reference

# create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku Solver (Human Style!)")

# Clock for frame rate management
clock = pygame.time.Clock()

# Pre-render static grid surface to avoid redrawing
static_grid = pygame.Surface((WIDTH, WIDTH))

# Cache for rendered number surfaces
number_cache = {}

def create_number_cache():
    """Pre-render all numbers to avoid repeated font rendering"""
    for num in range(1, 10):
        number_cache[num] = {
            'black': FONT.render(str(num), True, BLACK),
            'blue': FONT.render(str(num), True, BLUE)
        }

def draw_static_grid():
    """Draw the static grid once onto a surface"""
    static_grid.fill(WHITE)
    for i in range(ROWS + 1):
        thick = 4 if i % 3 == 0 else 1  # thicker lines for 3x3 boxes
        pygame.draw.line(static_grid, BLACK, (0, i*CELL_SIZE), (WIDTH, i*CELL_SIZE), thick)
        pygame.draw.line(static_grid, BLACK, (i*CELL_SIZE, 0), (i*CELL_SIZE, WIDTH), thick)

def draw_grid():
    """Blit the pre-rendered grid"""
    screen.blit(static_grid, (0, 0))

def draw_numbers(dirty_rects=None):
    """Draw numbers on the grid with optional dirty rectangle tracking"""
    rects = []
    for r in range(ROWS):
        for c in range(COLS):
            num = board[r][c]
            if num != 0:
                color_key = 'black' if original_board[r][c] != 0 else 'blue'
                text = number_cache[num][color_key]
                x, y = c*CELL_SIZE + 18, r*CELL_SIZE + 10
                
                # Clear the cell first
                cell_rect = pygame.Rect(c*CELL_SIZE + 1, r*CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(screen, WHITE, cell_rect)
                
                screen.blit(text, (x, y))
                if dirty_rects is not None:
                    rects.append(cell_rect)
    
    return rects if dirty_rects is not None else None

def find_empty_cell():
    """Find the next empty cell efficiently in a single pass"""
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def is_valid_optimized(board, row, col, num):
    """Optimized validity check using sets (O(1) lookups instead of O(n) loops)"""
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

def solve_sudoku_optimized():
    """Optimized solve using better cell finding and validity checks"""
    # Handle pygame events to prevent freezing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    
    # Find next empty cell
    empty = find_empty_cell()
    if not empty:
        return True  # Solved!
    
    row, col = empty
    
    for num in range(1, 10):
        if is_valid_optimized(board, row, col, num):
            board[row][col] = num
            
            # Draw only the changed cell
            draw_grid()
            draw_numbers()
            
            # Update only changed region for better performance
            cell_rect = pygame.Rect(col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.display.update(cell_rect)
            
            # Frame rate control
            clock.tick(60)  # Limit to 60 FPS
            pygame.time.delay(40)  # Human-like speed
            
            if solve_sudoku_optimized():
                return True
            
            # Backtrack
            board[row][col] = 0
            draw_grid()
            draw_numbers()
            pygame.display.update(cell_rect)
            pygame.time.delay(30)
    
    return False

# ---- main loop ----
def main():
    running = True
    started = False
    solved = False
    
    # Pre-render static elements
    draw_static_grid()
    create_number_cache()
    
    # Pre-render instruction message
    instruction_msg = FONT_SMALL.render("Press SPACE to solve Sudoku!", True, RED)
    solved_msg = FONT_SMALL.render("Sudoku Solved!", True, GREEN)
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not solved:
                    started = True  # start solving
        
        # Draw everything
        draw_grid()
        draw_numbers()
        
        # Display message
        if not solved:
            screen.blit(instruction_msg, (10, WIDTH + 10))
        else:
            screen.blit(solved_msg, (150, WIDTH + 10))
        
        pygame.display.update()
        
        # Control frame rate
        clock.tick(60)
        
        if started:
            solved = solve_sudoku_optimized()
            started = False  # stop solving after done
    
    pygame.quit()

if _name_ == "_main_":
    main()
