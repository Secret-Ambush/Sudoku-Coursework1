"""
Flask backend for Sudoku React frontend
"""
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import sys
import os
import pandas as pd
from io import StringIO 

# Add the parent directory to the path to import the sudoku solver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ArcConsistency_Implementation.sudoku_solver import SudokuSolver
from Pruning_Implementation.pruning_demo import PruningSudokuSolver
from sudoku_generator import generate_sudoku

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (temporary fix)

# Handle preflight requests explicitly
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/solve', methods=['POST'])
def solve_sudoku():
    """Solve a Sudoku puzzle using both AC-3 and Pruning algorithms for comparison"""
    try:
        data = request.get_json()
        grid = data.get('grid', [])
        
        if not grid or len(grid) != 9 or any(len(row) != 9 for row in grid):
            return jsonify({'error': 'Invalid grid format'}), 400
        
        # Solve with AC-3 (Arc Consistency)
        ac3_solver = SudokuSolver(grid)
        ac3_solved, ac3_solution, ac3_stats = ac3_solver.solve()
        
        # Solve with Pruning algorithm
        pruning_solver = PruningSudokuSolver(grid)
        pruning_solved, pruning_solution, pruning_stats = pruning_solver.solve()
        
        # Check if both algorithms found the same solution
        solutions_match = ac3_solved and pruning_solved and ac3_solution == pruning_solution
        
        if ac3_solved and pruning_solved:
            return jsonify({
                'success': True,
                'solution': ac3_solution,  # Use AC-3 solution as the main result
                'solutions_match': solutions_match,
                'algorithms': {
                    'ac3': {
                        'name': 'Arc Consistency (AC-3)',
                        'solve_time': ac3_stats['time'],
                        'nodes_explored': ac3_stats['assignments'],
                        'backtracks': ac3_stats['backtracks'],
                        'success': ac3_solved
                    },
                    'pruning': {
                        'name': 'Backtracking with Pruning',
                        'solve_time': pruning_stats['time'],
                        'nodes_explored': pruning_stats['assignments'],
                        'backtracks': pruning_stats['backtracks'],
                        'success': pruning_solved
                    }
                },
                'comparison': {
                    'faster_algorithm': 'ac3' if ac3_stats['time'] < pruning_stats['time'] else 'pruning',
                    'time_difference': abs(ac3_stats['time'] - pruning_stats['time']),
                    'efficiency_ratio': ac3_stats['assignments'] / pruning_stats['assignments'] if pruning_stats['assignments'] > 0 else float('inf')
                }
            })
        elif ac3_solved:
            return jsonify({
                'success': True,
                'solution': ac3_solution,
                'solutions_match': False,
                'algorithms': {
                    'ac3': {
                        'name': 'Arc Consistency (AC-3)',
                        'solve_time': ac3_stats['time'],
                        'nodes_explored': ac3_stats['assignments'],
                        'backtracks': ac3_stats['backtracks'],
                        'success': True
                    },
                    'pruning': {
                        'name': 'Backtracking with Pruning',
                        'solve_time': pruning_stats['time'],
                        'nodes_explored': pruning_stats['assignments'],
                        'backtracks': pruning_stats['backtracks'],
                        'success': False
                    }
                },
                'comparison': {
                    'faster_algorithm': 'ac3',
                    'time_difference': ac3_stats['time'],
                    'efficiency_ratio': float('inf')
                }
            })
        elif pruning_solved:
            return jsonify({
                'success': True,
                'solution': pruning_solution,
                'solutions_match': False,
                'algorithms': {
                    'ac3': {
                        'name': 'Arc Consistency (AC-3)',
                        'solve_time': ac3_stats['time'],
                        'nodes_explored': ac3_stats['assignments'],
                        'backtracks': ac3_stats['backtracks'],
                        'success': False
                    },
                    'pruning': {
                        'name': 'Backtracking with Pruning',
                        'solve_time': pruning_stats['time'],
                        'nodes_explored': pruning_stats['assignments'],
                        'backtracks': pruning_stats['backtracks'],
                        'success': True
                    }
                },
                'comparison': {
                    'faster_algorithm': 'pruning',
                    'time_difference': pruning_stats['time'],
                    'efficiency_ratio': 0
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No solution found by either algorithm',
                'algorithms': {
                    'ac3': {
                        'name': 'Arc Consistency (AC-3)',
                        'solve_time': ac3_stats['time'],
                        'nodes_explored': ac3_stats['assignments'],
                        'backtracks': ac3_stats['backtracks'],
                        'success': False
                    },
                    'pruning': {
                        'name': 'Backtracking with Pruning',
                        'solve_time': pruning_stats['time'],
                        'nodes_explored': pruning_stats['assignments'],
                        'backtracks': pruning_stats['backtracks'],
                        'success': False
                    }
                }
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_sudoku_puzzle():
    """Generate a new Sudoku puzzle"""
    try:
        print("Generate endpoint called")
        data = request.get_json()
        print(f"Received data: {data}")
        difficulty = data.get('difficulty', 'Easy')
        print(f"Difficulty: {difficulty}")
        
        # Generate puzzle and solution
        puzzle, solution = generate_sudoku(difficulty)
        print(f"Generated puzzle: {puzzle}")
        
        return jsonify({
            'success': True,
            'puzzle': puzzle,
            'solution': solution,
            'difficulty': difficulty
        })
    
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_sudoku():
    """Validate a Sudoku solution"""
    try:
        data = request.get_json()
        grid = data.get('grid', [])
        
        if not grid or len(grid) != 9 or any(len(row) != 9 for row in grid):
            return jsonify({'error': 'Invalid grid format'}), 400
        
        # Create solver instance for validation
        solver = SudokuSolver(grid)
        
        # Check if the grid is valid
        is_valid = solver.is_valid_sudoku()
        
        if is_valid:
            return jsonify({
                'success': True,
                'valid': True,
                'message': 'Valid Sudoku solution!'
            })
        else:
            return jsonify({
                'success': True,
                'valid': False,
                'message': 'Invalid Sudoku solution'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload and parse a CSV file containing a Sudoku puzzle"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read the CSV content
        content = file.read().decode('utf-8')
        
        # Parse CSV into DataFrame
        try:
            df = pd.read_csv(StringIO(content), header=None, nrows=9)
        except Exception as e:
            return jsonify({'error': f'Error parsing CSV: {str(e)}'}), 400
        
        # Validate dimensions
        if df.shape[0] < 9 or df.shape[1] < 9:
            return jsonify({'error': 'CSV must contain at least 9x9 grid'}), 400
        
        # Convert to grid format
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                cell_value = df.iat[i, j]
                
                # Handle different data types
                if pd.isna(cell_value) or cell_value == '' or str(cell_value).strip() == '':
                    row.append('')
                else:
                    # Convert to string and strip whitespace
                    cell_str = str(cell_value).strip()
                    
                    # Handle float values like "2.0" -> "2"
                    if '.' in cell_str and cell_str.replace('.', '').isdigit():
                        try:
                            float_val = float(cell_str)
                            if float_val.is_integer() and 1 <= float_val <= 9:
                                row.append(str(int(float_val)))
                            else:
                                return jsonify({'error': f'Invalid value "{cell_str}" at row {i+1}, column {j+1}. Only digits 1-9 or empty cells allowed.'}), 400
                        except ValueError:
                            return jsonify({'error': f'Invalid value "{cell_str}" at row {i+1}, column {j+1}. Only digits 1-9 or empty cells allowed.'}), 400
                    elif cell_str in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        row.append(cell_str)
                    elif cell_str.lower() in ['nan', 'none', 'null', '0', '0.0']:
                        row.append('')
                    else:
                        return jsonify({'error': f'Invalid value "{cell_str}" at row {i+1}, column {j+1}. Only digits 1-9 or empty cells allowed.'}), 400
            grid.append(row)
        
        # Validate the puzzle
        solver = SudokuSolver(grid)
        if not solver.is_valid_sudoku():
            return jsonify({'error': 'Invalid Sudoku puzzle: contains conflicts'}), 400
        
        return jsonify({
            'success': True,
            'grid': grid,
            'message': f'Successfully uploaded puzzle from {file.filename}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Sudoku API is running'})

# Serve React app static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React app"""
    if path != "" and os.path.exists(os.path.join('frontend/build', path)):
        return send_from_directory('frontend/build', path)
    else:
        return send_from_directory('frontend/build', 'index.html')

if __name__ == '__main__':
    import time
    # Use Railway's PORT environment variable, fallback to 8000 for local development
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
