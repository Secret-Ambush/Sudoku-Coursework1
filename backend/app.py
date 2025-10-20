from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import sys
import os
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime

# Add the parent directory to the path to import the sudoku solver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ArcConsistency_Implementation.sudoku_solver import SudokuSolver
from Pruning_Implementation.pruning_demo import PruningSudokuSolver
from sudoku_generator import generate_sudoku

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ML Models directory
MODEL_DIR = 'ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================== ML FEATURE EXTRACTION =====================

class SudokuMLFeatureExtractor:
    """Extract machine learning features from Sudoku puzzles"""
    
    @staticmethod
    def extract_features(grid):
        """
        Extract comprehensive features from a Sudoku puzzle
        Returns a dictionary of features
        """
        features = {}
        
        # Convert grid to numpy array for easier manipulation
        num_grid = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                if grid[i][j] and grid[i][j].strip():
                    num_grid[i, j] = int(grid[i][j])
        
        # Basic features
        features['empty_cells'] = np.sum(num_grid == 0)
        features['filled_cells'] = 81 - features['empty_cells']
        features['fill_ratio'] = features['filled_cells'] / 81.0
        
        # Distribution features
        empty_rows = [np.sum(num_grid[i] == 0) for i in range(9)]
        empty_cols = [np.sum(num_grid[:, j] == 0) for j in range(9)]
        
        features['empty_cells_std_rows'] = np.std(empty_rows)
        features['empty_cells_std_cols'] = np.std(empty_cols)
        features['max_empty_row'] = max(empty_rows)
        features['min_empty_row'] = min(empty_rows)
        features['max_empty_col'] = max(empty_cols)
        features['min_empty_col'] = min(empty_cols)
        
        # 3x3 box features
        empty_boxes = []
        for box_row in range(3):
            for box_col in range(3):
                box = num_grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                empty_boxes.append(np.sum(box == 0))
        
        features['empty_cells_std_boxes'] = np.std(empty_boxes)
        features['max_empty_box'] = max(empty_boxes)
        features['min_empty_box'] = min(empty_boxes)
        
        # Constraint analysis
        features['naked_singles'] = SudokuMLFeatureExtractor._count_naked_singles(num_grid)
        features['hidden_singles'] = SudokuMLFeatureExtractor._count_hidden_singles(num_grid)
        
        # Symmetry features
        features['horizontal_symmetry'] = SudokuMLFeatureExtractor._check_symmetry(num_grid, 'horizontal')
        features['vertical_symmetry'] = SudokuMLFeatureExtractor._check_symmetry(num_grid, 'vertical')
        features['diagonal_symmetry'] = SudokuMLFeatureExtractor._check_symmetry(num_grid, 'diagonal')
        
        # Clustering coefficient (connectivity)
        features['clustering_coefficient'] = SudokuMLFeatureExtractor._clustering_coefficient(num_grid)
        
        # Number distribution features
        for num in range(1, 10):
            features[f'count_{num}'] = np.sum(num_grid == num)
        
        # Entropy measure
        counts = [features[f'count_{i}'] for i in range(1, 10)]
        counts_arr = np.array(counts) + 1  # Add 1 to avoid log(0)
        probs = counts_arr / np.sum(counts_arr)
        features['entropy'] = -np.sum(probs * np.log2(probs))
        
        return features
    
    @staticmethod
    def _count_naked_singles(grid):
        """Count cells that have only one possible value"""
        count = 0
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    possible = SudokuMLFeatureExtractor._get_possible_values(grid, i, j)
                    if len(possible) == 1:
                        count += 1
        return count
    
    @staticmethod
    def _count_hidden_singles(grid):
        """Count hidden singles in rows, columns, and boxes"""
        count = 0
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    possible = SudokuMLFeatureExtractor._get_possible_values(grid, i, j)
                    for val in possible:
                        if SudokuMLFeatureExtractor._is_hidden_single(grid, i, j, val):
                            count += 1
                            break
        return count
    
    @staticmethod
    def _get_possible_values(grid, row, col):
        """Get possible values for a cell"""
        possible = set(range(1, 10))
        
        # Remove values in row
        possible -= set(grid[row, :])
        
        # Remove values in column
        possible -= set(grid[:, col])
        
        # Remove values in box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = grid[box_row:box_row+3, box_col:box_col+3]
        possible -= set(box.flatten())
        
        possible.discard(0)
        return possible
    
    @staticmethod
    def _is_hidden_single(grid, row, col, val):
        """Check if value is a hidden single"""
        # Check row
        row_possible = []
        for c in range(9):
            if c != col and grid[row, c] == 0:
                if val in SudokuMLFeatureExtractor._get_possible_values(grid, row, c):
                    row_possible.append(c)
        if len(row_possible) == 0:
            return True
        
        # Check column
        col_possible = []
        for r in range(9):
            if r != row and grid[r, col] == 0:
                if val in SudokuMLFeatureExtractor._get_possible_values(grid, r, col):
                    col_possible.append(r)
        if len(col_possible) == 0:
            return True
        
        # Check box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_possible = []
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and grid[r, c] == 0:
                    if val in SudokuMLFeatureExtractor._get_possible_values(grid, r, c):
                        box_possible.append((r, c))
        if len(box_possible) == 0:
            return True
        
        return False
    
    @staticmethod
    def _check_symmetry(grid, symmetry_type):
        """Check symmetry of empty cells"""
        empty_mask = (grid == 0).astype(int)
        
        if symmetry_type == 'horizontal':
            flipped = np.flipud(empty_mask)
        elif symmetry_type == 'vertical':
            flipped = np.fliplr(empty_mask)
        elif symmetry_type == 'diagonal':
            flipped = empty_mask.T
        else:
            return 0
        
        matches = np.sum(empty_mask == flipped)
        return matches / 81.0
    
    @staticmethod
    def _clustering_coefficient(grid):
        """Calculate clustering coefficient of filled cells"""
        filled_positions = np.argwhere(grid != 0)
        if len(filled_positions) < 2:
            return 0.0
        
        adjacent_count = 0
        for pos in filled_positions:
            r, c = pos
            # Check 4-connectivity
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in neighbors:
                if 0 <= nr < 9 and 0 <= nc < 9 and grid[nr, nc] != 0:
                    adjacent_count += 1
        
        max_possible = len(filled_positions) * 4
        return adjacent_count / max_possible if max_possible > 0 else 0.0

# ===================== ML MODELS =====================

class SudokuMLModels:
    """Machine learning models for Sudoku analysis"""
    
    def __init__(self):
        self.difficulty_classifier = None
        self.time_predictor = None
        self.algorithm_predictor = None
        self.puzzle_clusterer = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train_difficulty_classifier(self, training_data):
        """
        Train a model to classify puzzle difficulty
        training_data: list of (grid, difficulty_label) tuples
        """
        X = []
        y = []
        
        for grid, difficulty in training_data:
            features = SudokuMLFeatureExtractor.extract_features(grid)
            X.append(list(features.values()))
            y.append(difficulty)
        
        X = np.array(X)
        self.feature_names = list(SudokuMLFeatureExtractor.extract_features(training_data[0][0]).keys())
        
        # Train classifier
        self.difficulty_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.difficulty_classifier.fit(X, y)
        
        return self.difficulty_classifier.score(X, y)
    
    def train_time_predictor(self, training_data):
        """
        Train a model to predict solving time
        training_data: list of (grid, solve_time) tuples
        """
        X = []
        y = []
        
        for grid, solve_time in training_data:
            features = SudokuMLFeatureExtractor.extract_features(grid)
            X.append(list(features.values()))
            y.append(solve_time)
        
        X = np.array(X)
        
        # Train regressor
        self.time_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.time_predictor.fit(X, y)
        
        return self.time_predictor.score(X, y)
    
    def train_algorithm_predictor(self, training_data):
        """
        Train a model to predict which algorithm will be faster
        training_data: list of (grid, faster_algorithm) tuples where faster_algorithm is 'ac3' or 'pruning'
        """
        X = []
        y = []
        
        for grid, faster_algo in training_data:
            features = SudokuMLFeatureExtractor.extract_features(grid)
            X.append(list(features.values()))
            y.append(faster_algo)
        
        X = np.array(X)
        
        # Train classifier
        self.algorithm_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.algorithm_predictor.fit(X, y)
        
        return self.algorithm_predictor.score(X, y)
    
    def train_puzzle_clusterer(self, training_grids, n_clusters=5):
        """
        Train unsupervised clustering model to group similar puzzles
        training_grids: list of grids
        """
        X = []
        
        for grid in training_grids:
            features = SudokuMLFeatureExtractor.extract_features(grid)
            X.append(list(features.values()))
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-means clusterer
        self.puzzle_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.puzzle_clusterer.fit(X_scaled)
        
        return self.puzzle_clusterer.inertia_
    
    def predict_difficulty(self, grid):
        """Predict difficulty of a puzzle"""
        if self.difficulty_classifier is None:
            return None
        
        features = SudokuMLFeatureExtractor.extract_features(grid)
        X = np.array([list(features.values())])
        
        prediction = self.difficulty_classifier.predict(X)[0]
        probabilities = self.difficulty_classifier.predict_proba(X)[0]
        
        return {
            'predicted_difficulty': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.difficulty_classifier.classes_, probabilities)
            }
        }
    
    def predict_solve_time(self, grid):
        """Predict solving time for a puzzle"""
        if self.time_predictor is None:
            return None
        
        features = SudokuMLFeatureExtractor.extract_features(grid)
        X = np.array([list(features.values())])
        
        predicted_time = self.time_predictor.predict(X)[0]
        return float(predicted_time)
    
    def predict_best_algorithm(self, grid):
        """Predict which algorithm will be faster"""
        if self.algorithm_predictor is None:
            return None
        
        features = SudokuMLFeatureExtractor.extract_features(grid)
        X = np.array([list(features.values())])
        
        prediction = self.algorithm_predictor.predict(X)[0]
        probabilities = self.algorithm_predictor.predict_proba(X)[0]
        
        return {
            'predicted_algorithm': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.algorithm_predictor.classes_, probabilities)
            }
        }
    
    def cluster_puzzle(self, grid):
        """Assign puzzle to a cluster"""
        if self.puzzle_clusterer is None:
            return None
        
        features = SudokuMLFeatureExtractor.extract_features(grid)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        cluster = self.puzzle_clusterer.predict(X_scaled)[0]
        distances = self.puzzle_clusterer.transform(X_scaled)[0]
        
        return {
            'cluster_id': int(cluster),
            'distance_to_centroid': float(distances[cluster]),
            'distances_to_all_centroids': [float(d) for d in distances]
        }
    
    def get_feature_importance(self, model_type='difficulty'):
        """Get feature importance for specified model"""
        if model_type == 'difficulty' and self.difficulty_classifier:
            importances = self.difficulty_classifier.feature_importances_
        elif model_type == 'time' and self.time_predictor:
            importances = self.time_predictor.feature_importances_
        elif model_type == 'algorithm' and self.algorithm_predictor:
            importances = self.algorithm_predictor.feature_importances_
        else:
            return None
        
        if self.feature_names is None:
            return None
        
        feature_importance = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'feature': name, 'importance': float(importance)}
            for name, importance in feature_importance[:10]  # Top 10
        ]
    
    def save_models(self, path=MODEL_DIR):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        if self.difficulty_classifier:
            with open(os.path.join(path, 'difficulty_classifier.pkl'), 'wb') as f:
                pickle.dump(self.difficulty_classifier, f)
        
        if self.time_predictor:
            with open(os.path.join(path, 'time_predictor.pkl'), 'wb') as f:
                pickle.dump(self.time_predictor, f)
        
        if self.algorithm_predictor:
            with open(os.path.join(path, 'algorithm_predictor.pkl'), 'wb') as f:
                pickle.dump(self.algorithm_predictor, f)
        
        if self.puzzle_clusterer:
            with open(os.path.join(path, 'puzzle_clusterer.pkl'), 'wb') as f:
                pickle.dump(self.puzzle_clusterer, f)
            with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        if self.feature_names:
            with open(os.path.join(path, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_names, f)
    
    def load_models(self, path=MODEL_DIR):
        """Load all trained models"""
        try:
            if os.path.exists(os.path.join(path, 'difficulty_classifier.pkl')):
                with open(os.path.join(path, 'difficulty_classifier.pkl'), 'rb') as f:
                    self.difficulty_classifier = pickle.load(f)
            
            if os.path.exists(os.path.join(path, 'time_predictor.pkl')):
                with open(os.path.join(path, 'time_predictor.pkl'), 'rb') as f:
                    self.time_predictor = pickle.load(f)
            
            if os.path.exists(os.path.join(path, 'algorithm_predictor.pkl')):
                with open(os.path.join(path, 'algorithm_predictor.pkl'), 'rb') as f:
                    self.algorithm_predictor = pickle.load(f)
            
            if os.path.exists(os.path.join(path, 'puzzle_clusterer.pkl')):
                with open(os.path.join(path, 'puzzle_clusterer.pkl'), 'rb') as f:
                    self.puzzle_clusterer = pickle.load(f)
                with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if os.path.exists(os.path.join(path, 'feature_names.json')):
                with open(os.path.join(path, 'feature_names.json'), 'r') as f:
                    self.feature_names = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Initialize ML models
ml_models = SudokuMLModels()
ml_models.load_models()

# ===================== TRAINING DATA GENERATION =====================

def generate_training_data(n_samples=100):
    """Generate training data by solving puzzles of different difficulties"""
    training_data = {
        'difficulty': [],
        'time': [],
        'algorithm': []
    }
    
    difficulties = ['Easy', 'Medium', 'Hard']
    
    for difficulty in difficulties:
        for _ in range(n_samples // len(difficulties)):
            try:
                puzzle, _ = generate_sudoku(difficulty)
                
                # Solve with both algorithms
                ac3_solver = SudokuSolver(puzzle)
                ac3_solved, _, ac3_stats = ac3_solver.solve()
                
                pruning_solver = PruningSudokuSolver(puzzle)
                pruning_solved, _, pruning_stats = pruning_solver.solve()
                
                if ac3_solved and pruning_solved:
                    # Store difficulty data
                    training_data['difficulty'].append((puzzle, difficulty))
                    
                    # Store time data (average of both algorithms)
                    avg_time = (ac3_stats['time'] + pruning_stats['time']) / 2
                    training_data['time'].append((puzzle, avg_time))
                    
                    # Store algorithm preference data
                    faster_algo = 'ac3' if ac3_stats['time'] < pruning_stats['time'] else 'pruning'
                    training_data['algorithm'].append((puzzle, faster_algo))
            
            except Exception as e:
                print(f"Error generating training sample: {e}")
                continue
    
    return training_data

# ===================== EXISTING API ENDPOINTS =====================

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
        
        # ML Analysis
        ml_analysis = {}
        try:
            # Extract features
            features = SudokuMLFeatureExtractor.extract_features(grid)
            ml_analysis['puzzle_features'] = features
            
            # Predictions
            if ml_models.difficulty_classifier:
                ml_analysis['predicted_difficulty'] = ml_models.predict_difficulty(grid)
            
            if ml_models.time_predictor:
                ml_analysis['predicted_solve_time'] = ml_models.predict_solve_time(grid)
            
            if ml_models.algorithm_predictor:
                ml_analysis['predicted_best_algorithm'] = ml_models.predict_best_algorithm(grid)
            
            if ml_models.puzzle_clusterer:
                ml_analysis['cluster_info'] = ml_models.cluster_puzzle(grid)
        
        except Exception as e:
            ml_analysis['error'] = str(e)
        
        if ac3_solved and pruning_solved:
            return jsonify({
                'success': True,
                'solution': ac3_solution,
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
                },
                'ml_analysis': ml_analysis
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
                },
                'ml_analysis': ml_analysis
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
                },
                'ml_analysis': ml_analysis
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
                },
                'ml_analysis': ml_analysis
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
        
        # ML Analysis
        ml_analysis = {}
        try:
            features = SudokuMLFeatureExtractor.extract_features(puzzle)
            ml_analysis['puzzle_features'] = features
            
            if ml_models.difficulty_classifier:
                ml_analysis['predicted_difficulty'] = ml_models.predict_difficulty(puzzle)
            
            if ml_models.time_predictor:
                ml_analysis['predicted_solve_time'] = ml_models.predict_solve_time(puzzle)
            
            if ml_models.puzzle_clusterer:
                ml_analysis['cluster_info'] = ml_models.cluster_puzzle(puzzle)
        except Exception as e:
            ml_analysis['error'] = str(e)
        
        return jsonify({
            'success': True,
            'puzzle': puzzle,
            'solution': solution,
            'difficulty': difficulty,
            'ml_analysis': ml_analysis
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

# ===================== NEW ML API ENDPOINTS =====================

@app.route('/api/ml/analyze', methods=['POST'])
def analyze_puzzle():
    """Comprehensive ML analysis of a puzzle"""
    try:
        data = request.get_json()
        grid = data.get('grid', [])
        
        if not grid or len(grid) != 9 or any(len(row) != 9 for row in grid):
            return jsonify({'error': 'Invalid grid format'}), 400
        
        # Extract features
        features = SudokuMLFeatureExtractor.extract_features(grid)
        
        analysis = {
            'features': features,
            'predictions': {}
        }
        
        # Difficulty prediction
        if ml_models.difficulty_classifier:
            analysis['predictions']['difficulty'] = ml_models.predict_difficulty(grid)
        
        # Time prediction
        if ml_models.time_predictor:
            analysis['predictions']['solve_time'] = ml_models.predict_solve_time(grid)
        
        # Algorithm prediction
        if ml_models.algorithm_predictor:
            analysis['predictions']['best_algorithm'] = ml_models.predict_best_algorithm(grid)
        
        # Clustering
        if ml_models.puzzle_clusterer:
            analysis['predictions']['cluster'] = ml_models.cluster_puzzle(grid)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/train', methods=['POST'])
def train_models():
    """Train ML models on generated data"""
    try:
        data = request.get_json()
        n_samples = data.get('n_samples', 100)
        
        print(f"Generating {n_samples} training samples...")
        training_data = generate_training_data(n_samples)
        
        results = {}
        
        # Train difficulty classifier
        if len(training_data['difficulty']) > 0:
            print("Training difficulty classifier...")
            score = ml_models.train_difficulty_classifier(training_data['difficulty'])
            results['difficulty_classifier'] = {
                'trained': True,
                'accuracy': float(score),
                'samples': len(training_data['difficulty'])
            }
        
        # Train time predictor
        if len(training_data['time']) > 0:
            print("Training time predictor...")
            score = ml_models.train_time_predictor(training_data['time'])
            results['time_predictor'] = {
                'trained': True,
                'r2_score': float(score),
                'samples': len(training_data['time'])
            }
        
        # Train algorithm predictor
        if len(training_data['algorithm']) > 0:
            print("Training algorithm predictor...")
            score = ml_models.train_algorithm_predictor(training_data['algorithm'])
            results['algorithm_predictor'] = {
                'trained': True,
                'accuracy': float(score),
                'samples': len(training_data['algorithm'])
            }
        
        # Train clusterer
        all_grids = [item[0] for item in training_data['difficulty']]
        if len(all_grids) > 5:
            print("Training puzzle clusterer...")
            inertia = ml_models.train_puzzle_clusterer(all_grids, n_clusters=5)
            results['puzzle_clusterer'] = {
                'trained': True,
                'inertia': float(inertia),
                'samples': len(all_grids)
            }
        
        # Save models
        ml_models.save_models()
        results['models_saved'] = True
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance for ML models"""
    try:
        model_type = request.args.get('model', 'difficulty')
        
        importance = ml_models.get_feature_importance(model_type)
        
        if importance is None:
            return jsonify({
                'success': False,
                'error': f'Model {model_type} not trained or not available'
            }), 404
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'feature_importance': importance
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/models/status', methods=['GET'])
def models_status():
    """Get status of all ML models"""
    try:
        status = {
            'difficulty_classifier': ml_models.difficulty_classifier is not None,
            'time_predictor': ml_models.time_predictor is not None,
            'algorithm_predictor': ml_models.algorithm_predictor is not None,
            'puzzle_clusterer': ml_models.puzzle_clusterer is not None
        }
        
        return jsonify({
            'success': True,
            'models': status,
            'all_trained': all(status.values())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/compare-puzzles', methods=['POST'])
def compare_puzzles():
    """Compare multiple puzzles using ML features"""
    try:
        data = request.get_json()
        puzzles = data.get('puzzles', [])
        
        if not puzzles or len(puzzles) < 2:
            return jsonify({'error': 'At least 2 puzzles required for comparison'}), 400
        
        comparisons = []
        
        for idx, grid in enumerate(puzzles):
            if not grid or len(grid) != 9 or any(len(row) != 9 for row in grid):
                return jsonify({'error': f'Invalid grid format for puzzle {idx+1}'}), 400
            
            features = SudokuMLFeatureExtractor.extract_features(grid)
            
            puzzle_data = {
                'puzzle_id': idx + 1,
                'features': features
            }
            
            if ml_models.difficulty_classifier:
                puzzle_data['predicted_difficulty'] = ml_models.predict_difficulty(grid)
            
            if ml_models.puzzle_clusterer:
                puzzle_data['cluster'] = ml_models.cluster_puzzle(grid)
            
            comparisons.append(puzzle_data)
        
        return jsonify({
            'success': True,
            'comparisons': comparisons
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
