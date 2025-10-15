import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

function App() {
  const [grid, setGrid] = useState(Array(9).fill().map(() => Array(9).fill('')));
  const [originalGrid, setOriginalGrid] = useState(Array(9).fill().map(() => Array(9).fill('')));
  const [selectedCell, setSelectedCell] = useState(null);
  const [difficulty, setDifficulty] = useState('Easy');
  const [isSolving, setIsSolving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [solveStats, setSolveStats] = useState(null);
  const [algorithmComparison, setAlgorithmComparison] = useState(null);

  // Initialize empty grid
  const initializeEmptyGrid = useCallback(() => {
    const emptyGrid = Array(9).fill().map(() => Array(9).fill(''));
    setGrid(emptyGrid);
    setOriginalGrid(emptyGrid);
    setSelectedCell(null);
    setMessage('');
    setSolveStats(null);
    setAlgorithmComparison(null);
  }, []);

  // Generate new puzzle
  const generatePuzzle = async () => {
    setIsGenerating(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/generate`, {
        difficulty: difficulty
      });
      
      if (response.data.success) {
        setGrid(response.data.puzzle);
        setOriginalGrid(response.data.puzzle);
        setSelectedCell(null);
        setMessage(`Generated ${difficulty} puzzle successfully!`);
        setSolveStats(null);
        setAlgorithmComparison(null);
      }
    } catch (error) {
      setMessage(`Error generating puzzle: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // Upload CSV file
  const uploadCSV = async (file) => {
    setIsUploading(true);
    setMessage('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.success) {
        setGrid(response.data.grid);
        setOriginalGrid(response.data.grid);
        setSelectedCell(null);
        setMessage(response.data.message);
        setSolveStats(null);
        setAlgorithmComparison(null);
      }
    } catch (error) {
      setMessage(`Error uploading file: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle file input change
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      uploadCSV(file);
    }
  };

  // Solve current puzzle
  const solvePuzzle = async () => {
    setIsSolving(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/solve`, {
        grid: grid
      });
      
      if (response.data.success) {
        setGrid(response.data.solution);
        setSolveStats(response.data.algorithms);
        setAlgorithmComparison(response.data.comparison);
        
        if (response.data.solutions_match) {
          setMessage('Puzzle solved successfully! Both algorithms found the same solution.');
        } else {
          setMessage('Puzzle solved successfully! (Note: Algorithms found different solutions)');
        }
      } else {
        setMessage('No solution found for this puzzle.');
        setSolveStats(response.data.algorithms);
        setAlgorithmComparison(null);
      }
    } catch (error) {
      setMessage(`Error solving puzzle: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsSolving(false);
    }
  };

  // Validate current solution
  const validateSolution = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/validate`, {
        grid: grid
      });
      
      if (response.data.success) {
        setMessage(response.data.message);
      }
    } catch (error) {
      setMessage(`Error validating solution: ${error.response?.data?.error || error.message}`);
    }
  };

  // Handle cell click
  const handleCellClick = (row, col) => {
    setSelectedCell({ row, col });
  };

  // Handle number input
  const handleNumberInput = (number) => {
    if (selectedCell && !isGivenCell(selectedCell.row, selectedCell.col)) {
      const newGrid = grid.map(row => [...row]);
      newGrid[selectedCell.row][selectedCell.col] = number;
      setGrid(newGrid);
      
      // Auto-advance to next cell
      const nextCell = getNextCell(selectedCell.row, selectedCell.col);
      if (nextCell) {
        setSelectedCell(nextCell);
      }
    }
  };

  // Handle keyboard input
  const handleKeyPress = (e) => {
    if (selectedCell && !isGivenCell(selectedCell.row, selectedCell.col)) {
      const key = e.key;
      
      if (key >= '1' && key <= '9') {
        handleNumberInput(key);
      } else if (key === 'Backspace' || key === 'Delete') {
        const newGrid = grid.map(row => [...row]);
        newGrid[selectedCell.row][selectedCell.col] = '';
        setGrid(newGrid);
      } else if (key === 'ArrowUp' || key === 'ArrowDown' || 
                 key === 'ArrowLeft' || key === 'ArrowRight') {
        const nextCell = getNextCellWithDirection(selectedCell.row, selectedCell.col, key);
        if (nextCell) {
          setSelectedCell(nextCell);
        }
      }
    }
  };

  // Check if cell is a given clue
  const isGivenCell = (row, col) => {
    return originalGrid[row][col] !== '';
  };

  // Get next cell for auto-advance
  const getNextCell = (row, col) => {
    if (col < 8) {
      return { row, col: col + 1 };
    } else if (row < 8) {
      return { row: row + 1, col: 0 };
    }
    return null;
  };

  // Get next cell based on direction
  const getNextCellWithDirection = (row, col, direction) => {
    switch (direction) {
      case 'ArrowUp':
        return row > 0 ? { row: row - 1, col } : null;
      case 'ArrowDown':
        return row < 8 ? { row: row + 1, col } : null;
      case 'ArrowLeft':
        return col > 0 ? { row, col: col - 1 } : null;
      case 'ArrowRight':
        return col < 8 ? { row, col: col + 1 } : null;
      default:
        return null;
    }
  };

  // Add keyboard event listener
  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [selectedCell, grid]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧩 Sudoku Solver</h1>
        <p>Interactive Sudoku with AI-powered solving</p>
      </header>


      <main className="App-main">
        <div className="controls">
        <div className="upload-section">
            <label className="file-upload-label">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={isUploading}
                style={{ display: 'none' }}
              />
              <div className="file-upload-button">
                {isUploading ? '📤 Uploading...' : '📁 Upload CSV Puzzle'}
              </div>
            </label>
            <p className="upload-hint">Upload a 9x9 CSV file with digits 1-9 or empty cells</p>
          </div>
          <div className="control-group">
            <label>Difficulty:</label>
            <select 
              value={difficulty} 
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={isGenerating}
            >
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
            </select>
          </div>
          
          <div className="button-group">
            <button 
              onClick={generatePuzzle} 
              disabled={isGenerating}
              className="btn btn-primary"
            >
              {isGenerating ? 'Generating...' : 'Generate Puzzle'}
            </button>
            
            <button 
              onClick={initializeEmptyGrid}
              className="btn btn-secondary"
            >
              Clear Grid
            </button>
            
            <button 
              onClick={solvePuzzle} 
              disabled={isSolving}
              className="btn btn-success"
            >
              {isSolving ? 'Solving...' : 'Solve Puzzle'}
            </button>
            
            <button 
              onClick={validateSolution}
              className="btn btn-info"
            >
              Validate Solution
            </button>
          </div>
        </div>

        <div className="sudoku-container">
          <div className="sudoku-grid">
            {grid.map((row, rowIndex) => 
              row.map((cell, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`sudoku-cell ${
                    selectedCell?.row === rowIndex && selectedCell?.col === colIndex ? 'selected' : ''
                  } ${isGivenCell(rowIndex, colIndex) ? 'given' : ''}`}
                  onClick={() => handleCellClick(rowIndex, colIndex)}
                >
                  {cell}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="number-pad">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
            <button
              key={num}
              className="number-btn"
              onClick={() => handleNumberInput(num.toString())}
            >
              {num}
            </button>
          ))}
        </div>

        {message && (
          <div className="message">
            {message}
          </div>
        )}

        {solveStats && (
          <div className="stats">
            <h3>Algorithm Comparison</h3>
            
            <div className="algorithm-stats">
              <div className="algorithm-card">
                <h4>🔗 Arc Consistency (AC-3)</h4>
                <div className={`status ${solveStats.ac3.success ? 'success' : 'failed'}`}>
                  {solveStats.ac3.success ? '✅ Solved' : '❌ Failed'}
                </div>
                <div className="stat-row">
                  <span>Time:</span>
                  <span>{solveStats.ac3.solve_time.toFixed(3)}s</span>
                </div>
                <div className="stat-row">
                  <span>Nodes Explored:</span>
                  <span>{solveStats.ac3.nodes_explored}</span>
                </div>
                <div className="stat-row">
                  <span>Backtracks:</span>
                  <span>{solveStats.ac3.backtracks}</span>
                </div>
              </div>

              <div className="algorithm-card">
                <h4>✂️ Backtracking with Pruning</h4>
                <div className={`status ${solveStats.pruning.success ? 'success' : 'failed'}`}>
                  {solveStats.pruning.success ? '✅ Solved' : '❌ Failed'}
                </div>
                <div className="stat-row">
                  <span>Time:</span>
                  <span>{solveStats.pruning.solve_time.toFixed(3)}s</span>
                </div>
                <div className="stat-row">
                  <span>Nodes Explored:</span>
                  <span>{solveStats.pruning.nodes_explored}</span>
                </div>
                <div className="stat-row">
                  <span>Backtracks:</span>
                  <span>{solveStats.pruning.backtracks}</span>
                </div>
              </div>
            </div>

            {algorithmComparison && (
              <div className="comparison-summary">
                <h4>📊 Performance Summary</h4>
                <div className="comparison-item">
                  <span>Faster Algorithm:</span>
                  <span className="highlight">
                    {algorithmComparison.faster_algorithm === 'ac3' ? '🔗 Arc Consistency' : '✂️ Pruning'}
                  </span>
                </div>
                <div className="comparison-item">
                  <span>Time Difference:</span>
                  <span>{algorithmComparison.time_difference.toFixed(3)}s</span>
                </div>
                <div className="comparison-item">
                  <span>Efficiency Ratio:</span>
                  <span>
                    {algorithmComparison.efficiency_ratio === Infinity ? '∞' : 
                     algorithmComparison.efficiency_ratio === 0 ? '0' :
                     algorithmComparison.efficiency_ratio.toFixed(2)}x
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
