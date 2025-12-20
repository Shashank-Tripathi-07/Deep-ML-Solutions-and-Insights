"""

171. Minimax Algorithm for Tic-Tac-Toe [Medium] 

Implement the Minimax algorithm to choose the best move for a Tic-Tac-Toe AI player.
Given a Tic-Tac-Toe board and a player (either 'X' or 'O'),
write a function that returns the optimal next move as a tuple (row, col).
Your function should assume both players play optimally and return a move that maximizes the
AI's chance of winning (or minimizes the chance of losing if no win is possible).
The board is given as a 3x3 NumPy array with entries 'X', 'O', or '',
and the player to move is a string ('X' or 'O').

Do not use any external game libraries.

Example:
Input:
import numpy as np
board = np.array([['X', 'O', 'X'], ['', 'O', ''], ['', '', '']])
move = minimax_tictactoe(board, 'X')
print(move)
Output:
(2, 2)
Reasoning:
X can win by playing in (2, 2) if both sides play optimally.


Insights and Approach: 

 -Core Concept: 
 --Minimax is a recursive algorithm that:

-- Maximizes the AI player's score when it's the AI's turn
-- Minimizes the AI player's score when it's the opponent's turn (assuming optimal opponent play)
-- Evaluates all possible future game states to choose the best current move

-Scoring System:

+10: AI wins
-10: Opponent wins
0: Draw (tie game)

-How It Works:

1. Base Cases (Terminal States)
--The recursion stops when the game reaches an end state:

--AI player wins → return +10
--Opponent wins → return -10
--Board is full (draw) → return 0

2. Recursive Evaluation
--For non-terminal states:
--Maximizing Phase (AI's turn):

--Try all available moves
--For each move, simulate it and recursively call minimax for opponent's turn
--Choose the move with the highest score
--Undo the move (backtrack) and try next option

--Minimizing Phase (Opponent's turn):

--Try all available moves
--For each move, simulate it and recursively call minimax for AI's turn
--Assume opponent chooses the move with the lowest score for AI
--Undo the move and try next option

3. Move Selection
-The main function:

--Iterates through all available moves
--Simulates each move
--Calls minimax to evaluate the resulting game state
--Selects the move with the highest evaluation score
--Returns the optimal move as (row, col)


"""

#SOLUTION: 

import numpy as np

def is_winner(board, player):
    # Check rows, columns, and diagonals.
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    """
    Returns the optimal move (row, col) for the given player ('X' or 'O') on the current board using Minimax.
    Args:
        board: 3x3 NumPy array with entries 'X', 'O', or ''
        player: 'X' or 'O'
    Returns:
        Tuple (row, col) for the optimal move
    """
    def minimax(board, is_maximizing, current_player):
        """
        Recursive minimax function.
        Returns the score of the board state.
        """
        opponent = 'O' if current_player == 'X' else 'X'
        
        # Base cases: check if game is over.
        if is_winner(board, player):
            return 10  # AI wins
        if is_winner(board, 'O' if player == 'X' else 'X'):
            return -10  # Opponent wins
        if is_full(board):
            return 0  # Draw
        
        if is_maximizing:
            # Maximizing player (AI).
            max_score = -float('inf')
            for move in get_available_moves(board):
                i, j = move
                board[i, j] = current_player
                score = minimax(board, False, opponent)
                board[i, j] = ''  # Undo move
                max_score = max(max_score, score)
            return max_score
        else:
            # Minimizing player (opponent score). 
            min_score = float('inf')
            for move in get_available_moves(board):
                i, j = move
                board[i, j] = current_player
                score = minimax(board, True, opponent)
                board[i, j] = ''  # Undo move
                min_score = min(min_score, score)
            return min_score
    
    # Find the best move for the current player.
    best_score = -float('inf')    #representing the worst possible score. 
    best_move = None
    
    for move in get_available_moves(board):
        i, j = move
        board[i, j] = player
        score = minimax(board, False, 'O' if player == 'X' else 'X')
        board[i, j] = ''  # Undo move
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move
