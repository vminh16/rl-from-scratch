import numpy as np

__all__ = ["TicTacToe"]

# 3x3 board
WIN_STATES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  #row
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  #column
    (0, 4, 8), (2, 4, 6)              #diagonal
]
class TicTacToe:
	def __init__(self):
		self.board = np.zeros(9, dtype = int)
		self.rng = None

	def reset(self, rng) -> np.ndarray:
		"""
		reset the board --> new game
		"""
		self.board = np.zeros(9, dtype = int)
		self.rng = rng
		return self.board.copy()


	def check_win(self, board) -> int:
		"""
		input: board
		output: 1 - win, -1 lose, 0 draw
		"""
		for a, b, c in WIN_STATES:
			if board[a] != 0 and board[a] == board[b] == board[c]:
				return int(board[a])
		return 0

	def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
		"""
		Implement 1 action
		input: action
		output:
		next_obs, reward, terminate, truncate, info (win, draw, lose)
		"""	
		#check valid action
		if self.board[action] != 0:
			raise ValueError(f"Cell {action} was piece is in place, can not move this cell !")

		#agent turn
		self.board[action] = 1
		if self.check_win(self.board) == 1:
			return self.board.copy(), 1, True, False, {"outcome": "win"}
		if 0 not in self.board:
			return self.board.copy(), 0, True, False, {"outcome": "draw"}

		#random oposite turn
		empty_cell = np.where(self.board == 0)[0]
		opp_action = self.rng.choice(empty_cell)
		self.board[opp_action] = -1

		if self.check_win(self.board) == -1:
			return self.board.copy(), -1, True, False, {"outcome": "loss"}
		if not np.any(self.board == 0):
			return self.board.copy(), 0, True, False, {"outcome": "draw"}

		return self.board.copy(), 0, False, False, {}