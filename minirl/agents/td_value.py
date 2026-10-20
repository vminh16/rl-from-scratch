import numpy as np 
from typing import Any
from minirl.core import Transition 

__all__ = ["AfterstateTD"] # Temporal Difference

class AfterstateTD:
	def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, default_value: float = 0.5):
		self.alpha = alpha
		self.epsilon = epsilon
		self.default_value = default_value

		# value table 
		self.v_table: dict[tuple, float] = {} # sate and value of this state
		# Lưu afterstate của bước trước đó để cập nhật sai số TD
		self.prev_afterstate: tuple | None = None

	def _get_v(self, afterstate: tuple):
		return self.v_table.get(afterstate, self.default_value)

	def _make_afterstate(self, obs: np.ndarray | Any, action: int):
		"""
		input: 
		- obs: observability this board
		- action: action the agent do 
		output:
		- the state after action
		"""
		board = np.array(obs).flatten()	
		board[action] = 1
		return tuple(board)	

	def act (self, obs: np.ndarray | Any, rng: np.random.Generator, greedy: bool = False):
		"""
		Chọn nước đi theo chính sách epsilon-greedy dựa trên Afterstate
		Input: 
			obs: observabily of agent at t
			rng: random choice
			greedy: short term value with greedy algorithm
		Ouput:
			int: action after explore and Exploitation with Tmeporal Differene
		"""
		obs_flatten = np.array(obs).flatten()
		empty_cell = np.where(obs_flatten == 0)[0]

		# expore random in empty cell
		if not greedy and (rng.random() < self.epsilon):
			action = int(rng.choice(empty_cell))
			return action

		# Exploitation
		best_v = -float("inf")
		best_action = []

		for action in empty_cell:
			afterstate = self._make_afterstate(obs_flatten, action)
			val = self._get_v(afterstate)

			if val > best_v:
				best_v =val 
				best_action = [action]

			elif np.isclose(val, best_v):
				best_action.append(action)

		# muilti value largest -> random choice
		choice_action = int(rng.choice(best_action))
		return choice_action

	def update(self, t: Transition) -> dict[str, float]:
		"""
		TD value function update V_(t-1) = V_(t-1) + alpha(V_t - V_(t-1))
		Input: 
			t: Transition state 
		Output:
			dict: {str. float} -> td_erorr
		"""
		curr_afterstate = self._make_afterstate(t.obs, t.action)
		curr_v = self._get_v(curr_afterstate)
		td_error = 0
		# update 
		if self.prev_afterstate is not None:
			prev = self._get_v(self.prev_afterstate)
			td_error = curr_v - prev
			self.v_table[self.prev_afterstate] = prev + self.alpha * td_error
		
		if t.terminated:
			target = 1.0 if t.reward > 0 else (0.5 if t.reward == 0 else 0)
			
			teminated_td_error = target - curr_v
			
			self.v_table[curr_afterstate] = curr_v + self.alpha * teminated_td_error

			td_error = teminated_td_error

			# reset for the new game
			self.prev_afterstate = None
		else:
			self.prev_afterstate = curr_afterstate
		

		return {"td_error": float(abs(td_error))}

