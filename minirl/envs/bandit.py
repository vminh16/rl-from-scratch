import numpy as np 

__all__ = ['Vectorize10ArmsBandit']

class Vectorize10ArmsBandit:
	def __init__(self, arms: int = 10, envs: int = 2000, non_stationary = False, walk_std = 0.3):
		self.arms = arms
		self.envs = envs
		self.non_stationary = non_stationary
		self.walk_std = walk_std
		self.q_star = None

		self.optimal_actions = None


	def reset(self, rng: np.random.Generator) -> None:
		# reset matrix to defaul value with normal distribution N(0, 1)
		self.q_star = rng.normal(0.0, 1.0, size = (self.envs, self.arms))

		self.optimal_actions = np.argmax(self.q_star, axis = 1) #shape: (2000,)

	def step(self, action: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
		
		indicates = np.arange(self.envs)
		selected_q = self.q_star[indicates, action] #shape: (2000,)

		reward = rng.normal(selected_q, 1.0)
		is_optimal = (action == self.optimal_actions)
		
		# new q_star with random walk
		if self.non_stationary:
			random_walk = rng.normal(0.0, self.walk_std, size = self.q_star.shape)
			self.q_star += random_walk
			self.optimal_actions = np.argmax(self.q_star, axis = 1)

		return reward, is_optimal