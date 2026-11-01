import numpy as np 

__all__ = ['EpsilonGreedy', 'UCBBandit', 'GradientBandit']


class EpsilonGreedy:
	def __init__(self, arms: int = 10, envs: int = 2000, epsilon: float = 0.1, initial_q = 0.0, alpha: float = 0.0):
		self.arms = arms
		self.envs = envs
		self.epsilon = epsilon
		self.initial_q = initial_q
		self.alpha = alpha

		self.Q = np.full((envs, arms), initial_q, dtype = np.float64)
		self.N = np.zeros((envs, arms), dtype = np.int64)

	def reset(self) -> None:
		# reset for new game
		self.Q.fill(self.initial_q)
		self.N.fill(0)

	def act(self, rng: np.random.Generator) -> np.ndarray:
		# pick action with epsilon greedy algorithm
		tie_breaker = rng.normal(0.0, 1e-6, self.Q.shape)
		greedy_actions = np.argmax(self.Q + tie_breaker, axis = 1)

		# random choice 
		random_actions = rng.integers(0, self.arms, size = self.envs) #shape: (2000,)

		
		mask_envs = rng.random(size = self.envs) < self.epsilon #shape; (2000,)

		return np.where(mask_envs, random_actions, greedy_actions)

	def update(self, actions: np.ndarray, reward: np.ndarray) -> None:

		row_indicates = np.arange(0, self.envs)
		self.N[row_indicates, actions] += 1
		error = reward - self.Q[row_indicates, actions]
		# nonstationary with alpha step size is better
		if self.alpha != 0.0:
			self.Q[row_indicates, actions] += self.alpha * error 
		else:
			step_size = 1 / self.N[row_indicates, actions]
			self.Q[row_indicates, actions] += step_size * error


class UCBBandit:
	def __init__(self, arms: int = 10, envs: int = 2000, c: float = 2.0):
		self.arms = arms
		self.envs = envs 
		self.c = float(c)
		self.t = 1 # start at 1

		self.Q = np.zeros((envs, arms), dtype = np.float64)
		self.N = np.zeros((envs, arms), dtype = np.int64)

	def reset(self) -> None:
		self.Q.fill(0.0)
		self.N.fill(0)
		self.t = 1

	def act(self, rng: np.random.Generator) -> np.ndarray:
		ucb_value = np.zeros_like(self.Q)
		untried = self.N == 0
		tried = ~untried

		# UCB estimate with N != 0
		if np.any(tried):
			ucb_value[tried] = self.Q[tried] + self.c * np.sqrt(np.log(self.t) / self.N[tried])

		ucb_value[untried] = 1e7 # very large

		#tie-breaker 
		tie_breaker = rng.normal(0.0, 1e-6, size = self.Q.shape)
		return np.argmax(ucb_value + tie_breaker, axis =1 ) #shape = (2000, 1)

	def update(self, action: np.ndarray, reward: np.ndarray) -> None:
		self.t += 1
		row_indicates = np.arange(0, self.envs)
		self.N[row_indicates, action] += 1
		error = reward - self.Q[row_indicates, action] 
		#averange reward
		self.Q[row_indicates, action] += error / self.N[row_indicates, action]


class GradientBandit:
	def __init__(self, arms: int = 10, envs: int = 2000, alpha: float = 0.1, use_baseline: bool = True):
		self.arms = arms
		self.envs = envs
		self.alpha = alpha
		self.use_baseline = use_baseline
		self.t = 0
		
		self.H = np.zeros((envs, arms), dtype = np.float64)
		self.pi = np.full((envs, arms), 1 / self.arms, dtype = np.float64)
		self.average_reward = np.zeros(envs, dtype = np.float64) #baseline

	def reset(self) -> None:
		self.H.fill(0.0)
		self.pi.fill( 1 / self.arms)
		self.average_reward.fill(0.0)
		self.t = 0

	def act(self, rng: np.random.Generator) -> np.ndarray:
		#sotfmax
		max_h = np.max(self.H, axis = 1, keepdims = True)
		exp_h = np.exp(self.H - max_h)
		self.pi = exp_h/ np.sum(exp_h, axis = 1, keepdims = True)

		# pick random action with base CDF
		CDF_pi = np.cumsum(self.pi, axis = 1) #(2000, 10)
		r = rng.random(size=(self.envs, 1)) #(2000, 1)
		action = np.sum(r > CDF_pi, axis = 1) #shape: (2000,)
		return np.clip(action, 0, self.arms - 1)

	def update(self, actions: np.ndarray, reward: np.ndarray) -> None:
		row_indicates = np.arange(0, self.envs)
		if self.use_baseline:
			b = self.average_reward.copy()
		else:
			b = np.zeros(self.envs, dtype = np.float64)

		diff = reward - b #(2000, 1)
		one_hot = np.zeros_like(self.pi)
		one_hot[row_indicates, actions] = 1
		self.H = self.H + self.alpha * diff[:, None] * (one_hot - self.pi) # (2000, 10) + (1, 1) + (2000, 1) * (2000, 10)
		if self.use_baseline:
			self.t += 1
			self.average_reward += (reward - self.average_reward) / self.t
