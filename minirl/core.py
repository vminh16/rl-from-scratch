"""
Định nghĩa giao diện (interfaces) 
Vòng lặp tương tác chuẩn giữa Agent và Môi trường
"""
# minirl/core.py

from dataclasses import dataclass, field
from typing import Any, Protocol 
import numpy as np 

@dataclass
class Transition:
	"""
	Lưu trữ toàn bộ pha chuyển trạng thái
	Phân biệt rõ hai trạng thái kết thúc
	"""
	obs: Any
	action: Any
	reward: float
	next_obs: Any
	terminated: bool # kết thúc thật: thắng, thua, hòa, rớt vực, ...
	truncated: bool # kết thúc vì hết giờ; bootstrap được từ next_obs
	info: dict = field(default_factory=dict)

class Env(Protocol):
	def reset(self, rng: np.random.Generator) -> Any: ...
	def step(self, action) -> tuple[Any, float, bool, bool, dict]: ... # next_obs, reward, terminate, truncate, info

class Agent(Protocol):
	def act(self, obs, rng: np.random.Generator, greedy: bool = False): ...
	def update(self, t: Transition) -> dict: ... # số liệu để log

def run_episode(env: Env, agent: Agent, rng: np.random.Generator,
				train: bool = True, max_steps: int = 10_000) -> dict:

	"""
	vòng lặp duy nhất của cả repo: dùng chung để so sánh công bằng
	"""
	obs, stats = env.reset(rng), {"return": 0.0, "length": 0}
	while True:
		action = agent.act(obs, rng, greedy=not train)
		next_obs, reward, terminated, truncated, info = env.step(action)
		if train:
			t = Transition(obs, action, reward, next_obs, terminated, truncated, info)
			for k, v in agent.update(t).items():
				stats[k] = stats.get(k, 0.0) + v

		stats["return"] += reward
		stats["length"] += 1
		obs = next_obs
		if terminated or truncated or stats["length"] >= max_steps:
			stats.update(info)
			return stats




		