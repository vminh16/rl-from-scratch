import numpy as np

__all__ = ["make_rng"]

def make_rng(seed: int) -> np.random.Generator:
	"""
	Tạo rng độc lập cho mỗi lần chạy thử → kết quả tái lập được
	"""
	return np.random.default_rng(seed)
