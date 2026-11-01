"""
tests/test_bandit.py
Test suite linh hoạt tự động nhận diện tên class và tham số của bạn:
- Tự động tìm kiếm class Môi trường trong minirl/envs/bandit.py (Vectorize10ArmsBandit, Vectorized10ArmedBandit, BanditEnv, ...)
- Tự động tìm kiếm các Agent trong minirl/agents/bandit_agents.py (EsilonGreedy, EpsilonGreedy, UCBBandit, GradientBandit, ...)
- Linh hoạt nhận cả (arms, envs) hoặc (n_arms, n_envs)
- Tự động bỏ qua (Skip) những thuật toán bạn chưa kịp viết thay vì báo lỗi dừng cả test suite
"""

import inspect
import numpy as np
import pytest

from minirl.utils.seeding import make_rng
import minirl.envs.bandit as bandit_mod
import minirl.agents.bandit_agents as agents_mod


# =====================================================================
# Bộ công cụ linh động tìm kiếm Class và ánh xạ tham số
# =====================================================================

def get_env_class():
    """Tự động tìm kiếm class môi trường Bandit trong minirl.envs.bandit."""
    # 1. Tìm theo các tên phổ biến bạn có thể đặt
    candidate_names = [
        "Vectorize10ArmsBandit",
        "Vectorized10ArmedBandit",
        "VectorizedBanditEnv",
        "BanditEnv",
        "Bandit",
    ]
    for name in candidate_names:
        if hasattr(bandit_mod, name):
            return getattr(bandit_mod, name)

    # 2. Quét mọi class có hàm reset và step
    for _, obj in inspect.getmembers(bandit_mod, inspect.isclass):
        if hasattr(obj, "reset") and hasattr(obj, "step"):
            return obj

    raise ImportError("Không tìm thấy class Môi trường Bandit nào trong minirl/envs/bandit.py")


def create_env(arms: int = 10, envs: int = 2000, **kwargs):
    """Khởi tạo môi trường, tự thích ứng với tham số arms/n_arms và envs/n_envs."""
    cls = get_env_class()
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    args = {}

    if "arms" in params:
        args["arms"] = arms
    elif "n_arms" in params:
        args["n_arms"] = arms

    if "envs" in params:
        args["envs"] = envs
    elif "n_envs" in params:
        args["n_envs"] = envs

    for k, v in kwargs.items():
        if k in params:
            args[k] = v

    return cls(**args)


def get_agent_class(agent_type: str):
    """Tìm class agent linh hoạt theo từ khóa."""
    type_lower = agent_type.lower()
    for name, obj in inspect.getmembers(agents_mod, inspect.isclass):
        name_lower = name.lower()
        if type_lower == "epsilon_greedy":
            if any(k in name_lower for k in ["esilon", "epsilon", "greedy"]):
                return obj
        elif type_lower == "ucb":
            if "ucb" in name_lower:
                return obj
        elif type_lower == "gradient":
            if any(k in name_lower for k in ["gradient", "softmax", "preference"]):
                return obj

    return None


def create_agent(agent_type: str, arms: int = 10, envs: int = 2000, **kwargs):
    """Khởi tạo agent linh hoạt theo tham số trong __init__."""
    cls = get_agent_class(agent_type)
    if cls is None:
        pytest.skip(f"Chưa tìm thấy class cho '{agent_type}' trong minirl/agents/bandit_agents.py (tạm thời bỏ qua)")

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    args = {}

    if "arms" in params:
        args["arms"] = arms
    elif "n_arms" in params:
        args["n_arms"] = arms

    if "envs" in params:
        args["envs"] = envs
    elif "n_envs" in params:
        args["n_envs"] = envs

    for k, v in kwargs.items():
        if k in params:
            args[k] = v
        elif k == "initial_q":
            if "q1" in params:
                args["q1"] = v
            elif "q0" in params:
                args["q0"] = v
            elif "initial_value" in params:
                args["initial_value"] = v

    return cls(**args)


def get_optimal_actions(env):
    """Trích xuất optimal_actions từ env dù bạn đặt tên là gì."""
    for attr in ["optimal_actions", "optimal_action", "best_actions", "best_action"]:
        val = getattr(env, attr, None)
        if val is not None:
            return val
    q_star = get_q_star(env)
    if q_star is not None:
        return np.argmax(q_star, axis=1)
    raise AttributeError("Env cần lưu thuộc tính optimal_actions (hoặc best_actions)")


def get_q_star(env):
    """Trích xuất ma trận giá trị thật q*(a)."""
    for attr in ["q_star", "q_true", "true_q", "values"]:
        val = getattr(env, attr, None)
        if val is not None:
            return val
    return None


# =====================================================================
# 1. Tests cho Môi trường Bandit Vector hóa
# =====================================================================

class TestBanditEnvironment:
    def test_env_init_and_reset_shapes(self):
        """Reset phải tạo bảng q*(a) kích thước (2000, 10) phân phối chuẩn N(0, 1)."""
        rng = make_rng(42)
        env = create_env(arms=10, envs=2000)
        env.reset(rng)

        q_star = get_q_star(env)
        assert q_star is not None, "Env cần có thuộc tính lưu ma trận q_star"
        assert q_star.shape == (2000, 10), f"q_star phải có hình dạng (2000, 10), thực tế: {q_star.shape}"

        opt_act = get_optimal_actions(env)
        assert opt_act.shape == (2000,), f"optimal_actions phải có hình dạng (2000,), thực tế: {opt_act.shape}"
        np.testing.assert_array_equal(opt_act, np.argmax(q_star, axis=1))

        # Kiểm tra phân phối q*(a) ~ N(0, 1)
        mean_q = float(np.mean(q_star))
        std_q = float(np.std(q_star))
        assert abs(mean_q) < 0.05, f"Kỳ vọng mean ~ 0, thực tế {mean_q}"
        assert abs(std_q - 1.0) < 0.05, f"Kỳ vọng std ~ 1, thực tế {std_q}"

    def test_env_step_vectorized(self):
        """1 lời gọi step() xử lý đồng loạt cả 2.000 bài toán (không dùng vòng lặp)."""
        rng = make_rng(42)
        env = create_env(arms=10, envs=2000)
        env.reset(rng)

        opt_act = get_optimal_actions(env)
        rewards, is_optimal = env.step(opt_act, rng)

        assert rewards.shape == (2000,), f"rewards phải là mảng (2000,), thực tế {rewards.shape}"
        assert is_optimal.shape == (2000,), f"is_optimal phải là mảng (2000,), thực tế {is_optimal.shape}"
        assert np.all(is_optimal == True), "Khi kéo tay tối ưu, is_optimal phải toàn True"

        # Cố tình kéo lệch tay tối ưu
        sub_act = (opt_act + 1) % 10
        _, is_optimal_sub = env.step(sub_act, rng)
        assert np.all(is_optimal_sub == False), "Khi kéo lệch tay, is_optimal phải toàn False"

    def test_env_non_stationary_random_walk(self):
        """Bài tập 2.5: Hỗ trợ non_stationary với bước đi ngẫu nhiên q* sau mỗi step."""
        rng = make_rng(42)
        try:
            env = create_env(arms=10, envs=2000, non_stationary=True, walk_std=0.01)
        except TypeError:
            pytest.skip("Env chưa hỗ trợ tham số non_stationary (Bài tập 2.5)")

        env.reset(rng)
        q_before = get_q_star(env).copy()
        actions = np.zeros(2000, dtype=int)
        env.step(actions, rng)
        q_after = get_q_star(env).copy()

        assert not np.array_equal(q_before, q_after), "q* phải thay đổi sau mỗi step khi non_stationary=True"


# =====================================================================
# 2. Tests cho Agent Epsilon-Greedy
# =====================================================================

class TestEpsilonGreedyAgent:
    def test_init_shapes(self):
        """Q và N phải có hình dạng (2000, 10) toàn số 0."""
        agent = create_agent("epsilon_greedy", arms=10, envs=2000, epsilon=0.1)
        assert agent.Q.shape == (2000, 10)
        assert agent.N.shape == (2000, 10)
        assert np.all(agent.Q == 0.0)
        assert np.all(agent.N == 0)

    def test_pure_greedy_action(self):
        """Khi epsilon = 0, bắt buộc luôn chọn argmax của Q."""
        rng = make_rng(42)
        agent = create_agent("epsilon_greedy", arms=10, envs=2000, epsilon=0.0)
        agent.Q[:, 6] = 50.0  # Gán tay 6 cao nhất

        actions = agent.act(rng)
        assert actions.shape == (2000,), f"act() phải trả về mảng shape (2000,), thực tế: {actions.shape}"
        assert np.all(actions == 6), "Thuần tham lam (eps=0) phải luôn chọn tay có Q lớn nhất"

    def test_pure_explore_distribution(self):
        """Khi epsilon = 1.0, các hành động phải rải đều khắp 10 tay."""
        rng = make_rng(42)
        agent = create_agent("epsilon_greedy", arms=10, envs=10000, epsilon=1.0)
        actions = agent.act(rng)
        counts = np.bincount(actions, minlength=10)
        for arm in range(10):
            assert 800 <= counts[arm] <= 1200, f"Tay {arm} phân bố không đều: {counts[arm]}"

    def test_sample_average_update(self):
        """Kiểm tra toán học công thức cập nhật Trung bình mẫu: Q <- Q + 1/N * (R - Q)."""
        agent = create_agent("epsilon_greedy", arms=10, envs=1, epsilon=0.0)
        actions = np.array([1])

        # Step 1: R = 4.0 -> Q = 4.0
        agent.update(actions, np.array([4.0]))
        assert agent.N[0, 1] == 1
        assert pytest.approx(agent.Q[0, 1]) == 4.0

        # Step 2: R = 2.0 -> Q = (4 + 2) / 2 = 3.0
        agent.update(actions, np.array([2.0]))
        assert agent.N[0, 1] == 2
        assert pytest.approx(agent.Q[0, 1]) == 3.0


# =====================================================================
# 3. Tests cho Optimistic Initial Values
# =====================================================================

class TestOptimisticInitialValues:
    def test_optimistic_exploration(self):
        """Q ban đầu = 5.0, dù eps = 0 thì agent vẫn phải khám phá đủ 10 tay ở các bước đầu."""
        rng = make_rng(42)
        env = create_env(arms=10, envs=2000)
        env.reset(rng)
        try:
            agent = create_agent("epsilon_greedy", arms=10, envs=2000, epsilon=0.0, alpha=0.1, initial_q=5.0)
        except TypeError:
            pytest.skip("Agent chưa nhận tham số alpha hoặc initial_q")

        for _ in range(20):
            actions = agent.act(rng)
            rewards, _ = env.step(actions, rng)
            agent.update(actions, rewards)

        # Mọi tay gạt đều đã được gạt ít nhất 1 lần
        pulls_per_arm = np.sum(agent.N, axis=0)
        assert np.all(pulls_per_arm > 0), "Khởi tạo lạc quan Q1=5 phải kích thích khám phá đủ 10 tay"


# =====================================================================
# 4. Tests cho UCB (Upper Confidence Bound)
# =====================================================================

class TestUCB:
    def test_ucb_pulls_all_arms_first(self):
        """10 bước đầu tiên, UCB phải gạt đủ 10 tay khác nhau chưa từng thử."""
        rng = make_rng(42)
        env = create_env(arms=10, envs=100)
        env.reset(rng)
        agent = create_agent("ucb", arms=10, envs=100, c=2.0)

        for _ in range(10):
            actions = agent.act(rng)
            rewards, _ = env.step(actions, rng)
            agent.update(actions, rewards)

        assert np.all(agent.N == 1), "Trong 10 bước đầu, mỗi tay phải được gạt đúng 1 lần"


# =====================================================================
# 5. Tests cho Gradient Bandit
# =====================================================================

class TestGradientBandit:
    def test_softmax_probabilities(self):
        """Xác suất Softmax ban đầu của 10 tay phải bằng 0.1 và tổng xác suất = 1.0."""
        rng = make_rng(42)
        agent = create_agent("gradient", arms=10, envs=500, alpha=0.1)

        _ = agent.act(rng)
        probs = getattr(agent, "probs", getattr(agent, "pi", None))
        assert probs is not None, "Agent cần lưu xác suất softmax vào thuộc tính probs hoặc pi"
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(probs, 0.1, atol=1e-5)
