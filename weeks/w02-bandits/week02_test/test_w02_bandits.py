"""
Test Tuần 02: testbed bandit 10 tay (S&B chương 2).

File này là ĐẶC TẢ: bạn tự viết code trong minirl/envs/bandit.py và
minirl/agents/bandit_agents.py cho đến khi test qua. Test không chứa lời giải.

======================== HỢP ĐỒNG GIAO DIỆN ========================
Môi trường (minirl/envs/bandit.py), tên lớp một trong ENV_NAMES:
    Env(arms=10, envs=2000, non_stationary=False, walk_std=0.01)
    .reset(rng)                  -> tạo q_star (envs, arms) ~ N(0,1), optimal_actions (envs,)
    .step(actions, rng)          -> (rewards (envs,) float, is_optimal (envs,) bool)
    .q_star, .optimal_actions    -> thuộc tính công khai
    non_stationary=True: sau mỗi step, q_star += N(0, walk_std^2) và
    optimal_actions được tính lại theo q_star mới (Bài tập 2.5).

Mọi agent: .act(rng) -> actions (envs,) int ; .update(actions, rewards) -> None
                       (act KHÔNG được làm thay đổi trạng thái học)

EpsilonGreedy (tên trong EPS_NAMES):
    (arms=10, envs=2000, epsilon=0.1, alpha=None, initial_q=0.0)
    .Q, .N (envs, arms) ; .reset()
    alpha=None  -> trung bình mẫu (bước 1/N); alpha=số -> bước không đổi.
Optimistic: cùng lớp trên với initial_q=5.0, alpha=0.1, epsilon=0.0
            (hoặc một lớp tên trong OPT_NAMES nhận cùng tham số).
UCB (tên trong UCB_NAMES): (arms=10, envs=2000, c=2.0)
    .Q, .N ; tay chưa thử (N=0) được coi là tối đa; lần gọi act thứ t (t=1,2,...)
    chọn argmax Q + c*sqrt(ln t / N).
GradientBandit (tên trong GRAD_NAMES): (arms=10, envs=2000, alpha=0.1, use_baseline=True)
    .H (envs, arms) khởi tạo 0 ; act() tính lại softmax từ H và lưu vào .probs (hoặc .pi)
    baseline = trung bình các phần thưởng TRƯỚC bước hiện tại (bẫy trong lộ trình:
    không được cập nhật baseline trước khi dùng nó cho H).
Nếu bạn đặt tên khác, sửa các hằng *_NAMES hoặc tham số ngay dưới đây.
====================================================================
"""

import time

import numpy as np
import pytest

import minirl.agents.bandit_agents as agents_mod
import minirl.envs.bandit as bandit_mod
from minirl.utils.seeding import make_rng

ENV_NAMES = ("Vectorize10ArmsBandit", "VectorizedBandit", "BanditEnv")
EPS_NAMES = ("EpsilonGreedy", "EsilonGreedy")
OPT_NAMES = ("OptimisticGreedy", "OptimisticEpsilonGreedy") + EPS_NAMES
UCB_NAMES = ("UCB", "UCBBandit", "UCBAgent")
GRAD_NAMES = ("GradientBandit", "GradientBanditAgent", "Gradient")


# ---------------------------------------------------------------- tiện ích
def find_class(module, names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def make_env(envs=2000, arms=10, **kw):
    cls = find_class(bandit_mod, ENV_NAMES)
    assert cls is not None, f"Không thấy lớp môi trường nào trong {ENV_NAMES}"
    try:
        return cls(arms=arms, envs=envs, **kw)
    except TypeError:
        if kw:
            pytest.skip(f"Env chưa nhận tham số {list(kw)} (xem hợp đồng đầu file)")
        raise


def make_eps(envs, arms=10, epsilon=0.1, **kw):
    cls = find_class(agents_mod, EPS_NAMES)
    assert cls is not None, f"Không thấy lớp ε-greedy nào trong {EPS_NAMES}"
    try:
        return cls(arms=arms, envs=envs, epsilon=epsilon, **kw)
    except TypeError:
        if kw:
            pytest.skip(f"ε-greedy chưa nhận tham số {list(kw)} (xem hợp đồng đầu file)")
        raise


def make_optimistic(envs, arms=10):
    for name in OPT_NAMES:
        cls = getattr(agents_mod, name, None)
        if cls is None:
            continue
        for kw in (dict(epsilon=0.0, alpha=0.1, initial_q=5.0), dict(alpha=0.1, initial_q=5.0)):
            try:
                return cls(arms=arms, envs=envs, **kw)
            except TypeError:
                continue
    pytest.skip("Chưa có agent lạc quan (initial_q=5.0, alpha=0.1)")


def make_ucb(envs, arms=10, c=2.0):
    cls = find_class(agents_mod, UCB_NAMES)
    if cls is None:
        pytest.skip(f"Chưa cài UCB (tên một trong {UCB_NAMES})")
    return cls(arms=arms, envs=envs, c=c)


def make_gradient(envs, arms=10, alpha=0.1, **kw):
    cls = find_class(agents_mod, GRAD_NAMES)
    if cls is None:
        pytest.skip(f"Chưa cài gradient bandit (tên một trong {GRAD_NAMES})")
    try:
        return cls(arms=arms, envs=envs, alpha=alpha, **kw)
    except TypeError:
        if kw:
            pytest.skip(f"Gradient bandit chưa nhận tham số {list(kw)}")
        raise


def get_probs(agent):
    for name in ("probs", "pi"):
        if hasattr(agent, name):
            return np.asarray(getattr(agent, name))
    raise AssertionError("Gradient bandit cần lưu xác suất softmax vào .probs (hoặc .pi)")


def run_experiment(env, agent, steps, seed, shift=0.0):
    """Chạy `steps` bước; trả về (phần thưởng TB theo bước, % tay tối ưu theo bước)."""
    rng = make_rng(seed)
    env.reset(rng)
    if shift:
        env.q_star = env.q_star + shift  # dịch đều nên tay tối ưu không đổi
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    for t in range(steps):
        actions = agent.act(rng)
        r, is_opt = env.step(actions, rng)
        agent.update(actions, r)
        rewards[t] = r.mean()
        optimal[t] = is_opt.mean()
    return rewards, optimal


def tail(x, n=100):
    return float(np.mean(x[-n:]))


# ============================================================ 1. MÔI TRƯỜNG
class TestEnv:
    def test_reset_shapes_and_optimal_actions(self):
        env = make_env()
        env.reset(make_rng(0))
        assert env.q_star.shape == (2000, 10)
        assert env.optimal_actions.shape == (2000,)
        np.testing.assert_array_equal(env.optimal_actions, env.q_star.argmax(axis=1))

    def test_q_star_is_standard_normal(self):
        env = make_env()
        env.reset(make_rng(0))
        assert abs(env.q_star.mean()) < 0.03
        assert abs(env.q_star.std() - 1.0) < 0.03

    def test_reset_draws_new_problems_and_is_reproducible(self):
        env = make_env(envs=50)
        env.reset(make_rng(1))
        first = env.q_star.copy()
        env.reset(make_rng(2))
        assert not np.array_equal(first, env.q_star), "reset phải sinh bài toán mới"
        env.reset(make_rng(1))
        np.testing.assert_array_equal(first, env.q_star)  # cùng seed, cùng bài toán

    def test_step_output_shapes_and_types(self):
        env = make_env()
        rng = make_rng(0)
        env.reset(rng)
        rewards, is_opt = env.step(np.zeros(2000, dtype=int), rng)
        assert rewards.shape == (2000,) and rewards.dtype.kind == "f"
        assert is_opt.shape == (2000,) and is_opt.dtype == bool

    def test_reward_uses_the_pulled_arm_of_each_problem(self):
        """Bắt lỗi chỉ số: mỗi bài toán phải trả thưởng từ đúng tay mình chọn."""
        envs, arms = 200, 10
        env = make_env(envs=envs, arms=arms)
        rng = make_rng(0)
        env.reset(rng)
        env.q_star = 1000.0 * np.arange(envs)[:, None] + 100.0 * np.arange(arms)[None, :]
        env.optimal_actions = env.q_star.argmax(axis=1)
        actions = np.arange(envs) % arms
        rewards, _ = env.step(actions, rng)
        expected = env.q_star[np.arange(envs), actions]
        assert np.all(np.abs(rewards - expected) < 6.0), "phần thưởng phải ~ N(q*(a), 1)"

    def test_reward_noise_is_unit_normal_around_q_star(self):
        env = make_env()
        rng = make_rng(3)
        env.reset(rng)
        actions = rng.integers(0, 10, size=2000)
        noise = []
        for _ in range(100):
            r, _ = env.step(actions, rng)
            noise.append(r - env.q_star[np.arange(2000), actions])
        noise = np.concatenate(noise)
        assert abs(noise.mean()) < 0.02
        assert abs(noise.std() - 1.0) < 0.02

    def test_is_optimal_flags(self):
        env = make_env()
        rng = make_rng(0)
        env.reset(rng)
        best = env.optimal_actions.copy()
        _, hit = env.step(best, rng)
        assert hit.all()
        _, miss = env.step((best + 1) % 10, rng)
        assert not miss.any()

    def test_stationary_env_keeps_q_star_fixed(self):
        env = make_env(envs=100)
        rng = make_rng(0)
        env.reset(rng)
        before = env.q_star.copy()
        for _ in range(20):
            env.step(np.zeros(100, dtype=int), rng)
        np.testing.assert_array_equal(before, env.q_star)

    def test_one_step_call_handles_all_problems_fast(self):
        """Vector hóa: 1000 bước x 2000 bài toán phải chạy rất nhanh (không vòng lặp theo bài toán)."""
        env = make_env()
        rng = make_rng(0)
        env.reset(rng)
        actions = np.zeros(2000, dtype=int)
        start = time.perf_counter()
        for _ in range(1000):
            env.step(actions, rng)
        assert time.perf_counter() - start < 5.0


class TestNonStationaryEnv:
    def test_q_star_takes_random_walk_steps(self):
        env = make_env(non_stationary=True, walk_std=0.01)
        rng = make_rng(0)
        env.reset(rng)
        before = env.q_star.copy()
        env.step(np.zeros(2000, dtype=int), rng)
        delta = env.q_star - before
        assert abs(delta.mean()) < 0.0005
        assert abs(delta.std() - 0.01) < 0.0005

    def test_walk_accumulates_like_sqrt_of_steps(self):
        env = make_env(non_stationary=True, walk_std=0.01)
        rng = make_rng(0)
        env.reset(rng)
        start = env.q_star.copy()
        for _ in range(2000):
            env.step(np.zeros(2000, dtype=int), rng)
        drift = (env.q_star - start).std()
        assert abs(drift - 0.01 * np.sqrt(2000)) < 0.04  # ~0.447

    def test_optimal_actions_follow_the_walk(self):
        env = make_env(envs=300, non_stationary=True, walk_std=0.05)
        rng = make_rng(0)
        env.reset(rng)
        for _ in range(200):
            env.step(np.zeros(300, dtype=int), rng)
        np.testing.assert_array_equal(env.optimal_actions, env.q_star.argmax(axis=1))

    def test_constant_step_size_tracks_better_than_sample_average(self):
        """Bài tập 2.5: bài toán đổi theo thời gian thì α không đổi thắng trung bình mẫu."""
        results = {}
        for name, kw in (("sample", {}), ("constant", dict(alpha=0.1))):
            env = make_env(envs=500, non_stationary=True, walk_std=0.01)
            agent = make_eps(500, epsilon=0.1, **kw)
            _, opt = run_experiment(env, agent, steps=5000, seed=0)
            results[name] = tail(opt, 500)
        assert results["constant"] > results["sample"] + 0.06, results


# ====================================================== 2. ε-GREEDY (TB MẪU)
class TestEpsilonGreedy:
    def test_initial_state(self):
        agent = make_eps(2000)
        assert agent.Q.shape == (2000, 10) and agent.N.shape == (2000, 10)
        assert np.all(agent.Q == 0) and np.all(agent.N == 0)

    def test_greedy_picks_argmax_in_every_problem(self):
        agent = make_eps(2000, epsilon=0.0)
        agent.Q[:] = make_rng(0).normal(size=(2000, 10))
        actions = agent.act(make_rng(1))
        assert actions.shape == (2000,)
        np.testing.assert_array_equal(actions, agent.Q.argmax(axis=1))

    def test_ties_are_broken_randomly(self):
        """Q ban đầu toàn 0: nếu luôn chọn tay 0 thì không bao giờ khám phá."""
        agent = make_eps(20000, epsilon=0.0)
        counts = np.bincount(agent.act(make_rng(0)), minlength=10)
        assert counts.min() > 1700 and counts.max() < 2300, counts

    def test_epsilon_one_is_uniform(self):
        agent = make_eps(20000, epsilon=1.0)
        counts = np.bincount(agent.act(make_rng(0)), minlength=10)
        assert counts.min() > 1700 and counts.max() < 2300, counts

    def test_exploration_coin_is_flipped_per_problem(self):
        """ε=0.1, tay 0 tốt nhất: ~9% bài toán chọn tay khác (0,1 x 9/10)."""
        agent = make_eps(100000, epsilon=0.1)
        agent.Q[:] = 0.0
        agent.Q[:, 0] = 1.0
        frac_other = float(np.mean(agent.act(make_rng(0)) != 0))
        assert abs(frac_other - 0.09) < 0.01, frac_other

    def test_act_does_not_change_state(self):
        agent = make_eps(100)
        agent.Q[:] = make_rng(0).normal(size=(100, 10))
        q_before, n_before = agent.Q.copy(), agent.N.copy()
        agent.act(make_rng(1))
        np.testing.assert_array_equal(agent.Q, q_before)
        np.testing.assert_array_equal(agent.N, n_before)

    def test_update_touches_only_the_pulled_arm(self):
        agent = make_eps(6, arms=4)
        actions = np.array([0, 1, 2, 3, 0, 1])
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        agent.update(actions, rewards)
        np.testing.assert_array_equal(agent.N.sum(axis=1), np.ones(6))
        assert np.all(agent.N[np.arange(6), actions] == 1)
        np.testing.assert_allclose(agent.Q[np.arange(6), actions], rewards)
        mask = np.ones((6, 4), dtype=bool)
        mask[np.arange(6), actions] = False
        assert np.all(agent.Q[mask] == 0.0)

    def test_sample_average_equals_running_mean_in_every_problem(self):
        envs, arms = 5, 3
        agent = make_eps(envs, arms=arms)
        rng = make_rng(0)
        sums = np.zeros((envs, arms))
        counts = np.zeros((envs, arms))
        for _ in range(60):
            actions = rng.integers(0, arms, size=envs)
            rewards = rng.normal(size=envs)
            agent.update(actions, rewards)
            sums[np.arange(envs), actions] += rewards
            counts[np.arange(envs), actions] += 1
        expected = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        np.testing.assert_allclose(agent.Q, expected, atol=1e-9)
        np.testing.assert_array_equal(agent.N, counts)

    def test_reset_clears_learning(self):
        agent = make_eps(10)
        agent.update(np.zeros(10, dtype=int), np.ones(10))
        agent.reset()
        assert np.all(agent.Q == 0) and np.all(agent.N == 0)

    def test_same_seed_same_trajectory(self):
        def trajectory(seed):
            env, agent = make_env(envs=50), make_eps(50)
            rng = make_rng(seed)
            env.reset(rng)
            out = []
            for _ in range(30):
                a = agent.act(rng)
                r, _ = env.step(a, rng)
                agent.update(a, r)
                out.append(a.copy())
            return np.array(out)

        np.testing.assert_array_equal(trajectory(5), trajectory(5))
        assert not np.array_equal(trajectory(5), trajectory(6))


# ================================================= 3. GIÁ TRỊ KHỞI TẠO LẠC QUAN
class TestOptimisticInitialValues:
    def test_initial_values_and_constant_step_update(self):
        agent = make_optimistic(1, arms=3)
        assert np.all(agent.Q == 5.0)
        agent.update(np.array([1]), np.array([1.0]))
        assert agent.Q[0, 1] == pytest.approx(4.6)  # 5 + 0.1 * (1 - 5)
        agent.update(np.array([1]), np.array([1.0]))
        assert agent.Q[0, 1] == pytest.approx(4.24)  # 4.6 + 0.1 * (1 - 4.6)
        assert agent.Q[0, 0] == 5.0 and agent.Q[0, 2] == 5.0
        assert agent.N[0, 1] == 2

    def test_constant_step_is_not_the_sample_average(self):
        agent = make_optimistic(1, arms=3)
        agent.update(np.array([0]), np.array([4.0]))
        agent.update(np.array([0]), np.array([2.0]))
        assert agent.Q[0, 0] == pytest.approx(4.61)  # 4.9 + 0.1 * (2 - 4.9), không phải 3.0

    def test_greedy_still_tries_every_arm_first(self):
        """Thưởng toàn 0 < 5: mỗi tay bị 'thất vọng' sau 1 lần nên 10 bước đầu thử đủ 10 tay."""
        envs = 50
        agent = make_optimistic(envs)
        picked = []
        for _ in range(10):
            a = agent.act(make_rng(len(picked)))
            agent.update(a, np.zeros(envs))
            picked.append(a)
        picked = np.array(picked)  # (10, envs)
        np.testing.assert_array_equal(np.sort(picked, axis=0), np.tile(np.arange(10)[:, None], (1, envs)))

    def test_tie_break_among_equal_optimistic_values_is_random(self):
        agent = make_optimistic(1000)
        assert len(np.unique(agent.act(make_rng(0)))) >= 8


# ============================================================================ 4. UCB
class TestUCB:
    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_first_ten_steps_pull_each_arm_once_without_division_by_zero(self):
        envs = 100
        env, agent = make_env(envs=envs), make_ucb(envs)
        rng = make_rng(0)
        env.reset(rng)
        picked = []
        for _ in range(10):
            a = agent.act(rng)
            r, _ = env.step(a, rng)
            agent.update(a, r)
            picked.append(a)
        assert np.all(agent.N == 1)
        np.testing.assert_array_equal(np.sort(np.array(picked), axis=0), np.tile(np.arange(10)[:, None], (1, envs)))

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_selection_follows_the_ucb_formula(self):
        envs, c = 50, 2.0
        env, agent = make_env(envs=envs), make_ucb(envs, c=c)
        rng = make_rng(0)
        env.reset(rng)
        rows = np.arange(envs)
        checked = mismatches = 0
        for t in range(1, 301):
            q, n = agent.Q.copy(), agent.N.copy()
            actions = agent.act(rng)
            untried = (n == 0).any(axis=1)
            assert np.all(n[rows[untried], actions[untried]] == 0), f"bước {t}: còn tay chưa thử mà chọn tay đã thử"
            full = ~untried
            if full.any():
                score = q[full] + c * np.sqrt(np.log(t) / n[full])
                chosen = score[np.arange(full.sum()), actions[full]]
                mismatches += int(np.sum(chosen < score.max(axis=1) - 1e-6))
                checked += int(full.sum())
            r, _ = env.step(actions, rng)
            agent.update(actions, r)
        assert checked > 0
        assert mismatches / checked < 0.02, f"{mismatches}/{checked} lựa chọn lệch công thức (lần act thứ t dùng ln t)"

    def test_update_is_sample_average(self):
        agent = make_ucb(1, arms=3)
        agent.update(np.array([2]), np.array([4.0]))
        agent.update(np.array([2]), np.array([2.0]))
        assert agent.Q[0, 2] == pytest.approx(3.0)
        assert agent.N[0, 2] == 2

    def test_spike_at_step_11(self):
        """Bài tập 2.8: phần thưởng TB nhảy lên ở bước 11 rồi tụt ở bước 12."""
        envs = 4000
        env, agent = make_env(envs=envs), make_ucb(envs, c=2.0)
        rewards, _ = run_experiment(env, agent, steps=14, seed=0)
        r10, r11, r12 = rewards[9], rewards[10], rewards[11]
        assert r11 > r10 + 0.5, (r10, r11)
        assert r11 > r12 + 0.08, (r11, r12)


# ===================================================================== 5. GRADIENT
class TestGradientBandit:
    def test_initial_state_is_uniform_softmax(self):
        agent = make_gradient(500)
        assert agent.H.shape == (500, 10) and np.all(agent.H == 0)
        actions = agent.act(make_rng(0))
        assert actions.shape == (500,) and actions.min() >= 0 and actions.max() < 10
        probs = get_probs(agent)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(probs, 0.1, atol=1e-9)

    def test_actions_are_sampled_from_the_softmax(self):
        agent = make_gradient(100000)
        agent.H[:, 0] = np.log(9.0)  # P(tay 0) = 9/18 = 0.5, mỗi tay còn lại 0.5/9
        actions = agent.act(make_rng(0))
        freq = np.bincount(actions, minlength=10) / 100000
        assert abs(freq[0] - 0.5) < 0.01, freq
        assert np.all(np.abs(freq[1:] - 0.5 / 9) < 0.005), freq

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_softmax_is_numerically_stable(self):
        agent = make_gradient(200)
        agent.H[:, 0] = 1000.0
        actions = agent.act(make_rng(0))
        probs = get_probs(agent)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(actions == 0)

    def test_update_matches_formula_and_uses_baseline_before_current_reward(self):
        alpha = 0.1
        agent = make_gradient(1, arms=3, alpha=alpha)
        rng = make_rng(0)

        agent.act(rng)
        pi = get_probs(agent)[0].copy()  # [1/3, 1/3, 1/3]
        agent.update(np.array([0]), np.array([2.0]))  # baseline trước bước này = 0
        onehot = np.array([1.0, 0.0, 0.0])
        expected = alpha * (2.0 - 0.0) * (onehot - pi)
        np.testing.assert_allclose(agent.H[0], expected, atol=1e-9)
        assert expected[0] > 0, "kiểm tra đề: R=2 > baseline=0 phải tăng H của tay đã chọn"

        agent.act(rng)
        pi = get_probs(agent)[0].copy()
        h_before = agent.H[0].copy()
        agent.update(np.array([1]), np.array([4.0]))  # baseline = 2.0 (phần thưởng bước 1)
        onehot = np.array([0.0, 1.0, 0.0])
        expected = h_before + alpha * (4.0 - 2.0) * (onehot - pi)
        np.testing.assert_allclose(agent.H[0], expected, atol=1e-9)

    def test_reward_above_baseline_raises_pulled_arm_and_below_lowers_it(self):
        for reward, sign in ((5.0, +1), (-5.0, -1)):
            agent = make_gradient(1, arms=4)
            agent.act(make_rng(0))
            before = get_probs(agent)[0, 2]
            agent.update(np.array([2]), np.array([reward]))
            agent.act(make_rng(1))
            after = get_probs(agent)[0, 2]
            assert np.sign(after - before) == sign

    def test_preferences_keep_zero_sum_and_probabilities_sum_to_one(self):
        """Tổng các thay đổi của H bằng 0 (tính chất của softmax gradient)."""
        envs = 20
        agent = make_gradient(envs, alpha=0.1)
        rng = make_rng(0)
        for _ in range(100):
            actions = agent.act(rng)
            agent.update(actions, rng.normal(1.0, 2.0, size=envs))
            np.testing.assert_allclose(agent.H.sum(axis=1), 0.0, atol=1e-8)
        agent.act(rng)
        np.testing.assert_allclose(get_probs(agent).sum(axis=1), 1.0, atol=1e-9)


# =========================================== 6. TÁI HIỆN XU HƯỚNG HÌNH TRONG SÁCH
# Mỗi cấu hình chạy đúng 1 lần (2000 bài toán x 1000 bước) rồi dùng chung.
_CURVES = {}
_SECONDS = {}

BUILDERS = {
    "eps0": lambda: (make_eps(2000, epsilon=0.0), 0.0),
    "eps0.01": lambda: (make_eps(2000, epsilon=0.01), 0.0),
    "eps0.1": lambda: (make_eps(2000, epsilon=0.1), 0.0),
    "optimistic": lambda: (make_optimistic(2000), 0.0),
    "ucb": lambda: (make_ucb(2000, c=2.0), 0.0),
    "grad_baseline": lambda: (make_gradient(2000, alpha=0.1), 4.0),
    "grad_no_baseline": lambda: (make_gradient(2000, alpha=0.1, use_baseline=False), 4.0),
}


def curve(name):
    if name not in _CURVES:
        agent, shift = BUILDERS[name]()
        start = time.perf_counter()
        _CURVES[name] = run_experiment(make_env(envs=2000), agent, steps=1000, seed=0, shift=shift)
        _SECONDS[name] = time.perf_counter() - start
    return _CURVES[name]


class TestBookResults:
    def test_greedy_gets_stuck(self):
        r, o = curve("eps0")
        assert 0.95 < tail(r) < 1.12, tail(r)
        assert 0.28 < tail(o) < 0.42, tail(o)

    def test_epsilon_01_matches_figure_2_2(self):
        r, o = curve("eps0.1")
        assert 1.28 < tail(r) < 1.45, tail(r)
        assert 0.74 < tail(o) < 0.86, tail(o)

    def test_more_exploration_helps_within_1000_steps(self):
        o0, o001, o01 = (tail(curve(k)[1]) for k in ("eps0", "eps0.01", "eps0.1"))
        assert o001 > o0 + 0.1 and o01 > o001 + 0.1, (o0, o001, o01)

    def test_epsilon_01_learns_faster_than_greedy_early(self):
        assert curve("eps0.1")[0][:100].mean() > curve("eps0")[0][:100].mean() + 0.02

    def test_optimistic_greedy_matches_figure_2_3(self):
        r, o = curve("optimistic")
        assert 0.78 < tail(o) < 0.93, tail(o)
        assert tail(o) > tail(curve("eps0.1")[1]) + 0.03
        assert r[:20].mean() < curve("eps0.1")[0][:20].mean(), "đầu ván lạc quan phải trả giá khám phá"

    def test_ucb_beats_epsilon_greedy_matches_figure_2_4(self):
        r, o = curve("ucb")
        assert tail(r) > tail(curve("eps0.1")[0]) + 0.05, (tail(r), tail(curve("eps0.1")[0]))
        assert 0.78 < tail(o) < 0.93, tail(o)

    def test_gradient_baseline_matches_figure_2_5(self):
        """q* dịch +4: có baseline học tốt, không baseline kém hẳn."""
        with_b = tail(curve("grad_baseline")[1])
        no_b = tail(curve("grad_no_baseline")[1])
        assert with_b > 0.70, with_b
        assert no_b < 0.65, no_b
        assert with_b > no_b + 0.15, (with_b, no_b)

    def test_main_experiment_runs_within_budget(self):
        """Lộ trình: toàn bộ thí nghiệm chính < 2 phút nhờ vector hóa."""
        curve("eps0.1")
        assert _SECONDS["eps0.1"] < 30, _SECONDS["eps0.1"]
