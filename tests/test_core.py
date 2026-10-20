import numpy as np
import pytest


class DummyEnv:
    """Mock environment with predictable transitions for testing run_episode."""

    def __init__(self, max_steps: int = 3, truncate_at: int | None = None):
        self.max_steps = max_steps
        self.truncate_at = truncate_at
        self.current_step = 0

    def reset(self, rng: np.random.Generator):
        self.current_step = 0
        return "state_0"

    def step(self, action):
        self.current_step += 1
        next_obs = f"state_{self.current_step}"
        reward = 1.0
        terminated = self.current_step >= self.max_steps
        truncated = (
            self.truncate_at is not None and self.current_step >= self.truncate_at
        )
        info = {"step": self.current_step}
        if terminated:
            info["outcome"] = "win"
        return next_obs, reward, terminated, truncated, info


class DummyAgent:
    """Mock agent recording actions and updates."""

    def __init__(self):
        self.act_calls = []
        self.updates = []

    def act(self, obs, rng: np.random.Generator, greedy: bool = False):
        self.act_calls.append({"obs": obs, "greedy": greedy})
        return 0

    def update(self, t) -> dict:
        self.updates.append(t)
        return {"td_error": 0.5}


# --- Tests for Seeding ---


def test_make_rng_reproducibility():
    """Same seed must produce identical pseudo-random sequences."""
    from minirl.utils.seeding import make_rng

    rng1 = make_rng(42)
    rng2 = make_rng(42)
    val1 = rng1.integers(0, 100, size=5)
    val2 = rng2.integers(0, 100, size=5)
    np.testing.assert_array_equal(val1, val2)


def test_make_rng_different_seeds():
    """Different seeds must produce different sequences."""
    from minirl.utils.seeding import make_rng

    rng1 = make_rng(42)
    rng2 = make_rng(43)
    val1 = rng1.integers(0, 100, size=10)
    val2 = rng2.integers(0, 100, size=10)
    assert not np.array_equal(val1, val2)


# --- Tests for Transition ---


def test_transition_dataclass():
    """Transition must store transition components and have default dict info."""
    from minirl.core import Transition

    t = Transition(
        obs="s0",
        action=1,
        reward=2.5,
        next_obs="s1",
        terminated=True,
        truncated=False,
    )
    assert t.obs == "s0"
    assert t.action == 1
    assert t.reward == 2.5
    assert t.next_obs == "s1"
    assert t.terminated is True
    assert t.truncated is False
    assert t.info == {}


def test_transition_info_isolation():
    """Default info dict must be a fresh instance per Transition (default_factory)."""
    from minirl.core import Transition

    t1 = Transition("s0", 0, 0.0, "s1", False, False)
    t2 = Transition("s0", 0, 0.0, "s1", False, False)
    t1.info["key"] = "val"
    assert "key" not in t2.info


# --- Tests for run_episode ---


def test_run_episode_train_mode():
    """In train mode, agent.update must be called and its metrics accumulated."""
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    rng = make_rng(0)
    env = DummyEnv(max_steps=3)
    agent = DummyAgent()

    stats = run_episode(env, agent, rng, train=True)

    assert stats["length"] == 3
    assert stats["return"] == 3.0
    assert stats["outcome"] == "win"
    assert len(agent.updates) == 3
    assert stats.get("td_error") == pytest.approx(1.5)
    assert all(not call["greedy"] for call in agent.act_calls)


def test_run_episode_eval_mode():
    """In eval mode (train=False), agent.update must not be called, greedy=True."""
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    rng = make_rng(0)
    env = DummyEnv(max_steps=3)
    agent = DummyAgent()

    stats = run_episode(env, agent, rng, train=False)

    assert stats["length"] == 3
    assert stats["return"] == 3.0
    assert len(agent.updates) == 0
    assert "td_error" not in stats
    assert all(call["greedy"] for call in agent.act_calls)


def test_run_episode_truncated():
    """Truncated episodes must terminate early and merge info."""
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    rng = make_rng(0)
    env = DummyEnv(max_steps=10, truncate_at=2)
    agent = DummyAgent()

    stats = run_episode(env, agent, rng, train=True)

    assert stats["length"] == 2
    assert stats["return"] == 2.0
    assert len(agent.updates) == 2
    assert agent.updates[-1].truncated is True


def test_run_episode_max_steps():
    """run_episode must respect max_steps parameter even if env doesn't terminate."""
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    rng = make_rng(0)
    env = DummyEnv(max_steps=100)
    agent = DummyAgent()

    stats = run_episode(env, agent, rng, train=True, max_steps=5)

    assert stats["length"] == 5
    assert stats["return"] == 5.0
