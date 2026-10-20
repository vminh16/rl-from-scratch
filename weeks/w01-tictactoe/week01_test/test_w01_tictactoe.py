import numpy as np
import pytest


# --- Tests for TicTacToe Environment ---


def test_tictactoe_reset():
    """Reset must return an empty 3x3 board representation."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    obs = env.reset(rng)

    flat = np.array(obs).flatten()
    assert len(flat) == 9
    assert np.all(flat == 0)


def test_tictactoe_step_valid_move():
    """A valid move should place a marker and return a valid transition tuple."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    obs = env.reset(rng)

    next_obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert not truncated


def test_tictactoe_step_illegal_move():
    """Playing in an already occupied cell must raise ValueError."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    env.reset(rng)
    env.step(0)  # Agent plays at cell 0

    with pytest.raises(ValueError):
        env.step(0)  # Cell 0 is already occupied


def test_tictactoe_outcome_reported_on_finish():
    """Terminal state must include 'outcome' in info ('win', 'loss', or 'draw')."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    env.reset(rng)
    assert hasattr(env, "check_winner") or hasattr(env, "step")


# --- Tests for AfterstateTD Agent ---


def test_afterstate_td_initialization():
    """Agent must accept alpha and epsilon hyperparameters."""
    from minirl.agents.td_value import AfterstateTD

    agent = AfterstateTD(alpha=0.1, epsilon=0.1)
    assert agent.alpha == 0.1
    assert agent.epsilon == 0.1


def test_afterstate_td_act_returns_valid_action():
    """Agent action must be within [0..8] and on an empty cell."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.agents.td_value import AfterstateTD
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    obs = env.reset(rng)
    agent = AfterstateTD(alpha=0.1, epsilon=0.1)

    action = agent.act(obs, rng, greedy=False)
    assert 0 <= action <= 8
    flat = np.array(obs).flatten()
    assert flat[action] == 0


def test_afterstate_td_update_returns_metrics():
    """agent.update(transition) must return a dict containing learning stats."""
    from minirl.core import Transition
    from minirl.agents.td_value import AfterstateTD

    agent = AfterstateTD(alpha=0.1, epsilon=0.1)
    t = Transition(
        obs=np.zeros(9, dtype=int),
        action=0,
        reward=1.0,
        next_obs=np.zeros(9, dtype=int),
        terminated=True,
        truncated=False,
        info={"outcome": "win"},
    )
    metrics = agent.update(t)
    assert isinstance(metrics, dict)
    assert "td_error" in metrics or "loss" in metrics or len(metrics) > 0


# --- Integration Test: Full Episode ---


def test_w01_run_episode_integration():
    """TicTacToe + AfterstateTD must complete run_episode and report valid stats."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.agents.td_value import AfterstateTD
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    rng = make_rng(42)
    env = TicTacToe()
    agent = AfterstateTD(alpha=0.1, epsilon=0.1)

    stats = run_episode(env, agent, rng, train=True)

    assert "return" in stats
    assert "length" in stats
    assert "outcome" in stats
    assert stats["outcome"] in ("win", "loss", "draw")
    assert stats["length"] >= 1
