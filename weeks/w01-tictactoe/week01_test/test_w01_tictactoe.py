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

    # chơi đến khi hết ván, ván cuối phải có outcome hợp lệ
    done = False
    info = {}
    while not done:
        empty = np.where(env.board == 0)[0]
        _, _, done, _, info = env.step(int(empty[0]))
    assert info["outcome"] in ("win", "loss", "draw")


def test_tictactoe_detects_all_8_winning_lines():
    """3 hàng + 3 cột + 2 đường chéo, mỗi đường đều phải được nhận ra."""
    from minirl.envs.tictactoe import TicTacToe, WIN_STATES

    env = TicTacToe()
    assert len(WIN_STATES) == 8
    for line in WIN_STATES:
        for player in (1, -1):
            board = np.zeros(9, dtype=int)
            board[list(line)] = player
            assert env.check_win(board) == player

    # bàn trống và bàn chưa đủ 3 quân thì không có người thắng
    assert env.check_win(np.zeros(9, dtype=int)) == 0
    assert env.check_win(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])) == 0


def test_tictactoe_reachable_states_count():
    """DFS từ bàn trống (X đi trước, dừng khi có kết quả) đếm đúng 5.478 trạng thái."""
    from minirl.envs.tictactoe import TicTacToe

    env = TicTacToe()
    seen = set()

    def dfs(board, player):
        if board in seen:
            return
        seen.add(board)
        if env.check_win(np.array(board)) != 0 or 0 not in board:
            return
        for i in range(9):
            if board[i] == 0:
                dfs(board[:i] + (player,) + board[i + 1:], -player)

    dfs((0,) * 9, 1)
    assert len(seen) == 5478


def test_same_seed_gives_same_results():
    """Cùng seed chạy hai lần phải ra đúng cùng chuỗi kết quả."""
    from minirl.envs.tictactoe import TicTacToe
    from minirl.agents.td_value import AfterstateTD
    from minirl.core import run_episode
    from minirl.utils.seeding import make_rng

    def run(seed):
        rng = make_rng(seed)
        env, agent = TicTacToe(), AfterstateTD(alpha=0.1, epsilon=0.1)
        return [run_episode(env, agent, rng)["outcome"] for _ in range(200)]

    assert run(7) == run(7)
    assert run(7) != run(8)  # seed khác thì khác


def test_skip_explore_update_leaves_table_unchanged():
    """learn_from_explore=False: nước khám phá không được làm thay đổi bảng V."""
    from minirl.core import Transition
    from minirl.agents.td_value import AfterstateTD

    agent = AfterstateTD(alpha=0.1, epsilon=1.0, learn_from_explore=False)

    class AlwaysLow:  # rng giả: luôn "khám phá" và chọn phần tử đầu
        def random(self): return 0.0
        def choice(self, x): return x[0]

    obs = np.zeros(9, dtype=int)
    action = agent.act(obs, AlwaysLow())
    t = Transition(obs, action, 0.0, obs, False, False, {})
    agent.update(t)
    assert agent.v_table == {}


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
