# MiniRL: Reinforcement Learning from Scratch

A clean, modular Reinforcement Learning library implemented from first principles for Embodied AI (2026–2028 roadmap). Built following Sutton & Barto (2018) specifications and developed with Test-Driven Development (TDD).

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Weekly Implementations](#weekly-implementations)
  - [Week 01: Tic-Tac-Toe (Afterstate TD)](#week-01-tic-tac-toe-afterstate-td)
  - [Week 02: Multi-Armed Bandits](#week-02-multi-armed-bandits)
- [Testing](#testing)

---

## Overview

MiniRL provides clean, reproducible implementations of foundational reinforcement learning algorithms. Every component is verified with unit tests and evaluated across multiple random seeds using standard statistical metrics (Mean ± SEM).

---

## Architecture

- `minirl.core`: Protocol definitions (`Env`, `Agent`), `Transition` schema, and the standardized `run_episode` loop.
- `minirl.utils`: Seeded RNG generation (`seeding.py`), episodic CSV logging (`logger.py`), and moving average curve plotting (`plotting.py`).
- `minirl.envs`: Custom and benchmark environments.
- `minirl.agents`: Value-based, policy-based, and model-based RL agents.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mixp123/rl-from-scratch.git
cd rl-from-scratch

# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## Weekly Implementations

### Week 01: Tic-Tac-Toe (Afterstate TD)

- **Problem**: 3x3 board game against a stochastic opponent.
- **Method & Algorithm**: Afterstate Temporal Difference Learning ($TD(0)$). Rather than evaluating state-action pairs $(s, a)$, the value function directly evaluates the afterstate $S'$ immediately following the agent's move:
  $$V(S'_{t-1}) \leftarrow V(S'_{t-1}) + \alpha \big[ V(S'_t) - V(S'_{t-1}) \big]$$
  Exploration is controlled via $\epsilon$-greedy action selection over candidate afterstates.
- **Results**: Evaluated across 5 random seeds with 20,000 episodes per seed. The agent reaches convergence within 3,000–5,000 episodes, effectively eliminating losses against random play.
- **Artifacts & Evidence**:
  - Environment: [`minirl/envs/tictactoe.py`](minirl/envs/tictactoe.py)
  - Agent: [`minirl/agents/td_value.py`](minirl/agents/td_value.py)
  - Test Suite: [`weeks/w01-tictactoe/week01_test/test_w01_tictactoe.py`](weeks/w01-tictactoe/week01_test/test_w01_tictactoe.py)
  - Benchmark Data: [`weeks/w01-tictactoe/results/`](weeks/w01-tictactoe/results/)
  - Interactive Simulator: [`weeks/w01-tictactoe/play.py`](weeks/w01-tictactoe/play.py)

![Week 01 Learning Curve](weeks/w01-tictactoe/figures/learning_curve.png)

---

### Week 02: Multi-Armed Bandits

- **Problem**: Stationary and non-stationary $k$-armed bandit testbeds.
- **Method & Algorithm**: $\epsilon$-greedy action-value estimation, Upper Confidence Bound (UCB), and Gradient Bandit algorithms.
- **Status**: In progress ([`weeks/w02-bandits/`](weeks/w02-bandits/)).

---

## Testing

Run the full pytest suite to verify all core components and algorithms:

```bash
pytest
```
