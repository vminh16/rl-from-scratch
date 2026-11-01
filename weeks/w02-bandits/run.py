import json
import argparse 
import pathlib
import numpy as np
from minirl.agents.bandit_agents import EpsilonGreedy, UCBBandit, GradientBandit
from minirl.envs.bandit import Vectorize10ArmsBandit 
from minirl.utils.seeding import make_rng

def run_experiment(agent, env, steps: int, rng: np.random.Generator, shift: float = 0.0):
    """ run experiment 
    Args:
        agent: agent to train
        env: env to train on
        steps: number of steps to train
        rng: random number generator
        shift: reward shift added to q_star
    Returns:
        tuple of (rewards, is_optimal)
    """

    average_reward = np.zeros(steps, dtype = np.float64)
    optimal_action_pct = np.zeros(steps, dtype = np.float64)

    agent.reset()
    env.reset(rng)
    if shift != 0.0:
        env.q_star = env.q_star + shift

    for t in range(steps):
        # get action from agent
        action = agent.act(rng)
        # step env
        reward, is_optimal = env.step(action, rng)
        agent.update(action, reward)
        #record reward and optimal action
        average_reward[t] = np.mean(reward)
        optimal_action_pct[t] = np.mean(is_optimal) * 100
    return average_reward, optimal_action_pct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument(
        '--exp', 
        type=str, 
        default='eps_greedy',
        choices = ['eps_greedy', 'ucb', 'gradient', 'gradient_without_baseline', 'gradient_shift4', 'gradient_without_baseline_shift4', 'optimistic', 'nonstationary']
    )
    parser.add_argument('--seeds', default = [1, 2, 3, 4, 5], type = int, nargs = '*')
    parser.add_argument('--c', default = 2.0, type = float)
    parser.add_argument('--epsilon', default = 0.1, type = float)
    parser.add_argument('--alpha', default = 0.1, type = float)
    parser.add_argument('--initial_q', default = 0.0, type = float)
    parser.add_argument('--alpha_nonstationary', default=0.1, type=float)
    parser.add_argument('--shift', default=0.0, type=float)
    
    args = parser.parse_args()

    out_dir = pathlib.Path(__file__).parent / "results" / args.exp
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Start run experiment: {args.exp} with {args.runs} runs and {args.steps} steps")
    
    results = {}

    for seed in args.seeds:
        print(f"Run experiment with seed: {seed}")
        rng = make_rng(seed)
        if args.exp == "eps_greedy":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs) #stationary
            agent = EpsilonGreedy(arms=10, envs=args.runs, epsilon=args.epsilon, initial_q = 0.0)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'eps_greedy_{seed}_rewards'] = reward
            results[f'eps_greedy_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "ucb":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            agent = UCBBandit(arms=10, envs=args.runs, c = args.c)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'ucb_{seed}_rewards'] = reward
            results[f'ucb_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "gradient":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            agent = GradientBandit(arms=10, envs=args.runs, alpha = args.alpha, use_baseline=True)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'gradient_{seed}_rewards'] = reward
            results[f'gradient_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "gradient_without_baseline":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            agent = GradientBandit(arms=10, envs=args.runs, alpha = args.alpha, use_baseline=False)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'gradient_without_baseline_{seed}_rewards'] = reward
            results[f'gradient_without_baseline_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "gradient_shift4":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            agent = GradientBandit(arms=10, envs=args.runs, alpha = args.alpha, use_baseline=True)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=4.0)
            results[f'gradient_shift4_{seed}_rewards'] = reward
            results[f'gradient_shift4_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "gradient_without_baseline_shift4":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            agent = GradientBandit(arms=10, envs=args.runs, alpha = args.alpha, use_baseline=False)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=4.0)
            results[f'gradient_without_baseline_shift4_{seed}_rewards'] = reward
            results[f'gradient_without_baseline_shift4_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "optimistic":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs)
            # Optimistic Greedy chuẩn Sutton & Barto (Fig 2.3): initial_q=5.0, epsilon=0.0, alpha=0.1
            init_q = args.initial_q if args.initial_q != 0.0 else 5.0
            eps = args.epsilon if args.epsilon != 0.1 else 0.0
            agent = EpsilonGreedy(arms=10, envs=args.runs, epsilon=eps, initial_q=init_q, alpha=args.alpha)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'optimistic_{seed}_rewards'] = reward
            results[f'optimistic_{seed}_optimal_action_pct'] = opt_action

        elif args.exp == "nonstationary":
            env = Vectorize10ArmsBandit(arms=10, envs=args.runs, non_stationary=True, walk_std = 0.3)
            agent = EpsilonGreedy(arms=10, envs=args.runs, epsilon= 0.1, initial_q=0.0, alpha=args.alpha_nonstationary)
            reward, opt_action = run_experiment(agent, env, args.steps, rng, shift=args.shift)
            results[f'nonstationary_{seed}_rewards'] = reward
            results[f'nonstationary_{seed}_optimal_action_pct'] = opt_action
    # save results
    for key, value in results.items():
        np.save(out_dir / f'{key}.npy', value)
    print("Done!")
    # save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    
if __name__ == "__main__":
    main()
    
    
        