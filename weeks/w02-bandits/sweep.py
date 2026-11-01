import csv
import numpy as np
import pathlib
from minirl.envs.bandit import Vectorize10ArmsBandit
from minirl.agents.bandit_agents import EpsilonGreedy, UCBBandit, GradientBandit
from minirl.utils.seeding import make_rng

def evaluate_parameter(agent, runs=2000, steps=1000, seed = 42, nonstationary=False):
    rng = make_rng(seed)
    env = Vectorize10ArmsBandit(arms=10, envs=runs, non_stationary=nonstationary)
    agent.reset()
    env.reset(rng)
    total_reward = 0.0

    for _ in range(steps):
        action = agent.act(rng)
        rewards, _ = env.step(action, rng)
        total_reward += float(np.mean(rewards, axis = 0))
        agent.update(action, rewards)

    return total_reward / steps


def main():
    runs = 2000
    steps = 1000
    seed = 42
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok = True, parents = True)
    csv_file = out_dir / "sweep.csv"

    records = []
    print ("="*50)
    print (f'\nParameter sweep for bandits\n')
    print (f'Runs: {runs}, Steps: {steps}, Seed: {seed}\n')
    print (f"Result save in {csv_file}\n")
    print ("="*50)

    # 1. Sweep Epsilon Greedy parameters (epsilon)
    print("\n=====Epsilon Greedy=====\n")
    eps_list = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    for eps in eps_list:
        agent = EpsilonGreedy(arms=10, envs=runs, epsilon=eps)
        score = evaluate_parameter(agent, runs, steps)
        records.append({
            'algorithm': 'epsilon_greedy',
            'param_name': 'epsilon',
            'param_value': eps,
            'average_reward': score
        })
        print(f'epsilon: {eps:.5f} score: {score:.4f}')

    # 2. Sweep Optimistic greedy (Q0, alpha=0.1, epsilon=0.0)
    print("\n=====Optimistic Initial Values (alpha=0.1)=====\n")
    optimistic_values = [1/4, 1/2, 1.0, 2.0, 4.0]
    for q0 in optimistic_values:
        agent = EpsilonGreedy(arms=10, envs=runs, epsilon=0.0, initial_q=q0, alpha=0.1)
        score = evaluate_parameter(agent, runs, steps)
        records.append({
            'algorithm': 'optimistic',
            'param_name': 'initial_value',
            'param_value': q0,
            'average_reward': score
        })
        print(f'Optimistic Q0: {q0:.4f} score: {score:.4f}')

    # 3. UCB algorithm parameters (c)
    print("\n=====UCB Bandits=====\n")
    c_list = [1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
    for c in c_list:
        agent = UCBBandit(arms=10, envs=runs, c=c)
        score = evaluate_parameter(agent, runs, steps)
        records.append({
            'algorithm': 'ucb',
            'param_name': 'c',
            'param_value': c,
            'average_reward': score
        })
        print(f'C: {c:.4f} score: {score:.4f}')

    # 4. Gradient Bandit parameters (alpha, with baseline)
    print("\n=====Gradient Bandit (with baseline)=====\n")
    alpha_list = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
    for alpha in alpha_list:
        agent = GradientBandit(arms=10, envs=runs, alpha=alpha, use_baseline=True)
        score = evaluate_parameter(agent, runs, steps)
        records.append({
            'algorithm': 'gradient',
            'param_name': 'alpha',
            'param_value': alpha,
            'average_reward': score
        })
        print(f'Alpha: {alpha:.5f} score: {score:.4f}')    

    #write csv 
    with open(csv_file, 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = ['algorithm', 'param_name', 'param_value', 'average_reward'])
        writer.writeheader()
        writer.writerows(records)
    

if __name__ == "__main__":
    main()
