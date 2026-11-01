import csv
import pathlib
import matplotlib.pyplot as plt 
import numpy as np 


res_dir = pathlib.Path(__file__).parent / 'results'
fig_dir = pathlib.Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True, parents=True)

def load_aggregated_runs(exp_prefix: str, metric: str):
    """
    exp_prefix: prefix of experiment (eps_greedy, ucb, gradient, optimistic, nonstationary)
    metric: metric to plot ('rewards', 'optimal_action_pct')
    Returns:
        tuple of (aggregated_curve, sem_curve)
    """
    prefix_patterns = [
        f'{exp_prefix}/{exp_prefix}_*_{metric}.npy',
        f'{exp_prefix}_*_{metric}.npy'
    ]
    files = []
    for pat in prefix_patterns:
        files.extend(res_dir.glob(pat))
    files = sorted(set(files))

    if not files:
        return None, None

    runs = [np.load(f) for f in files]
    arr = np.array(runs)

    mean = np.mean(arr, axis=0)
    n_seeds = len(files)
    sem = np.std(arr, axis=0) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)

    return mean, sem 

def plot_with_sem(ax, x, mean, sem, label, color):
    (line,) = ax.plot(x, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(x, mean - sem, mean + sem, alpha=0.2, color=line.get_color(), linewidth=0)


# Figure 1: Compare performance of different algorithms 
def plot_overall_comparison():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    configs = [
        ("eps_greedy", r"$\epsilon$-Greedy ($\epsilon=0.1$)", "tab:red"),
        ("ucb", "UCB (c=2)", "tab:blue"),
        ("gradient", r"Gradient Bandit ($\alpha=0.1$)", "tab:green"),
        ("optimistic", r"Optimistic Initialization ($Q_0=5$)", "tab:purple"),
        ("nonstationary", r"Nonstationary ($\epsilon=0.1, \alpha=0.1$)", "tab:orange")
    ]
    
    has_data = False

    for exp_name, label, color in configs:
        r_mean, r_sem = load_aggregated_runs(exp_name, "rewards")
        opt_mean, opt_sem = load_aggregated_runs(exp_name, "optimal_action_pct")

        if r_mean is not None:
            has_data = True
            step = np.arange(1, len(r_mean) + 1)
            plot_with_sem(ax1, step, r_mean, r_sem, label, color)
            plot_with_sem(ax2, step, opt_mean, opt_sem, label, color)

    if not has_data:
        print("No data found for overall comparison.")
        plt.close(fig)
        return 

    # Format the two subplots 
    ax1.set_ylabel("Average reward")
    ax1.set_title("10-Armed Bandit | Algorithms Comparison | 5 Seeds")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Optimal action %\n$\\mathbb{P}[A_t = a^*]$")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = fig_dir / "bandit_algorithms_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


# Figure 2: Optimistic initialization vs epsilon greedy  
def plot_optimistic():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    opt_r, opt_r_sem = load_aggregated_runs("optimistic", "rewards")
    opt_opt, opt_opt_sem = load_aggregated_runs("optimistic", "optimal_action_pct")

    eps_r, eps_r_sem = load_aggregated_runs("eps_greedy", "rewards")
    eps_opt, eps_opt_sem = load_aggregated_runs("eps_greedy", "optimal_action_pct")

    if opt_r is None or eps_r is None:
        print("No data found for optimistic comparison.")
        plt.close(fig)
        return 
    
    steps = np.arange(1, len(opt_r) + 1)

    plot_with_sem(ax1, steps, opt_r, opt_r_sem, r"Optimistic ($Q_0=5, \epsilon=0$)", "tab:red")
    plot_with_sem(ax1, steps, eps_r, eps_r_sem, r"$\epsilon$-Greedy ($\epsilon=0.1$)", "tab:blue")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Optimistic Initial Values vs Epsilon-Greedy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    plot_with_sem(ax2, steps, opt_opt, opt_opt_sem, r"Optimistic ($Q_0=5, \epsilon=0$)", "tab:red")
    plot_with_sem(ax2, steps, eps_opt, eps_opt_sem, r"$\epsilon$-Greedy ($\epsilon=0.1$)", "tab:blue")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Optimal action %\n$\\mathbb{P}[A_t = a^*]$")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = fig_dir / "bandit_optimistic_initial_values.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)

    
# Figure 3: UCB vs Epsilon-greedy
def plot_ucb():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ucb_r, ucb_r_sem = load_aggregated_runs('ucb', 'rewards')
    ucb_act, ucb_act_sem = load_aggregated_runs('ucb', 'optimal_action_pct')

    eps_r, eps_r_sem = load_aggregated_runs('eps_greedy', 'rewards')
    eps_act, eps_act_sem = load_aggregated_runs('eps_greedy', 'optimal_action_pct')

    if ucb_r is None or eps_r is None: 
        print("No data found for UCB.")
        plt.close(fig)
        return 
    
    steps = np.arange(1, len(ucb_r) + 1)

    plot_with_sem(ax1, steps, ucb_r, ucb_r_sem, "UCB (c=2)", "tab:blue")
    plot_with_sem(ax1, steps, eps_r, eps_r_sem, r"$\epsilon$-Greedy ($\epsilon=0.1$)", "tab:red")
    ax1.set_ylabel("Average reward")
    ax1.set_title("UCB vs Epsilon-Greedy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    plot_with_sem(ax2, steps, ucb_act, ucb_act_sem, "UCB (c=2)", "tab:blue")
    plot_with_sem(ax2, steps, eps_act, eps_act_sem, r"$\epsilon$-Greedy ($\epsilon=0.1$)", "tab:red")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Optimal action %\n$\\mathbb{P}[A_t = a^*]$")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = fig_dir / "bandit_ucb_vs_epsilon_greedy.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


# Figure 4: Parameter sweep (Figure 2.6)
def plot_parameter_sweep():
    csv_file = res_dir / 'sweep.csv'
    if not csv_file.exists():
        print("No sweep.csv found.")
        return 
    data = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row['algorithm']
            val = float(row['param_value'])
            score = float(row['average_reward'])
            data.setdefault(alg, []).append((val, score))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 4 thuật toán kinh điển chuẩn Sutton & Barto Hình 2.6
    styles = {
        "epsilon_greedy": (r'$\epsilon$-greedy ($\epsilon$)', '#d62728', 'o'),
        "optimistic": (r'Optimistic greedy ($Q_0$, $\alpha=0.1$)', '#ff7f0e', 'D'),
        "ucb": (r'UCB ($c$)', '#1f77b4', '^'),
        "gradient": (r'Gradient Bandit ($\alpha$)', '#2ca02c', 's'),
    }
    
    for alg, (label, color, marker) in styles.items():
        if alg in data:
            points = sorted([p for p in data[alg] if p[0] > 0], key=lambda x: x[0])
            if points:
                xs = np.array([p[0] for p in points])
                ys = np.array([p[1] for p in points])
                ax.plot(xs, ys, label=label, color=color, marker=marker, linestyle='-', markersize=7, linewidth=2.0)
                
    ax.set_xscale("log", base=2)
    ticks = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
    tick_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=11)
    
    ax.set_ylim(0.7, 1.55)
    ax.set_xlabel(r"Parameter value: $\epsilon, \alpha, c,$ or $Q_0$ (log scale)", fontsize=12)
    ax.set_ylabel("Average reward over first 1000 steps", fontsize=12)
    ax.set_title("Parameter Sweep Comparison (Sutton & Barto Figure 2.6)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = fig_dir / "bandit_parameter_sweep.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)

    
# Figure 5: Nonstationary Bandits 
def plot_nonstationary():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ns_r, ns_r_sem = load_aggregated_runs("nonstationary", "rewards")
    ns_opt, ns_opt_sem = load_aggregated_runs("nonstationary", "optimal_action_pct")
    
    if ns_r is None:
        print("No data found for nonstationary.")
        plt.close(fig)
        return 
    
    steps = np.arange(1, len(ns_r) + 1)

    plot_with_sem(ax1, steps, ns_r, ns_r_sem, r"Nonstationary ($\epsilon=0.1, \alpha=0.1$)", "tab:orange")
    plot_with_sem(ax2, steps, ns_opt, ns_opt_sem, r"Nonstationary ($\epsilon=0.1, \alpha=0.1$)", "tab:orange")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Tracking a Nonstationary Problem")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Optimal action %\n$\\mathbb{P}[A_t = a^*]$")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = fig_dir / "bandit_nonstationary.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


# Figure 6: Gradient Bandit Performance
def plot_gradient_bandit():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    grad_r, grad_r_sem = load_aggregated_runs("gradient", "rewards")
    grad_opt, grad_opt_sem = load_aggregated_runs("gradient", "optimal_action_pct")
    
    grad_wo_baseline_r, grad_wo_baseline_r_sem = load_aggregated_runs("gradient_without_baseline", "rewards")
    grad_wo_baseline_opt, grad_wo_baseline_opt_sem = load_aggregated_runs("gradient_without_baseline", "optimal_action_pct")
    
    if grad_r is None and grad_wo_baseline_r is None:
        print("No data found for gradient bandit.")
        plt.close(fig)
        return 
    
    if grad_r is not None:
        steps = np.arange(1, len(grad_r) + 1)
        plot_with_sem(ax1, steps, grad_r, grad_r_sem, "Gradient Bandit (with baseline)", "tab:green")
        plot_with_sem(ax2, steps, grad_opt, grad_opt_sem, "Gradient Bandit (with baseline)", "tab:green")

    if grad_wo_baseline_r is not None:
        steps = np.arange(1, len(grad_wo_baseline_r) + 1)
        plot_with_sem(ax1, steps, grad_wo_baseline_r, grad_wo_baseline_r_sem, "Gradient Bandit (no baseline)", "tab:red")
        plot_with_sem(ax2, steps, grad_wo_baseline_opt, grad_wo_baseline_opt_sem, "Gradient Bandit (no baseline)", "tab:red")

    ax1.set_ylabel("Average reward")
    ax1.set_title("Gradient Bandit Performance")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Optimal action %\n$\\mathbb{P}[A_t = a^*]$")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = fig_dir / "bandit_gradient_bandit.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_overall_comparison()
    plot_optimistic()
    plot_ucb()
    plot_parameter_sweep()
    plot_nonstationary()
    plot_gradient_bandit()
    print("Done!")