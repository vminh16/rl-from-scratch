import argparse, csv, json, pathlib
from minirl.core import run_episode
from minirl.envs.tictactoe import TicTacToe
from minirl.agents.td_value import AfterstateTD
from minirl.utils.seeding import make_rng
from minirl.utils.logger import CSVLogger

p = argparse.ArgumentParser()
p.add_argument('--seeds', type = int, nargs="+", default = [0, 1, 2, 3, 4])
p.add_argument("--episodes", type = int, default=20_000)
p.add_argument("--alpha", type=float, default=0.1)
p.add_argument("--epsilon", type=float, default=0.1)
args = p.parse_args()

out = pathlib.Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
(out / "config.json").write_text(json.dumps(vars(args), indent=2))
for seed in args.seeds:
    print(f'Runing seed: {seed}')
    rng = make_rng(seed)
    env, agent = TicTacToe(), AfterstateTD(alpha=args.alpha, epsilon=args.epsilon)
    
    csv_path = out / f"seed_{seed}.csv"
    with CSVLogger(csv_path, fieldnames=["episode", "return", "length", "outcome"]) as logger:
        for ep in range(args.episodes):
            if ep % 2000 == 0:
                print(f"  Episode {ep}/{args.episodes}...")

            r = run_episode(env, agent, rng)
            logger.log({
                "episode": ep,
                "return": r["return"],
                "length": r["length"],
                "outcome": r["outcome"],
            })