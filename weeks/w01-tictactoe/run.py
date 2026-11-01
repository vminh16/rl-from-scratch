import argparse, json, pathlib
import pickle
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
# bật cờ này: không cập nhật sau nước đi khám phá (đúng như sách S&B 1.5)
p.add_argument("--no_explore_update", action="store_true")
args = p.parse_args()

# mỗi biến thể ghi vào thư mục riêng để so sánh
folder = "results_no_explore" if args.no_explore_update else "results"
out = pathlib.Path(__file__).parent / folder
out.mkdir(exist_ok=True)
(out / "config.json").write_text(json.dumps(vars(args), indent=2))

for seed in args.seeds:
    print(f'Runing seed: {seed}')
    rng = make_rng(seed)
    env = TicTacToe()
    agent = AfterstateTD(alpha=args.alpha, epsilon=args.epsilon,
                         learn_from_explore=not args.no_explore_update)

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

# lưu bảng V của seed cuối cho play.py
with open(out / "agent_v_table.pkl", "wb") as f:
    pickle.dump(agent.v_table, f)
print(f"Saved agent at: {out / 'agent_v_table.pkl'}")
