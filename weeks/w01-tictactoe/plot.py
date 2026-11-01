"""
Đọc kết quả đa seed từ thư mục results và xuất biểu đồ học tập
- learning_curve.png: return trung bình
- outcomes.png: tỉ lệ thắng / hòa / thua theo số ván
- compare_explore.png: so sánh có / không cập nhật sau nước khám phá (nếu có results_no_explore/)
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from minirl.utils.plotting import plot_mean_with_sem, moving_average

here = Path(__file__).parent
results_dir = here / "results"
figures_dir = here / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

WINDOW = 500  # moving average 500 ván
LAST = 2000   # số ván cuối dùng để tính bảng kết quả


def load_outcomes(csv_files):
    """Trả về mảng (n_seed, n_ván) với giá trị 'win' / 'draw' / 'loss'."""
    runs = []
    for path in csv_files:
        with open(path, encoding="utf-8") as f:
            runs.append([row["outcome"] for row in csv.DictReader(f)])
    n = min(len(r) for r in runs)  # cắt về cùng độ dài
    return np.array([r[:n] for r in runs])


def rate_curve(outcomes, name):
    """Tỉ lệ `name` theo ván cho từng seed, đã làm mượt. Shape (n_seed, n_điểm)."""
    return np.array([moving_average((o == name).astype(float), WINDOW) for o in outcomes])


def draw_with_sem(ax, curves, label, color):
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    x = np.arange(len(mean)) + WINDOW
    ax.plot(x, mean, label=label, color=color, linewidth=1.2)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)


def mean_sem(values):
    values = np.array(values)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


csv_files = sorted(results_dir.glob("seed_*.csv"))
if not csv_files:
    print("Chưa tìm thấy file seed_*.csv nào trong results/. Hãy chạy run.py trước!")
    exit(1)

# ---------- 1. đường cong return ----------
fig, ax = plt.subplots(figsize=(9, 5))
plot_mean_with_sem(ax=ax, csv_paths=csv_files, column="return",
                   label="AfterstateTD (vs Random)", window_size=WINDOW, color="tab:blue")
ax.set_title("Đường cong học tập Tic-Tac-Toe (Afterstate TD)", fontsize=13)
ax.set_xlabel("Số ván đấu (Episode)")
ax.set_ylabel("Phần thưởng trung bình (Return)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(figures_dir / "learning_curve.png", dpi=300)
plt.close(fig)

# ---------- 2. tỉ lệ thắng / hòa / thua ----------
outcomes = load_outcomes(csv_files)
n_seeds, n_eps = outcomes.shape

fig, ax = plt.subplots(figsize=(9, 5))
for name, label, color in [("win", "Thắng", "tab:green"),
                           ("draw", "Hòa", "tab:orange"),
                           ("loss", "Thua", "tab:red")]:
    draw_with_sem(ax, rate_curve(outcomes, name), f"{label} (N={n_seeds})", color)
ax.set_title("Tỉ lệ thắng / hòa / thua của Afterstate TD vs đối thủ ngẫu nhiên", fontsize=13)
ax.set_xlabel("Số ván đấu (Episode)")
ax.set_ylabel(f"Tỉ lệ (trung bình trượt {WINDOW} ván)")
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig(figures_dir / "outcomes.png", dpi=300)
plt.close(fig)

# ---------- 3. bảng số cho README ----------
print(f"\n{n_seeds} seed x {n_eps} ván. Trung bình ± SE trên {LAST} ván cuối:")
for name in ["win", "draw", "loss"]:
    m, s = mean_sem([(o[-LAST:] == name).mean() for o in outcomes])
    print(f"  {name:5s}: {m:.3f} ± {s:.3f}")
m, s = mean_sem([(o[-LAST:] != "loss").mean() for o in outcomes])
print(f"  không thua: {m:.3f} ± {s:.3f}")

# ---------- 4. so sánh có / không cập nhật sau nước khám phá ----------
other_files = sorted((here / "results_no_explore").glob("seed_*.csv"))
if other_files:
    other = load_outcomes(other_files)
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_with_sem(ax, rate_curve(outcomes, "loss"), "Cập nhật cả nước khám phá", "tab:blue")
    draw_with_sem(ax, rate_curve(other, "loss"), "Bỏ qua nước khám phá (như sách)", "tab:purple")
    ax.set_title("Tỉ lệ thua: cập nhật vs bỏ qua nước đi khám phá", fontsize=13)
    ax.set_xlabel("Số ván đấu (Episode)")
    ax.set_ylabel(f"Tỉ lệ thua (trung bình trượt {WINDOW} ván)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "compare_explore.png", dpi=300)
    plt.close(fig)

    print("\nBỏ qua nước khám phá, trên", LAST, "ván cuối:")
    for name in ["win", "draw", "loss"]:
        m, s = mean_sem([(o[-LAST:] == name).mean() for o in other])
        print(f"  {name:5s}: {m:.3f} ± {s:.3f}")
    m, s = mean_sem([(o[-LAST:] != "loss").mean() for o in other])
    print(f"  không thua: {m:.3f} ± {s:.3f}")

print("\nĐã lưu hình vào", figures_dir)
