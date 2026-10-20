"""
Đọc kết quả đa seed từ thư mục results và xuất biểu đồ học tập
"""

from pathlib import Path
import matplotlib.pyplot as plt
from minirl.utils.plotting import plot_mean_with_sem

results_dir = Path(__file__).parent / "results"
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

csv_files = sorted(list(results_dir.glob("seed_*.csv")))
if not csv_files:
    print("Chưa tìm thấy file seed_*.csv nào trong results/. Hãy chạy run.py trước!")
    exit(1)

fig, ax = plt.subplots(figsize=(9, 5))

# Vẽ đường trung bình return kèm khoảng tin cậy SEM
plot_mean_with_sem(
    ax=ax,
    csv_paths=csv_files,
    column="return",
    label="AfterstateTD (vs Random)",
    window_size=500,  # Moving average 500 ván
    color="tab:blue",
)

ax.set_title("Đường cong học tập Tic-Tac-Toe (Afterstate TD)", fontsize=13)
ax.set_xlabel("Số ván đấu (Episode)", fontsize=11)
ax.set_ylabel("Phần thưởng trung bình (Return)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")

output_path = figures_dir / "learning_curve.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Biểu đồ đã được lưu tại: {output_path}")
