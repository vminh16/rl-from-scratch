"""
đọc kết quả nhiều seed, làm mượt và vẽ đồ thị
"""

import csv
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt 
import numpy as np

__all__ = ["moving_average", "load_seed_runs", "plot_mean_with_sem"]



def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Tính trung bình động cho 1 chiều.
    Nếu data ngắn hơn window_size thì giữ nguyên data gốc.

    Args:
        data: 1D array
        window_size: độ rộng cửa sổ

    Returns:
        1D array chứa đường trung bình
    """
    if window_size <= 1 or len(data) < window_size:
        return data

    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def load_seed_runs(csv_path: Sequence[str | Path], column: str) -> np.ndarray:
    """
    Đọc 1 cột từ nhiều CSV, mỗi CSV ứng với 1 seed.
    Giả định tất cả CSV có cùng số dòng.

    Args:
        csv_path: danh sách đường dẫn tới file CSV
        column: tên cột cần đọc (vd: "episode", "return", "length")

    Returns:
        2D array (n_seeds, max_len)
    """
    runs = []
    min_len = float("inf")

    for path in csv_path:
        value = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value.append(float(row[column]))
        runs.append(value)
        min_len = min(min_len, len(value))

    # cắt ngắn về cùng độ dài
    runs = [r[:min_len] for r in runs]
    return np.array(runs, dtype=float)

def plot_mean_with_sem(
        ax: plt.Axes,
        csv_paths: Sequence[str | Path],
        column: str,
        label: str,
        window_size: int = 100,
        color: str | None = None,
        alpha: float = 0.3
        ) -> None:
    """
    Vẽ đường trung bình và confidence interval.

    Args:
        ax: Axes matplotlib để vẽ
        csv_paths: danh sách file CSV
        column: cột cần lấy (vd: "return")
        label: label cho legend
        window_size: kernel cho moving average
        color: màu đường vẽ và fill
        alpha: opacity cho fill
    
    """
    runs = load_seed_runs(csv_paths, column=column) #shape (n_seeds, max_steps)
    n_seeds, _ = runs.shape

    # làm mượt từng seed trước khi tính thống kê
    smoothed_runs = np.array([moving_average(runs[i], window_size) for i in range(n_seeds)]) #shape (n_seeds, smooth_steps)


    #tính mean và sai số chuẩn
    mean = np.mean(smoothed_runs, axis=0)
    # std với ddof=1 (hiệu chỉnh mẫu Bessel), sau đó chia cho sqrt(N)
    std = np.std(smoothed_runs, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
    sem = std / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)
    
    
    
    # Vẽ trục hoành 
    x = np.arange(len(mean)) + (window_size if window_size > 1 else 1)

    # vẽ đường mean và vùng mở Fill Between 
    (line,) = ax.plot(x, mean, label=f'{label}(N={n_seeds})', color=color, linewidth=1.2)
    ax.fill_between(
        x,
        mean - sem,
        mean + sem,
        alpha=alpha,
        color=line.get_color(),
        linewidth=0,
        label="_nolegend_",
    )
