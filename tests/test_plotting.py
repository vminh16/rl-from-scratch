import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from minirl.utils.plotting import load_seed_runs, moving_average, plot_mean_with_sem


# --- Tests for moving_average ---


def test_moving_average_basic():
    """Moving average must calculate 1D convolution correctly."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = moving_average(data, window_size=3)

    # Valid window averages: (1+2+3)/3 = 2.0, (2+3+4)/3 = 3.0, (3+4+5)/3 = 4.0
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(smoothed, expected)


def test_moving_average_window_size_one_or_smaller():
    """window_size <= 1 must return the original data unchanged."""
    data = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(moving_average(data, window_size=1), data)
    np.testing.assert_array_equal(moving_average(data, window_size=0), data)


def test_moving_average_data_shorter_than_window():
    """If data length is shorter than window_size, return original data."""
    data = np.array([1.0, 2.0])
    np.testing.assert_array_equal(moving_average(data, window_size=5), data)


# --- Tests for load_seed_runs ---


def test_load_seed_runs_aligns_lengths(tmp_path: Path):
    """load_seed_runs must align multiple seed runs to the shortest run length."""
    # Seed 0: 3 rows
    csv0 = tmp_path / "seed_0.csv"
    with open(csv0, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return"])
        writer.writeheader()
        writer.writerows([{"episode": 0, "return": 1.0},
                          {"episode": 1, "return": 2.0},
                          {"episode": 2, "return": 3.0}])

    # Seed 1: 4 rows
    csv1 = tmp_path / "seed_1.csv"
    with open(csv1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return"])
        writer.writeheader()
        writer.writerows([{"episode": 0, "return": 5.0},
                          {"episode": 1, "return": 6.0},
                          {"episode": 2, "return": 7.0},
                          {"episode": 3, "return": 8.0}])

    runs = load_seed_runs([csv0, csv1], column="return")

    assert isinstance(runs, np.ndarray)
    assert runs.shape == (2, 3)
    np.testing.assert_array_equal(runs[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(runs[1], [5.0, 6.0, 7.0])


# --- Tests for plot_mean_with_sem ---


def test_plot_mean_with_sem(tmp_path: Path):
    """plot_mean_with_sem must plot a mean line and fill_between confidence interval."""
    # Tạo 2 tệp CSV giả lập
    csv_paths = []
    for i in range(2):
        path = tmp_path / f"seed_{i}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["return"])
            w.writeheader()
            for val in [float(i), float(i + 1), float(i + 2), float(i + 3)]:
                w.writerow({"return": val})
        csv_paths.append(path)

    fig, ax = plt.subplots()
    plot_mean_with_sem(
        ax=ax,
        csv_paths=csv_paths,
        column="return",
        label="TestAgent",
        window_size=2,
    )

    # Kiểm tra đường vẽ (line) và vùng đổ bóng (PolyCollection)
    assert len(ax.lines) == 1
    assert "TestAgent(N=2)" in ax.lines[0].get_label()
    assert len(ax.collections) >= 1  # fill_between tạo ít nhất 1 PolyCollection
    plt.close(fig)


def test_plot_mean_with_sem_single_seed(tmp_path: Path):
    """A single seed (N=1) must plot without zero-division error (sem = 0)."""
    csv_path = tmp_path / "seed_0.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["return"])
        w.writeheader()
        for val in [1.0, 2.0, 3.0]:
            w.writerow({"return": val})

    fig, ax = plt.subplots()
    plot_mean_with_sem(
        ax=ax,
        csv_paths=[csv_path],
        column="return",
        label="SingleSeed",
        window_size=1,
    )

    assert len(ax.lines) == 1
    assert "SingleSeed(N=1)" in ax.lines[0].get_label()
    plt.close(fig)
