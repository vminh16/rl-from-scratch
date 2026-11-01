"""
Kiểm tra SẢN PHẨM NỘP của Tuần 02 theo lộ trình (mục "Nộp" và "Đạt khi").
Các test này sẽ fail cho đến khi bạn làm xong phần tương ứng: dùng như checklist.
Chỉ chạy test thuật toán:  pytest weeks/w02-bandits -k "not deliverables"
"""

from pathlib import Path

import pytest

WEEK = Path(__file__).resolve().parents[1]  # weeks/w02-bandits
REPO = WEEK.parents[1]                      # gốc repo

README_SECTIONS = ["Mục đích", "Cách chạy", "Kết quả", "Đối chiếu tiêu chí", "Chứng minh", "Ba điều rút ra"]


def test_readme_follows_the_template():
    readme = WEEK / "README.md"
    assert readme.exists() and readme.stat().st_size > 0, "README.md đang rỗng"
    text = readme.read_text(encoding="utf-8")
    missing = [s for s in README_SECTIONS if s not in text]
    assert not missing, f"README thiếu mục: {missing}"


@pytest.mark.parametrize("script", ["run.py", "sweep.py", "plot.py"])
def test_scripts_are_not_empty(script):
    path = WEEK / script
    assert path.exists() and path.stat().st_size > 0, f"{script} đang rỗng"


def test_run_script_exposes_the_documented_flags():
    text = (WEEK / "run.py").read_text(encoding="utf-8")
    for flag in ("--runs", "--steps", "--exp"):
        assert flag in text, f"run.py cần tham số {flag} (xem lộ trình)"


def test_sweep_results_exist():
    sweep = WEEK / "results" / "sweep.csv"
    assert sweep.exists(), "chưa có results/sweep.csv"
    assert len(sweep.read_text(encoding="utf-8").strip().splitlines()) >= 2, "sweep.csv chỉ có tiêu đề"


def test_at_least_five_figures():
    """4 hình (phần thưởng TB, % tay tối ưu, so sánh UCB, quét tham số) + hình bài toán không dừng."""
    pngs = list((WEEK / "figures").glob("*.png"))
    assert len(pngs) >= 5, f"mới có {len(pngs)} hình"


def test_proof_pdf_exists():
    assert (REPO / "proofs" / "w02.pdf").exists(), "chưa có proofs/w02.pdf"
