import csv
from pathlib import Path
from minirl.utils.logger import CSVLogger


def test_logger_creates_file_and_header(tmp_path: Path):
    """CSVLogger must create parent directories and write header immediately."""
    csv_file = tmp_path / "nested" / "dir" / "test.csv"
    fields = ["episode", "return", "length"]

    logger = CSVLogger(csv_file, fieldnames=fields)
    logger.close()

    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 1
        assert reader[0] == ["episode", "return", "length"]


def test_logger_writes_rows_and_flushes(tmp_path: Path):
    """CSVLogger must write row data and flush changes."""
    csv_file = tmp_path / "test.csv"
    fields = ["episode", "return", "length"]

    with CSVLogger(csv_file, fieldnames=fields) as logger:
        logger.log({"episode": 0, "return": 1.5, "length": 10})
        logger.log({"episode": 1, "return": -0.5, "length": 4})

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["episode"] == "0"
        assert float(reader[0]["return"]) == 1.5
        assert int(reader[0]["length"]) == 10
        assert reader[1]["episode"] == "1"
        assert float(reader[1]["return"]) == -0.5
        assert int(reader[1]["length"]) == 4


def test_logger_ignores_extra_fields(tmp_path: Path):
    """CSVLogger must safely ignore extra dictionary keys not in fieldnames."""
    csv_file = tmp_path / "test.csv"
    fields = ["episode", "return"]

    with CSVLogger(csv_file, fieldnames=fields) as logger:
        logger.log({"episode": 0, "return": 10.0, "unwanted_metric": 42})

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert "unwanted_metric" not in reader[0]
        assert reader[0]["return"] == "10.0"


def test_logger_context_manager_closes_file(tmp_path: Path):
    """Exiting the with block must properly close the underlying file."""
    csv_file = tmp_path / "test.csv"

    with CSVLogger(csv_file, fieldnames=["ep"]) as logger:
        assert not logger._file.closed

    assert logger._file.closed
