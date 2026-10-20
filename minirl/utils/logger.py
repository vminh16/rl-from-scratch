"""
Công cụ ghi chép dữ liệu huấn luyện (Episodic CSV Logger)

"""

import csv
from pathlib import Path
from typing import Any, Sequence

__all__ = ["CSVLogger"]

class CSVLogger:
    """
    Ghi log chỉ số của từng episodic vào tệ CSV
    """
    def __init__(self, filepath: str | Path, fieldnames: Sequence[str]):
        self.filepath = Path(filepath)
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        self._file = open(self.filepath, "w", newline="", encoding='utf-8')
        
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction='ignore'  # bỏ qua key không có trong fieldnames
        )
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        #đóng file thủ công
        if self._file and not self._file.closed:
            self._file.close()
    
    # cho phép dùng "with" statement
    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
	