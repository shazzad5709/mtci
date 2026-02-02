from __future__ import annotations

import json
from pathlib import Path
from typing import List


class DatasetError(Exception):
    pass


def load_jsonl(path: str | Path, field: str) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    values: List[str] = []
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSONL at line {idx + 1}: {exc}") from exc
        if field not in data:
            raise DatasetError(f"Missing field '{field}' at line {idx + 1}")
        values.append(str(data[field]))
    if not values:
        raise DatasetError("Dataset is empty")
    return values
