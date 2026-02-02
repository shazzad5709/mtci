from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List

STATE_DIR = ".mtci"
STATE_FILE = "state.json"


@dataclass
class MRStats:
    runs: int = 0
    fails: int = 0
    flaky_count: int = 0
    median_runtime_s: float = 1.0
    runtimes: List[float] | None = None

    def update_runtime(self, runtime: float) -> None:
        if self.runtimes is None:
            self.runtimes = []
        self.runtimes.append(runtime)
        if len(self.runtimes) > 50:
            self.runtimes = self.runtimes[-50:]
        self.median_runtime_s = float(median(self.runtimes))


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / STATE_DIR / STATE_FILE
        self._data: Dict[str, MRStats] = {}

    def load(self) -> Dict[str, MRStats]:
        if not self.path.exists():
            self._data = {}
            return self._data
        raw = json.loads(self.path.read_text())
        data: Dict[str, MRStats] = {}
        for name, stats in raw.get("mrs", {}).items():
            data[name] = MRStats(
                runs=stats.get("runs", 0),
                fails=stats.get("fails", 0),
                flaky_count=stats.get("flaky_count", 0),
                median_runtime_s=stats.get("median_runtime_s", 1.0),
                runtimes=stats.get("runtimes", None),
            )
        self._data = data
        return data

    def save(self) -> None:
        payload = {
            "mrs": {
                name: {
                    "runs": stats.runs,
                    "fails": stats.fails,
                    "flaky_count": stats.flaky_count,
                    "median_runtime_s": stats.median_runtime_s,
                    "runtimes": stats.runtimes,
                }
                for name, stats in self._data.items()
            }
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def get(self, name: str) -> MRStats:
        if name not in self._data:
            self._data[name] = MRStats()
        return self._data[name]
