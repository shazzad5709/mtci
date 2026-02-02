from __future__ import annotations

from mtci.selection import select_mrs
from mtci.state import MRStats


def test_budgeted_selection_deterministic():
    stats = {
        "mr_a": MRStats(runs=3, fails=2, flaky_count=0, median_runtime_s=2.0),
        "mr_b": MRStats(runs=5, fails=0, flaky_count=1, median_runtime_s=1.0),
        "mr_c": MRStats(runs=1, fails=1, flaky_count=0, median_runtime_s=4.0),
    }
    selection = select_mrs(["mr_a", "mr_b", "mr_c"], stats, budget_seconds=3)
    names = [item.name for item in selection]
    assert "mr_b" in names
    assert len(selection) >= 1


def test_cold_start_smoke():
    selection = select_mrs(["mr_x", "mr_y", "mr_z"], {}, budget_seconds=2)
    assert selection[0].reason == "cold-start smoke MR"
    assert selection[0].name == "mr_x"
