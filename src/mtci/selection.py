from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from mtci.state import MRStats


@dataclass
class SelectionMetadata:
    name: str
    score: float
    predicted_runtime_s: float
    reason: str


def score_mr(stats: MRStats) -> float:
    return (stats.fails + 1) / (stats.median_runtime_s + 0.1)


def select_mrs(
    mr_names: Iterable[str],
    stats_by_name: dict[str, MRStats],
    budget_seconds: float,
    smoke_count: int = 2,
    default_runtime: float = 1.0,
) -> List[SelectionMetadata]:
    names = list(mr_names)
    selections: List[SelectionMetadata] = []
    remaining = budget_seconds

    if not names:
        return selections

    cold = all(stats_by_name.get(name) is None for name in names)
    if cold:
        for name in names[:smoke_count]:
            runtime = default_runtime
            selections.append(
                SelectionMetadata(
                    name=name,
                    score=1.0 / (runtime + 0.1),
                    predicted_runtime_s=runtime,
                    reason="cold-start smoke MR",
                )
            )
            remaining -= runtime
        names = [n for n in names if n not in {s.name for s in selections}]

    scored = []
    for name in names:
        stats = stats_by_name.get(name) or MRStats()
        runtime = stats.median_runtime_s if stats.median_runtime_s > 0 else default_runtime
        scored.append((name, score_mr(stats), runtime))

    scored.sort(key=lambda item: (-item[1], item[0]))

    for name, score, runtime in scored:
        if runtime <= remaining:
            selections.append(
                SelectionMetadata(
                    name=name,
                    score=score,
                    predicted_runtime_s=runtime,
                    reason="score-ranked",
                )
            )
            remaining -= runtime

    if not selections:
        name = names[0]
        selections.append(
            SelectionMetadata(
                name=name,
                score=1.0,
                predicted_runtime_s=default_runtime,
                reason="budget fallback",
            )
        )

    return selections
