from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mtci.config import Tolerance


@dataclass
class MRFailure:
    index: int
    original: str
    transformed: str | None
    output_original: float
    output_transformed: float
    diff: float


@dataclass
class MRResult:
    name: str
    passed: bool
    message: str
    failures: list[MRFailure]


class BaseMR:
    name: str = "base"
    description: str = ""
    requires_endpoint: bool = False

    def run(
        self,
        model,
        inputs: Sequence[str],
        max_examples: int,
        tolerance: Tolerance,
    ) -> MRResult:
        raise NotImplementedError


def within_tolerance(a: float, b: float, tol: Tolerance) -> bool:
    diff = abs(a - b)
    return diff <= tol.atol + tol.rtol * abs(b)
