from __future__ import annotations

from typing import Sequence

from mtci.mrs.base import BaseMR, MRFailure, MRResult
from mtci.config import Tolerance

# Module-level flag: persists within ONE `mtci run` process,
# but resets on the next run (new process).
_SEEN_ONCE = False


class FlakeDemoMR(BaseMR):
    """
    Intentionally flaky MR for testing flake-aware gating:
    - attempt #1: FAIL
    - attempt #2+: PASS
    """
    name = "flake_demo"
    description = "Intentionally flaky MR for testing flake-aware gating."

    def run(
        self,
        model,
        inputs: Sequence[str],
        max_examples: int,
        tolerance: Tolerance,
    ) -> MRResult:
        global _SEEN_ONCE

        if not _SEEN_ONCE:
            _SEEN_ONCE = True
            text = inputs[0] if inputs else ""
            failures = [
                MRFailure(
                    index=0,
                    original=text,
                    transformed=None,
                    output_original=0.0,
                    output_transformed=1.0,
                    diff=1.0,
                )
            ]
            return MRResult(
                self.name,
                False,
                "Intentional flake: failing first attempt only.",
                failures,
            )

        return MRResult(self.name, True, "OK (passed on retry).", [])
