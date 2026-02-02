from __future__ import annotations

from typing import Sequence

from mtci.mrs.base import BaseMR, MRFailure, MRResult, within_tolerance


class BatchingInvarianceMR(BaseMR):
    name = "batching_invariance"
    description = "Single-item predictions should match their batch position"

    def run(self, model, inputs: Sequence[str], max_examples: int, tolerance):
        failures: list[MRFailure] = []
        n = min(len(inputs), max_examples)
        if n < 2:
            return MRResult(self.name, True, "not enough samples", failures)
        for i in range(n - 1):
            x = inputs[i]
            y = inputs[i + 1]
            single = model.predict([x])[0]
            batch = model.predict([x, y])[0]
            if not within_tolerance(single, batch, tolerance):
                failures.append(
                    MRFailure(
                        index=i,
                        original=x,
                        transformed=y,
                        output_original=single,
                        output_transformed=batch,
                        diff=abs(single - batch),
                    )
                )
        passed = len(failures) == 0
        message = "pass" if passed else f"{len(failures)} mismatches"
        return MRResult(self.name, passed, message, failures)
