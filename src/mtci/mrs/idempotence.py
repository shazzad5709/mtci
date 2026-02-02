from __future__ import annotations

from typing import Sequence

from mtci.mrs.base import BaseMR, MRFailure, MRResult, within_tolerance


class IdempotenceMR(BaseMR):
    name = "idempotence"
    description = "Same request should yield same response"
    requires_endpoint = True

    def run(self, model, inputs: Sequence[str], max_examples: int, tolerance):
        failures: list[MRFailure] = []
        n = min(len(inputs), max_examples)
        if n == 0:
            return MRResult(self.name, True, "no samples", failures)
        batch = list(inputs[:n])
        out_a = model.predict(batch)
        out_b = model.predict(batch)
        for i, (a, b) in enumerate(zip(out_a, out_b)):
            if not within_tolerance(a, b, tolerance):
                failures.append(
                    MRFailure(
                        index=i,
                        original=batch[i],
                        transformed=batch[i],
                        output_original=a,
                        output_transformed=b,
                        diff=abs(a - b),
                    )
                )
        passed = len(failures) == 0
        message = "pass" if passed else f"{len(failures)} mismatches"
        return MRResult(self.name, passed, message, failures)
