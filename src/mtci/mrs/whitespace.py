from __future__ import annotations

from typing import Sequence

from mtci.mrs.base import BaseMR, MRFailure, MRResult, within_tolerance


class WhitespaceInvarianceMR(BaseMR):
    name = "whitespace_invariance"
    description = "Extra whitespace should not change output"

    @staticmethod
    def _transform(text: str) -> str:
        return "\n  " + text.replace(" ", "  ") + "  \n"

    def run(self, model, inputs: Sequence[str], max_examples: int, tolerance):
        failures: list[MRFailure] = []
        n = min(len(inputs), max_examples)
        for i in range(n):
            original = inputs[i]
            transformed = self._transform(original)
            out_a = model.predict([original])[0]
            out_b = model.predict([transformed])[0]
            if not within_tolerance(out_a, out_b, tolerance):
                failures.append(
                    MRFailure(
                        index=i,
                        original=original,
                        transformed=transformed,
                        output_original=out_a,
                        output_transformed=out_b,
                        diff=abs(out_a - out_b),
                    )
                )
        passed = len(failures) == 0
        message = "pass" if passed else f"{len(failures)} mismatches"
        return MRResult(self.name, passed, message, failures)
