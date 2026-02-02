from __future__ import annotations

import json
from typing import Sequence

from mtci.adapters import HTTPEndpointModel
from mtci.mrs.base import BaseMR, MRFailure, MRResult, within_tolerance


class SerializationInvarianceMR(BaseMR):
    name = "serialization_invariance"
    description = "Equivalent JSON payloads should yield the same output"
    requires_endpoint = True

    def run(self, model, inputs: Sequence[str], max_examples: int, tolerance):
        if not isinstance(model, HTTPEndpointModel):
            return MRResult(self.name, True, "endpoint-only MR", [])
        failures: list[MRFailure] = []
        n = min(len(inputs), max_examples)
        for i in range(n):
            text = inputs[i]
            payload = {"inputs": [text]}
            raw_a = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            raw_b = json.dumps(payload, indent=2, sort_keys=False)
            out_a = model.post_raw(raw_a)[0]
            out_b = model.post_raw(raw_b)[0]
            if not within_tolerance(out_a, out_b, tolerance):
                failures.append(
                    MRFailure(
                        index=i,
                        original=text,
                        transformed=text,
                        output_original=out_a,
                        output_transformed=out_b,
                        diff=abs(out_a - out_b),
                    )
                )
        passed = len(failures) == 0
        message = "pass" if passed else f"{len(failures)} mismatches"
        return MRResult(self.name, passed, message, failures)
