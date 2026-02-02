from __future__ import annotations

from mtci.mrs.base import BaseMR, MRResult


class FailThenPassMR(BaseMR):
    name = "fail_then_pass"

    def __init__(self):
        self._called = 0

    def run(self, model, inputs, max_examples, tolerance):
        self._called += 1
        if self._called == 1:
            return MRResult(self.name, False, "first fail", [])
        return MRResult(self.name, True, "pass", [])


class AlwaysFailMR(BaseMR):
    name = "always_fail"

    def run(self, model, inputs, max_examples, tolerance):
        return MRResult(self.name, False, "fail", [])
