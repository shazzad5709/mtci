# mtci

Metamorphic Testing for CI (MTCI) is a CI-native orchestration tool for running metamorphic relations (MRs) against ML systems under strict time budgets, with flake-aware gating and CI-friendly reports.

## Install (uv)

```bash
uv sync --dev
```

Optional Hugging Face support:

```bash
uv sync --dev --extra hf
```

## Run locally (local model mode)

```bash
uv run mtci doctor --config mtci.yml
uv run mtci run --profile pr-fast --out mtci_artifacts
```

## Run locally (endpoint mode)

Terminal A:

```bash
uv run mtci serve
```

Terminal B:

```bash
uv run mtci run --config mtci.endpoint.yml --profile pr-fast --out mtci_artifacts
```

## Config example

```yaml
profiles:
  pr-fast:
    budget_seconds: 8
    max_examples: 5
    retries_on_fail: 1
    fail_on_flake: true
    tolerance:
      atol: 0.0
      rtol: 0.05
    mrs:
      - mtci.mrs.batching.BatchingInvarianceMR
      - mtci.mrs.whitespace.WhitespaceInvarianceMR
      - mtci.mrs.idempotence.IdempotenceMR
      - mtci.mrs.serialization.SerializationInvarianceMR
    junit_flaky_as_failure: true

dataset:
  path: assets/sample.jsonl
  jsonl_field: text

model:
  mode: endpoint
  base_url: http://127.0.0.1:8000
  predict_path: /predict
  timeout_s: 10
```

## Add a new MR

1) Create a class that implements `BaseMR.run`.
2) Return `MRResult` with `failures` populated for diffs.
3) Add the class entrypoint to the `mrs` list in your profile.

Example:

```python
from mtci.mrs.base import BaseMR, MRResult

class MyMR(BaseMR):
    name = "my_mr"
    def run(self, model, inputs, max_examples, tolerance):
        return MRResult(self.name, True, "pass", [])
```

## Reports

`report.json` includes:
- selected MR list with score/runtime metadata
- per-MR result status, attempts, runtime, message, failures
- flake summary and retry counts

`junit.xml` includes one testcase per MR. Flaky results are encoded as failures by default; set `junit_flaky_as_failure: false` per profile to emit `<skipped>` instead.
