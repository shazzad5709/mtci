from __future__ import annotations

import json
import textwrap

from mtci.execution import run_profile
from mtci.config import load_config


def test_flake_classification(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{\"text\": \"good\"}\n")

    cfg_text = textwrap.dedent(
        f"""
        profiles:
          pr-fast:
            budget_seconds: 10
            max_examples: 1
            retries_on_fail: 1
            fail_on_flake: true
            tolerance:
              atol: 0.0
              rtol: 0.01
            mrs:
              - mtci.testing_mrs.FailThenPassMR
              - mtci.testing_mrs.AlwaysFailMR
        dataset:
          path: {dataset}
          jsonl_field: text
        model:
          mode: local
          entrypoint: mtci.models.simple.SimpleSentimentModel
        """
    )
    cfg_path = tmp_path / "mtci.yml"
    cfg_path.write_text(cfg_text)

    monkeypatch.chdir(tmp_path)

    cfg = load_config(cfg_path)
    exit_code, out_dir = run_profile(cfg, "pr-fast", tmp_path / "out")
    report = json.loads((out_dir / "report.json").read_text())
    results = {r["name"]: r for r in report["results"]}

    assert results["fail_then_pass"]["status"] == "flaky"
    assert results["always_fail"]["status"] == "fail"
    assert exit_code == 1
