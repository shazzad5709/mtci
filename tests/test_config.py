from __future__ import annotations

import textwrap

import pytest

from mtci.config import ConfigError, load_config


def test_config_validation_ok(tmp_path):
    cfg = textwrap.dedent(
        """
        profiles:
          pr-fast:
            budget_seconds: 5
            max_examples: 3
            retries_on_fail: 1
            fail_on_flake: true
            tolerance:
              atol: 0.0
              rtol: 0.1
            mrs:
              - mtci.mrs.whitespace.WhitespaceInvarianceMR
        dataset:
          path: assets/sample.jsonl
          jsonl_field: text
        model:
          mode: local
          entrypoint: mtci.models.simple.SimpleSentimentModel
        """
    )
    path = tmp_path / "mtci.yml"
    path.write_text(cfg)
    config = load_config(path)
    assert "pr-fast" in config.profiles


def test_config_validation_error(tmp_path):
    path = tmp_path / "mtci.yml"
    path.write_text("profiles: {}")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "profiles" in str(exc.value)
