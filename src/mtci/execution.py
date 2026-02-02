from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from mtci.adapters import BaseModelAdapter, HTTPEndpointModel, build_adapter
from mtci.config import Config, Profile
from mtci.data import load_jsonl
from mtci.mrs.base import BaseMR, MRResult
from mtci.reporting import write_junit, write_report
from mtci.selection import SelectionMetadata, select_mrs
from mtci.state import MRStats, StateStore


class MRLoadError(Exception):
    pass


@dataclass
class MRRunResult:
    name: str
    status: str
    attempts: int
    runtime_s: float
    message: str
    failures: list[dict]


def load_mr(entrypoint: str) -> BaseMR:
    import importlib

    if ":" in entrypoint:
        module_name, attr = entrypoint.split(":", 1)
    else:
        module_name, attr = entrypoint.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr)
    if isinstance(target, type):
        return target()
    if callable(target):
        return target()
    if isinstance(target, BaseMR):
        return target
    raise MRLoadError(f"Invalid MR entrypoint: {entrypoint}")


def _filter_mrs(mrs: Iterable[BaseMR], model: BaseModelAdapter) -> list[BaseMR]:
    filtered = []
    for mr in mrs:
        if mr.requires_endpoint and not isinstance(model, HTTPEndpointModel):
            continue
        filtered.append(mr)
    return filtered


def _serialize_failures(failures: list) -> list[dict]:
    return [asdict(failure) for failure in failures]


def run_profile(config: Config, profile_name: str, out_root: str | Path) -> tuple[int, Path]:
    if profile_name not in config.profiles:
        raise ValueError(f"Profile not found: {profile_name}")
    profile: Profile = config.profiles[profile_name]
    data = load_jsonl(config.dataset.path, config.dataset.jsonl_field)
    model = build_adapter(config.model)

    mr_instances = [load_mr(entry) for entry in profile.mrs]
    mr_instances = _filter_mrs(mr_instances, model)

    store = StateStore(Path.cwd())
    stats = store.load()

    selection = select_mrs(
        [mr.name for mr in mr_instances],
        stats,
        profile.budget_seconds,
    )

    mr_by_name = {mr.name: mr for mr in mr_instances}
    selected_mrs = [mr_by_name[item.name] for item in selection if item.name in mr_by_name]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / f"{profile_name}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[MRRunResult] = []
    total_retries = 0
    flaky_count = 0

    start_time = time.perf_counter()

    for mr in selected_mrs:
        elapsed = time.perf_counter() - start_time
        if elapsed >= profile.budget_seconds:
            results.append(
                MRRunResult(
                    name=mr.name,
                    status="skipped",
                    attempts=0,
                    runtime_s=0.0,
                    message="budget exceeded",
                    failures=[],
                )
            )
            continue

        attempts = 0
        failures: list[dict] = []
        status = "fail"
        message = ""
        runtime_s = 0.0

        for attempt in range(profile.retries_on_fail + 1):
            attempts += 1
            attempt_start = time.perf_counter()
            result: MRResult = mr.run(model, data, profile.max_examples, profile.tolerance)
            runtime_s = time.perf_counter() - attempt_start
            if result.passed:
                status = "pass" if attempt == 0 else "flaky"
                message = result.message
                failures = _serialize_failures(result.failures)
                break
            message = result.message
            failures = _serialize_failures(result.failures)
            if attempt < profile.retries_on_fail:
                total_retries += 1
        else:
            status = "fail"

        if status == "flaky":
            flaky_count += 1

        results.append(
            MRRunResult(
                name=mr.name,
                status=status,
                attempts=attempts,
                runtime_s=runtime_s,
                message=message,
                failures=failures,
            )
        )

        stats_entry = stats.get(mr.name) or MRStats()
        stats_entry.runs += 1
        if status == "fail":
            stats_entry.fails += 1
        if status == "flaky":
            stats_entry.flaky_count += 1
        stats_entry.update_runtime(runtime_s)
        stats[mr.name] = stats_entry

        failure_dir = out_dir / "failures" / mr.name
        if status in {"fail", "flaky"}:
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "message.txt").write_text(message)
            (failure_dir / "failures.json").write_text(json.dumps(failures, indent=2))

    store.save()

    report = {
        "profile": profile_name,
        "budget_seconds": profile.budget_seconds,
        "max_examples": profile.max_examples,
        "selected_mrs": [
            {
                "name": item.name,
                "score": item.score,
                "predicted_runtime_s": item.predicted_runtime_s,
                "reason": item.reason,
            }
            for item in selection
        ],
        "results": [asdict(result) for result in results],
        "flake_summary": {
            "total_retries": total_retries,
            "flaky_count": flaky_count,
            "fail_on_flake": profile.fail_on_flake,
        },
    }

    write_report(out_dir, report)
    write_junit(out_dir, [asdict(result) for result in results], profile.junit_flaky_as_failure)

    exit_code = 0
    if any(r.status == "fail" for r in results):
        exit_code = 1
    elif profile.fail_on_flake and any(r.status == "flaky" for r in results):
        exit_code = 1

    return exit_code, out_dir
