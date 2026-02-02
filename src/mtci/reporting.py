from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


def write_junit(
    out_dir: Path,
    results: list[dict],
    junit_flaky_as_failure: bool,
) -> None:
    testsuite = ET.Element("testsuite", name="mtci")
    for result in results:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            classname="mtci",
            name=result["name"],
            time=f"{result.get('runtime_s', 0):.3f}",
        )
        status = result["status"]
        message = result.get("message", "")
        if status == "fail":
            failure = ET.SubElement(testcase, "failure", message=message or "fail")
            failure.text = message
        elif status == "flaky":
            if junit_flaky_as_failure:
                failure = ET.SubElement(testcase, "failure", message="flaky")
                failure.text = message or "flaky"
            else:
                skipped = ET.SubElement(testcase, "skipped", message="flaky")
                skipped.text = message or "flaky"
        elif status == "skipped":
            skipped = ET.SubElement(testcase, "skipped", message=message or "skipped")
            skipped.text = message or "skipped"

    tree = ET.ElementTree(testsuite)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree.write(out_dir / "junit.xml", encoding="utf-8", xml_declaration=True)
