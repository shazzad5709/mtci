from __future__ import annotations

from xml.etree import ElementTree as ET

from mtci.reporting import write_junit


def test_junit_generation(tmp_path):
    results = [
        {"name": "mr_pass", "status": "pass", "runtime_s": 0.1, "message": "ok"},
        {"name": "mr_fail", "status": "fail", "runtime_s": 0.2, "message": "fail"},
        {"name": "mr_flaky", "status": "flaky", "runtime_s": 0.3, "message": "flaky"},
    ]
    write_junit(tmp_path, results, junit_flaky_as_failure=True)
    tree = ET.parse(tmp_path / "junit.xml")
    root = tree.getroot()
    names = [case.attrib["name"] for case in root.findall("testcase")]
    assert "mr_pass" in names
    assert root.find(".//failure") is not None
