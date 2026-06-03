from __future__ import annotations

import pytest

from helpers.sanity import reporting as sanity_reporting_module
from helpers.sanity.models import CheckResult, SanityReport

EXPECTED_SPLIT_HEADERS = 3


def test_format_status_uses_expected_icons() -> None:
    assert sanity_reporting_module.format_status("PASS") == "[PASS] PASS"
    assert sanity_reporting_module.format_status("WARN") == "[WARN] WARN"
    assert sanity_reporting_module.format_status("FAIL") == "[FAIL] FAIL"
    assert sanity_reporting_module.format_status("N/A") == "[N/A] N/A"
    assert sanity_reporting_module.format_status("UNKNOWN") == "[N/A] UNKNOWN"


def test_render_sanity_report_includes_dataset_and_split_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.platform", lambda: "TestOS-1.0")
    report = SanityReport(
        base_dir="/tmp/project",
        dataset_checks={
            "Manifest": CheckResult(status="PASS", details="All rows valid"),
            "Parity": CheckResult(status="WARN", details="One missing stat"),
        },
        split_checks={
            "TRAIN": {
                "Pairs": CheckResult(status="PASS", details="Exact match"),
            },
            "VALIDATION": {
                "Pairs": CheckResult(status="WARN", details="Sampled only"),
            },
            "TEST": {
                "Pairs": CheckResult(status="FAIL", details="Mismatch found"),
            },
        },
        verdict="SPLITS REJECTED",
    )

    rendered = sanity_reporting_module.render_sanity_report(report)
    lines = rendered.splitlines()

    assert lines[:6] == [
        "=" * 88,
        "SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT",
        "=" * 88,
        "Base directory: /tmp/project",
        "Platform: TestOS-1.0",
        "-" * 88,
    ]
    assert "Manifest" in rendered
    assert "Parity" in rendered
    assert "[SPLIT SUMMARY TABLE]" in rendered
    assert "Pairs      " in rendered
    assert "TRAIN" in rendered
    assert "VALIDATION" in rendered
    assert "TEST" in rendered
    assert "[PASS] PASS  Exact match" in rendered
    assert "[WARN] WARN  Sampled only" in rendered
    assert "[FAIL] FAIL  Mismatch found" in rendered
    assert "Mismatch found" in rendered
    assert "FINAL VERDICT: SPLITS REJECTED" in rendered


def test_render_sanity_report_handles_missing_split_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.platform", lambda: "TestOS-1.0")
    report = SanityReport(
        base_dir="/tmp/project",
        dataset_checks={"Manifest": CheckResult(status="PASS", details="All rows valid")},
        split_checks={},
        verdict="SPLITS APPROVED",
    )

    rendered = sanity_reporting_module.render_sanity_report(report)

    assert "No split-level checks were recorded." in rendered
    assert rendered.count("SPLIT: ") == EXPECTED_SPLIT_HEADERS
    assert "FINAL VERDICT: SPLITS APPROVED" in rendered


def test_build_fatal_error_report_marks_splits_rejected() -> None:
    rendered = sanity_reporting_module.build_fatal_error_report(RuntimeError("boom"))

    assert rendered.splitlines() == [
        "=" * 88,
        "SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT",
        "=" * 88,
        "[FAIL] FATAL ERROR: boom",
        "FINAL VERDICT: SPLITS REJECTED",
    ]


def test_reporting_namespace_smoke_uses_module_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.platform", lambda: "SmokeOS")
    report = SanityReport(
        base_dir="/tmp/smoke",
        dataset_checks={"Dataset": CheckResult(status="PASS", details="ok")},
        split_checks={},
        verdict="SPLITS APPROVED",
    )

    assert sanity_reporting_module.format_status("PASS") == "[PASS] PASS"
    assert "Base directory: /tmp/smoke" in sanity_reporting_module.render_sanity_report(report)
    assert "FATAL ERROR: smoke" in sanity_reporting_module.build_fatal_error_report(
        RuntimeError("smoke")
    )


def test_sanity_reporting_namespace_smoke_covers_rendering_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.platform", lambda: "SmokeOS-2.0")
    detailed_report = SanityReport(
        base_dir="/tmp/full",
        dataset_checks={"Dataset": CheckResult(status="PASS", details="ok")},
        split_checks={
            "TRAIN": {"Pairs": CheckResult(status="PASS", details="good")},
            "VALIDATION": {"Pairs": CheckResult(status="WARN", details="sampled")},
            "TEST": {"Pairs": CheckResult(status="FAIL", details="bad")},
        },
        verdict="SPLITS REJECTED",
    )
    empty_report = SanityReport(
        base_dir="/tmp/empty",
        dataset_checks={"Dataset": CheckResult(status="PASS", details="ok")},
        split_checks={},
        verdict="SPLITS APPROVED",
    )

    rendered = sanity_reporting_module.render_sanity_report(detailed_report)
    empty_rendered = sanity_reporting_module.render_sanity_report(empty_report)
    fatal_rendered = sanity_reporting_module.build_fatal_error_report(RuntimeError("fatal"))

    assert sanity_reporting_module.format_status("WARN") == "[WARN] WARN"
    assert rendered.splitlines()[0] == "=" * 88
    assert rendered.splitlines()[5] == "-" * 88
    assert "[PASS] PASS  good" in rendered
    assert "[WARN] WARN  sampled" in rendered
    assert "[FAIL] FAIL  bad" in rendered
    assert "FINAL VERDICT: SPLITS REJECTED" in rendered
    assert "No split-level checks were recorded." in empty_rendered
    assert "FINAL VERDICT: SPLITS APPROVED" in empty_rendered
    assert fatal_rendered.splitlines()[0] == "=" * 88
    assert fatal_rendered.splitlines()[3] == "[FAIL] FATAL ERROR: fatal"
