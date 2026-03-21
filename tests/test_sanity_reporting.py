from __future__ import annotations

import pytest

from helpers.sanity.models import CheckResult, SanityReport
from helpers.sanity.reporting import build_fatal_error_report, format_status, render_sanity_report


def test_format_status_uses_expected_icons() -> None:
    assert format_status("PASS") == "[PASS] PASS"
    assert format_status("WARN") == "[WARN] WARN"
    assert format_status("FAIL") == "[FAIL] FAIL"
    assert format_status("N/A") == "[N/A] N/A"
    assert format_status("UNKNOWN") == "[N/A] UNKNOWN"


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

    rendered = render_sanity_report(report)

    assert "SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT" in rendered
    assert "Platform: TestOS-1.0" in rendered
    assert "Manifest" in rendered
    assert "Parity" in rendered
    assert "[SPLIT SUMMARY TABLE]" in rendered
    assert "TRAIN" in rendered
    assert "VALIDATION" in rendered
    assert "TEST" in rendered
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

    rendered = render_sanity_report(report)

    assert "No split-level checks were recorded." in rendered
    assert "FINAL VERDICT: SPLITS APPROVED" in rendered


def test_build_fatal_error_report_marks_splits_rejected() -> None:
    rendered = build_fatal_error_report(RuntimeError("boom"))

    assert "[FAIL] FATAL ERROR: boom" in rendered
    assert "FINAL VERDICT: SPLITS REJECTED" in rendered
