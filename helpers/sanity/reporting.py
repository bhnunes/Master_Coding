from __future__ import annotations

import platform

import pandas as pd

from helpers.sanity.models import SPLITS, SanityReport


def format_status(status: str) -> str:
    icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "N/A": "[N/A]"}
    return f"{icon.get(status, '[N/A]')} {status}"


def render_sanity_report(report: SanityReport) -> str:
    lines = [
        "=" * 88,
        "SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT",
        "=" * 88,
        f"Base directory: {report.base_dir}",
        f"Platform: {platform.platform()}",
        "-" * 88,
        "",
        "[DATASET-WIDE CHECKS]",
    ]
    for name, result in report.dataset_checks.items():
        lines.append(f"  - {name:<26} {format_status(result.status)}  {result.details}")
    lines.extend(["", "[SPLIT SUMMARY TABLE]"])
    if report.split_checks:
        rows: list[dict[str, str]] = []
        for split_name in SPLITS:
            row = {"Split": split_name}
            for check_name, result in report.split_checks.get(split_name, {}).items():
                row[check_name] = result.status
            rows.append(row)
        lines.append(pd.DataFrame(rows).set_index("Split").to_string())
    else:
        lines.append("No split-level checks were recorded.")
    lines.extend(["", "[DETAILED SPLIT FINDINGS]"])
    for split_name in SPLITS:
        lines.extend(["", "-" * 88, f"SPLIT: {split_name}", "-" * 88])
        for name, result in report.split_checks.get(split_name, {}).items():
            lines.append(f"  - {name:<26} {format_status(result.status)}  {result.details}")
    lines.extend(["", "=" * 88, f"FINAL VERDICT: {report.verdict}", "=" * 88])
    return "\n".join(lines)


def build_fatal_error_report(error: Exception) -> str:
    return "\n".join(
        [
            "=" * 88,
            "SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT",
            "=" * 88,
            f"[FAIL] FATAL ERROR: {error}",
            "FINAL VERDICT: SPLITS REJECTED",
        ]
    )
