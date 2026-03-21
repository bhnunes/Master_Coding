from __future__ import annotations

import json
from pathlib import Path

from helpers.extraction.profiling import (
    PhaseStats,
    build_profile_summary,
    create_phase_stats,
    merge_phase_stats,
    record_phase,
    write_profile_summary,
)


def test_merge_phase_stats_accumulates_totals_and_calls() -> None:
    target = create_phase_stats(["read_region", "image_save"])
    source = create_phase_stats(["read_region", "image_save"])
    record_phase(target, "read_region", 1.25)
    record_phase(source, "read_region", 0.75)
    record_phase(source, "image_save", 0.5)

    merge_phase_stats(target, source)

    assert target == {
        "read_region": PhaseStats(total_seconds=2.0, calls=2),
        "image_save": PhaseStats(total_seconds=0.5, calls=1),
    }


def test_build_profile_summary_reports_percentages_and_counts() -> None:
    phase_stats = create_phase_stats(["read_region", "image_save"])
    record_phase(phase_stats, "read_region", 3.0)
    record_phase(phase_stats, "image_save", 1.0)

    summary = build_profile_summary(
        slide_name="sample.svs",
        total_runtime_seconds=10.0,
        candidate_windows=20,
        status_counts={"SAVED_CANCER": 4, "SKIPPED_TISSUE": 6, "SKIPPED_OVERLAP": 10},
        phase_stats=phase_stats,
        slide_phase_seconds={"setup": 2.0, "parallel": 8.0},
    )

    assert summary["slide_name"] == "sample.svs"
    assert summary["accepted_patches"] == 4
    assert summary["skipped_tissue"] == 6
    assert summary["skipped_overlap"] == 10
    assert summary["phases"][0]["name"] == "read_region"
    assert summary["phases"][0]["share_of_runtime"] == 0.3
    assert summary["slide_phases"]["parallel"]["share_of_runtime"] == 0.8


def test_write_profile_summary_persists_json(tmp_path: Path) -> None:
    output_path = tmp_path / "profile.json"
    summary = {"slide_name": "sample.svs", "total_runtime_seconds": 1.23}

    write_profile_summary(output_path, summary)

    assert json.loads(output_path.read_text(encoding="utf-8")) == summary
