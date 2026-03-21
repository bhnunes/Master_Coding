from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(eq=True)
class PhaseStats:
    total_seconds: float = 0.0
    calls: int = 0


def create_phase_stats(phase_names: list[str] | tuple[str, ...]) -> dict[str, PhaseStats]:
    return {phase_name: PhaseStats() for phase_name in phase_names}


def record_phase(
    phase_stats: dict[str, PhaseStats], phase_name: str, elapsed_seconds: float
) -> None:
    stats = phase_stats.setdefault(phase_name, PhaseStats())
    stats.total_seconds += elapsed_seconds
    stats.calls += 1


def merge_phase_stats(
    target: dict[str, PhaseStats],
    source: dict[str, PhaseStats],
) -> None:
    for phase_name, stats in source.items():
        merged = target.setdefault(phase_name, PhaseStats())
        merged.total_seconds += stats.total_seconds
        merged.calls += stats.calls


def build_profile_summary(
    *,
    slide_name: str,
    total_runtime_seconds: float,
    candidate_windows: int,
    status_counts: dict[str, int],
    phase_stats: dict[str, PhaseStats],
    slide_phase_seconds: dict[str, float],
) -> dict[str, Any]:
    accepted_patches = status_counts.get("SAVED_CANCER", 0) + status_counts.get(
        "SAVED_NOT_CANCER", 0
    )
    ordered_phase_items = sorted(
        ((phase_name, stats) for phase_name, stats in phase_stats.items() if stats.calls > 0),
        key=lambda item: item[1].total_seconds,
        reverse=True,
    )
    phase_rows = [
        {
            "name": phase_name,
            "total_seconds": round(stats.total_seconds, 6),
            "calls": stats.calls,
            "avg_seconds": round(stats.total_seconds / stats.calls, 6) if stats.calls else 0.0,
            "share_of_runtime": round(stats.total_seconds / total_runtime_seconds, 6)
            if total_runtime_seconds
            else 0.0,
        }
        for phase_name, stats in ordered_phase_items
    ]

    slide_phases = {
        phase_name: {
            "seconds": round(seconds, 6),
            "share_of_runtime": round(seconds / total_runtime_seconds, 6)
            if total_runtime_seconds
            else 0.0,
        }
        for phase_name, seconds in slide_phase_seconds.items()
    }

    return {
        "slide_name": slide_name,
        "total_runtime_seconds": round(total_runtime_seconds, 6),
        "candidate_windows": candidate_windows,
        "accepted_patches": accepted_patches,
        "skipped_tissue": status_counts.get("SKIPPED_TISSUE", 0),
        "skipped_overlap": status_counts.get("SKIPPED_OVERLAP", 0),
        "errors": status_counts.get("ERROR", 0),
        "status_counts": status_counts,
        "phases": phase_rows,
        "slide_phases": slide_phases,
    }


def write_profile_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
