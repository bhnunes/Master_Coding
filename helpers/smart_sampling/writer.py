from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from helpers.smart_sampling.config import SmartSamplerConfig


def write_filter_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "filter_summary.json"
    logging.info("Writing filter summary to %s", summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    return summary_path


def write_sidecar_artifacts(
    config: SmartSamplerConfig,
    *,
    selection_manifest: list[dict[str, Any]],
    stats_log: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    if not config.write_sidecars:
        return None, None, None

    resolved_output_dir = output_dir or config.output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    selection_csv_path = resolved_output_dir / "train_filtered_selection.csv"
    stats_csv_path = resolved_output_dir / "patient_filter_stats.csv"
    run_config_path = resolved_output_dir / "filter_run_config.json"
    logging.info("Writing smart-sampling sidecars to %s", resolved_output_dir)

    pd.DataFrame(selection_manifest).to_csv(selection_csv_path, index=False)
    pd.DataFrame(stats_log).to_csv(stats_csv_path, index=False)
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, default=str)

    return selection_csv_path, stats_csv_path, run_config_path
