from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectedModelMetadata:
    architecture: str
    encoder: str
    checkpoint_path: str
    metadata_filename: str
    sort_metric_value: float
    raw_metadata: dict[str, Any]


def load_and_select_models(
    metadata_dir: Path,
    n_top_models: int,
    sort_metric: str,
) -> list[SelectedModelMetadata]:
    valid_sort_metrics = {"best_validation_DICE", "best_val_auprc_pixel_score"}
    if sort_metric not in valid_sort_metrics:
        raise ValueError(
            f"Invalid sort_metric: '{sort_metric}'. Must be one of {sorted(valid_sort_metrics)}"
        )

    metadata_files = sorted(metadata_dir.glob("*_meta.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {metadata_dir}")

    selected: list[SelectedModelMetadata] = []
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        metric_value = payload.get(sort_metric)
        if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
            continue
        architecture = str(payload.get("architecture", "")).upper()
        encoder = str(payload.get("encoder", "")).strip()
        checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
        if not architecture or not encoder or not checkpoint_path:
            continue
        selected.append(
            SelectedModelMetadata(
                architecture=architecture,
                encoder=encoder,
                checkpoint_path=checkpoint_path,
                metadata_filename=metadata_file.name,
                sort_metric_value=float(metric_value),
                raw_metadata=payload,
            )
        )

    if not selected:
        raise ValueError("No valid metadata loaded after filtering for sort metric.")

    selected.sort(key=lambda item: item.sort_metric_value, reverse=True)
    return selected[:n_top_models]
