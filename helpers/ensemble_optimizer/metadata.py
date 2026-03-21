from __future__ import annotations

import json
from collections.abc import Callable
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


def _validate_fail_closed_provenance(payload: dict[str, Any], metadata_file: Path) -> str:
    compatibility_signature = payload.get("compatibility_signature")
    provenance = payload.get("provenance")
    if not isinstance(compatibility_signature, str) or not compatibility_signature.strip():
        raise ValueError(
            "Metadata file "
            f"'{metadata_file.name}' is missing fail-closed provenance compatibility_signature."
        )
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Metadata file '{metadata_file.name}' is missing fail-closed provenance payload."
        )

    required_sections = {
        "dataset",
        "split_lineage",
        "packaging_lineage",
        "normalization_lineage",
        "smart_sampling_lineage",
        "artifact_aware_loss",
    }
    missing_sections = sorted(section for section in required_sections if section not in provenance)
    if missing_sections:
        raise ValueError(
            f"Metadata file '{metadata_file.name}' is missing fail-closed provenance sections: "
            f"{missing_sections}."
        )
    return compatibility_signature.strip()


def load_model_candidates(
    metadata_dir: Path,
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
    observed_signatures: set[str] = set()
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        metric_value = payload.get(sort_metric)
        if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
            continue
        compatibility_signature = _validate_fail_closed_provenance(payload, metadata_file)
        observed_signatures.add(compatibility_signature)
        architecture = str(payload.get("architecture", "")).upper()
        encoder = str(payload.get("encoder", "")).strip()
        checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
        if not architecture or not encoder or not checkpoint_path:
            continue
        payload.setdefault("_metadata_path", str(metadata_file))
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
    if len(observed_signatures) > 1:
        raise ValueError(
            "Metadata directory contains incompatible provenance groups; refuse to mix candidates."
        )

    return selected


def select_top_models(
    candidates: list[SelectedModelMetadata],
    *,
    n_top_models: int,
    score_getter: Callable[[SelectedModelMetadata], float] | None = None,
) -> list[SelectedModelMetadata]:
    getter = score_getter or (lambda item: item.sort_metric_value)
    ranked = sorted(candidates, key=getter, reverse=True)
    return ranked[:n_top_models]


def load_and_select_models(
    metadata_dir: Path,
    n_top_models: int,
    sort_metric: str,
) -> list[SelectedModelMetadata]:
    candidates = load_model_candidates(metadata_dir, sort_metric)
    return select_top_models(candidates, n_top_models=n_top_models)
