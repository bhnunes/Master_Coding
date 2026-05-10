from __future__ import annotations

import json
from collections.abc import Sequence
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
        "validation_dataset",
        "split_lineage",
        "packaging_lineage",
        "normalization_lineage",
        "smart_sampling_lineage",
        "artifact_aware_loss",
        "validation_lineage",
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
    *,
    requested_architectures: Sequence[str] | None = None,
) -> list[SelectedModelMetadata]:
    valid_sort_metrics = {"best_validation_DICE", "best_val_auprc_pixel_score"}
    if sort_metric not in valid_sort_metrics:
        raise ValueError(
            f"Invalid sort_metric: '{sort_metric}'. Must be one of {sorted(valid_sort_metrics)}"
        )

    metadata_files = sorted(metadata_dir.glob("*_meta.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {metadata_dir}")

    requested_set = (
        {architecture.upper() for architecture in requested_architectures}
        if requested_architectures is not None
        else None
    )
    selected: list[SelectedModelMetadata] = []
    observed_signatures: set[str] = set()
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        architecture = str(payload.get("architecture", "")).upper()
        if requested_set is not None and architecture not in requested_set:
            continue
        metric_value = payload.get(sort_metric)
        if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
            continue
        compatibility_signature = _validate_fail_closed_provenance(payload, metadata_file)
        observed_signatures.add(compatibility_signature)
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

    if not selected and requested_architectures is None:
        raise ValueError("No valid metadata loaded after filtering for sort metric.")
    if len(observed_signatures) > 1:
        raise ValueError(
            "Metadata directory contains incompatible provenance groups; refuse to mix candidates."
        )

    return selected


def select_unique_candidates_by_architecture(
    candidates: list[SelectedModelMetadata],
    *,
    requested_architectures: Sequence[str],
    minimum_models: int = 2,
) -> list[SelectedModelMetadata]:
    ordered_requested = tuple(
        dict.fromkeys(architecture.upper() for architecture in requested_architectures)
    )
    requested_set = set(ordered_requested)
    candidates_by_architecture: dict[str, list[SelectedModelMetadata]] = {
        architecture: [] for architecture in ordered_requested
    }
    for candidate in candidates:
        architecture = candidate.architecture.upper()
        if architecture in requested_set:
            candidates_by_architecture[architecture].append(candidate)

    missing = [
        architecture
        for architecture, architecture_candidates in candidates_by_architecture.items()
        if not architecture_candidates
    ]
    duplicates = {
        architecture: [candidate.metadata_filename for candidate in architecture_candidates]
        for architecture, architecture_candidates in candidates_by_architecture.items()
        if len(architecture_candidates) > 1
    }
    if missing or duplicates:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            duplicate_details = {
                architecture: sorted(filenames) for architecture, filenames in duplicates.items()
            }
            details.append(f"duplicates={duplicate_details}")
        raise ValueError(
            "Phase 9 requires exactly one valid metadata file per requested architecture; "
            + "; ".join(details)
        )

    selected = [candidates_by_architecture[architecture][0] for architecture in ordered_requested]
    if len(selected) < minimum_models:
        raise ValueError(
            "Fewer than 2 requested architectures had valid unique metadata files. "
            f"Requested={list(ordered_requested)} selected={len(selected)}"
        )
    return selected
