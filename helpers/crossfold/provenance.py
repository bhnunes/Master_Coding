from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helpers.provenance import (
    collect_hdf5_provenance,
    collect_runtime_environment,
    get_git_commit_hash,
)


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_hdf5_manifest_from_split_dfs(
    output_dir: Path,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    split_data: dict[str, Any],
) -> pd.DataFrame:
    """Build a deterministic manifest for Stage 5 split assignments."""

    del output_dir
    rows: list[dict[str, Any]] = []

    def add_rows(split_name: str, split_df: pd.DataFrame) -> None:
        if split_df.empty:
            return
        for row_index, row in enumerate(split_df.itertuples(index=False)):
            rows.append(
                {
                    "run_id": run_id,
                    "split": split_name,
                    "label": int(row.label),
                    "patient_id": int(row.patient_id),
                    "filename": row.filename,
                    "normalization_method": normalization_method,
                    "is_normalized": bool(is_normalized),
                    "source_hdf5_path": getattr(row, "source_hdf5_path", ""),
                    "source_row_index": int(getattr(row, "source_row_index", row_index)),
                    "source_image_path": getattr(row, "image_path", ""),
                    "source_mask_path": getattr(row, "mask_path", ""),
                }
            )

    add_rows("TRAIN", split_data["train_df"])
    add_rows("VALIDATION", split_data["val_df"])
    add_rows("TEST", split_data["test_df"])
    manifest_df = pd.DataFrame(rows)
    return manifest_df.sort_values(
        by=["split", "label", "patient_id", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_split_stats_dataframe(split_data: dict[str, Any], run_id: str) -> pd.DataFrame:
    """Compute split-level descriptive statistics from in-memory dataframes."""

    rows: list[dict[str, Any]] = []
    for split_name, split_df in (
        ("TRAIN", split_data["train_df"]),
        ("VALIDATION", split_data["val_df"]),
        ("TEST", split_data["test_df"]),
    ):
        if split_df.empty:
            continue
        patient_counts = (
            split_df.groupby(["patient_id", "label"]).size().reset_index(name="n_images_patient")
        )
        per_patient = patient_counts.groupby("patient_id")["n_images_patient"].sum()
        per_class_images = split_df.groupby("label").size().to_dict()
        rows.append(
            {
                "run_id": run_id,
                "split": split_name,
                "n_patients": int(patient_counts["patient_id"].nunique()),
                "n_pos_patients": int(
                    patient_counts[patient_counts["label"] == 1]["patient_id"].nunique()
                ),
                "n_neg_patients": int(
                    patient_counts[patient_counts["label"] == 0]["patient_id"].nunique()
                ),
                "n_images": int(len(split_df)),
                "patches_per_patient_mean": float(per_patient.mean()),
                "patches_per_patient_std": float(per_patient.std(ddof=1))
                if len(per_patient) > 1
                else 0.0,
                "patches_per_patient_min": int(per_patient.min()),
                "patches_per_patient_q1": float(per_patient.quantile(0.25)),
                "patches_per_patient_median": float(per_patient.quantile(0.50)),
                "patches_per_patient_q3": float(per_patient.quantile(0.75)),
                "patches_per_patient_max": int(per_patient.max()),
                "n_images_neg": int(per_class_images.get(0, 0)),
                "n_images_pos": int(per_class_images.get(1, 0)),
            }
        )
    return pd.DataFrame(rows)


def recompute_split_stats_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute split-level descriptive statistics from an HDF5-native manifest."""

    rows: list[dict[str, Any]] = []
    for split_name in ("TRAIN", "VALIDATION", "TEST"):
        split_df = manifest_df.loc[manifest_df["split"] == split_name].copy()
        if split_df.empty:
            continue
        patient_labels = split_df.groupby("patient_id")["label"].max()
        per_patient = split_df.groupby("patient_id").size()
        rows.append(
            {
                "run_id": split_df["run_id"].iloc[0],
                "split": split_name,
                "n_patients": int(split_df["patient_id"].nunique()),
                "n_pos_patients": int(patient_labels.sum()),
                "n_neg_patients": int(len(patient_labels) - patient_labels.sum()),
                "n_images": int(len(split_df)),
                "patches_per_patient_mean": float(per_patient.mean()),
                "patches_per_patient_std": float(per_patient.std(ddof=1))
                if len(per_patient) > 1
                else 0.0,
                "patches_per_patient_min": int(per_patient.min()),
                "patches_per_patient_q1": float(per_patient.quantile(0.25)),
                "patches_per_patient_median": float(per_patient.quantile(0.50)),
                "patches_per_patient_q3": float(per_patient.quantile(0.75)),
                "patches_per_patient_max": int(per_patient.max()),
                "n_images_neg": int((split_df["label"] == 0).sum()),
                "n_images_pos": int((split_df["label"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_manifest_and_log_stats(
    output_dir: Path,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    source_hdf5_path: Path,
    split_data: dict[str, Any],
    manifest_df: pd.DataFrame,
    calc_checksums: bool = True,
    source_hdf5_provenance: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist Stage 5 manifest, split statistics, and run metadata artifacts."""

    logging.info("Writing manifest.csv, split_stats.csv, run_config.json ...")
    if calc_checksums:
        raise ValueError(
            "HDF5-native Stage 5 does not support per-row file checksums. "
            "Set 'CROSSFOLD_CALC_CHECKSUMS=False'."
        )
    manifest_output = manifest_df.copy()
    manifest_path = output_dir / "manifest.csv"
    manifest_output.to_csv(manifest_path, index=False)
    logging.info("Manifest written: %s (rows=%s)", manifest_path, len(manifest_output))

    split_stats_df = build_split_stats_dataframe(split_data, run_id)
    split_stats_path = output_dir / "split_stats.csv"
    split_stats_df.to_csv(split_stats_path, index=False)
    logging.info("Split stats written: %s", split_stats_path)

    run_config = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_hdf5_path": str(source_hdf5_path),
        "output_dir": str(output_dir),
        "normalization_method": normalization_method,
        "is_normalized": bool(is_normalized),
        "constraints": split_data.get("constraints", {}),
        "split_seed": split_data.get("split_seed"),
        "split_attempt": split_data.get("split_attempt"),
        "objective_score": split_data.get("objective_score"),
        "patients": {
            "train": [int(value) for value in split_data.get("train_patients", [])],
            "validation": [int(value) for value in split_data.get("val_patients", [])],
            "test": [int(value) for value in split_data.get("test_patients", [])],
        },
        "source_hdf5_provenance": source_hdf5_provenance
        or collect_hdf5_provenance(source_hdf5_path),
        "library_versions": collect_runtime_environment(),
        "git_commit": get_git_commit_hash(),
        "extra": extra or {},
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    logging.info("Run config written: %s", config_path)
