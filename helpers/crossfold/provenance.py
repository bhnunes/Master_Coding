from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 checksum for one file path."""

    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fill_manifest_checksums_inplace(
    manifest_df: pd.DataFrame,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Populate SHA256 columns from absolute destination paths in the manifest."""

    def hash_pair(image_path: str, mask_path: str) -> tuple[str | None, str | None]:
        image_hash = sha256_file(image_path) if Path(image_path).is_file() else None
        mask_hash = sha256_file(mask_path) if Path(mask_path).is_file() else None
        return image_hash, mask_hash

    image_paths = manifest_df["abs_image_path"].tolist()
    mask_paths = manifest_df["abs_mask_path"].tolist()
    output_image_hashes: list[str | None] = [None] * len(manifest_df)
    output_mask_hashes: list[str | None] = [None] * len(manifest_df)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(hash_pair, image_paths[index], mask_paths[index]): index
            for index in range(len(manifest_df))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="SHA256", leave=False):
            index = futures[future]
            try:
                image_hash, mask_hash = future.result()
                output_image_hashes[index] = image_hash
                output_mask_hashes[index] = mask_hash
            except Exception:
                continue
    manifest_df["sha256_image"] = output_image_hashes
    manifest_df["sha256_mask"] = output_mask_hashes
    return manifest_df


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, if available."""

    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return result.strip() or None
    except Exception:
        return None


def collect_library_versions() -> dict[str, Any]:
    """Collect a compact library-version snapshot for run provenance."""

    versions: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "opencv": getattr(cv2, "__version__", None),
    }
    try:
        import sklearn

        versions["sklearn"] = getattr(sklearn, "__version__", None)
    except Exception:
        versions["sklearn"] = None
    try:
        import tiatoolbox

        versions["tiatoolbox"] = getattr(tiatoolbox, "__version__", None)
    except Exception:
        versions["tiatoolbox"] = None
    return versions


def build_manifest_from_split_dfs(
    output_dir: Path,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    split_data: dict[str, Any],
) -> pd.DataFrame:
    """Build a deterministic manifest from in-memory split dataframes."""

    rows: list[dict[str, Any]] = []

    def add_rows(split_name: str, split_df: pd.DataFrame) -> None:
        if split_df.empty:
            return
        for row in split_df.itertuples(index=False):
            label_dir = "CANCER" if int(row.label) == 1 else "NOT_CANCER"
            relative_image = f"{split_name}/{label_dir}/{row.filename}"
            relative_mask = f"{split_name}/{label_dir}_MASK/{row.filename}"
            rows.append(
                {
                    "run_id": run_id,
                    "split": split_name,
                    "label": int(row.label),
                    "patient_id": int(row.patient_id),
                    "filename": row.filename,
                    "normalization_method": normalization_method,
                    "is_normalized": bool(is_normalized),
                    "relative_path_image": relative_image,
                    "relative_path_mask": relative_mask,
                    "source_image_path": row.image_path,
                    "source_mask_path": row.mask_path,
                    "sha256_image": None,
                    "sha256_mask": None,
                    "abs_image_path": str(output_dir / split_name / label_dir / row.filename),
                    "abs_mask_path": str(
                        output_dir / split_name / f"{label_dir}_MASK" / row.filename
                    ),
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


def build_hdf5_manifest_from_split_dfs(
    output_dir: Path,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    split_data: dict[str, Any],
) -> pd.DataFrame:
    """Build a deterministic manifest for HDF5-native split outputs."""

    rows: list[dict[str, Any]] = []

    def add_rows(split_name: str, split_df: pd.DataFrame) -> None:
        if split_df.empty:
            return
        relative_hdf5_path = f"{split_name}.h5"
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
                    "relative_hdf5_path": relative_hdf5_path,
                    "hdf5_row_index": row_index,
                    "source_row_index": int(getattr(row, "source_row_index", row_index)),
                    "source_image_path": getattr(row, "image_path", ""),
                    "source_mask_path": getattr(row, "mask_path", ""),
                    "abs_hdf5_path": str(output_dir / relative_hdf5_path),
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


def write_manifest_and_log_stats(
    output_dir: Path,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    data_directory: Path,
    split_data: dict[str, Any],
    manifest_df: pd.DataFrame,
    calc_checksums: bool = True,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist Stage 5 manifest, split statistics, and run metadata artifacts."""

    logging.info("Writing manifest.csv, split_stats.csv, run_config.json ...")
    if calc_checksums:
        logging.warning("calc_checksums=True: computing SHA256 for all files (can take hours).")
        manifest_df = fill_manifest_checksums_inplace(
            manifest_df,
            num_workers=min(16, (__import__("os").cpu_count() or 1)),
        )
    manifest_output = manifest_df.drop(columns=["abs_image_path", "abs_mask_path"], errors="ignore")
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
        "data_directory": str(data_directory),
        "output_dir": str(output_dir),
        "normalization_method": normalization_method,
        "is_normalized": bool(is_normalized),
        "constraints": split_data.get("constraints", {}),
        "split_seed": split_data.get("split_seed"),
        "split_attempt": split_data.get("split_attempt"),
        "objective_score": split_data.get("objective_score"),
        "objective_score_split": split_data.get("objective_score_split"),
        "patients": {
            "train": [int(value) for value in split_data.get("train_patients", [])],
            "validation": [int(value) for value in split_data.get("val_patients", [])],
            "test": [int(value) for value in split_data.get("test_patients", [])],
        },
        "library_versions": collect_library_versions(),
        "git_commit": get_git_commit_hash(),
        "extra": extra or {},
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    logging.info("Run config written: %s", config_path)
