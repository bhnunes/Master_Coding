from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import h5py
import pandas as pd

from helpers.extraction.manifest_paths import resolve_manifest_path_ref, to_runtime_hdf5_ref
from helpers.provenance import hash_file_sha256


def load_patch_dataset(source_path: Path) -> pd.DataFrame:
    """Load the Stage 5 source dataset into a row-indexed dataframe."""

    suffix = source_path.suffix.lower()
    if suffix == ".h5":
        return _load_hdf5_patch_dataset(source_path)
    if suffix == ".sqlite":
        return _load_sqlite_patch_dataset(source_path)
    raise ValueError(f"Stage 5 requires a source .h5 or .sqlite dataset, got: {source_path}")


def collect_source_dataset_provenance(source_path: Path) -> dict[str, object]:
    """Collect Stage 5 source provenance from HDF5 or SQLite."""

    if source_path.suffix.lower() == ".h5":
        from helpers.provenance import collect_hdf5_provenance

        return collect_hdf5_provenance(source_path)
    if source_path.suffix.lower() != ".sqlite":
        raise ValueError(f"Unsupported Stage 5 source dataset: {source_path}")

    source_sha256 = hash_file_sha256(source_path)
    with sqlite3.connect(source_path) as connection:
        accepted_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM patch_stage_state WHERE is_stage4_accepted = 1"
            ).fetchone()[0]
        )
        signatures = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT source_signature FROM patches WHERE source_signature IS NOT NULL "
                "ORDER BY source_signature ASC"
            ).fetchall()
        ]
    return {
        "path": str(source_path),
        "sha256": source_sha256,
        "source_signature": None,
        "attrs": {
            "stage4_cleaning_manifest_path": str(source_path),
            "stage4_cleaning_manifest_sha256": source_sha256,
            "stage4_cleaning_selected_rows": accepted_rows,
            "upstream_source_signature": ",".join(signatures) if signatures else None,
        },
    }


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_logical_hdf5_ref(source_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{source_path}::{dataset_name}[{row_index}]"


def _decode_string_array(values: Sequence[object]) -> list[str]:
    return [_decode_string(value) for value in values]


def _load_hdf5_patch_dataset(source_path: Path) -> pd.DataFrame:
    logging.info("Loading Stage 5 source rows from HDF5 %s", source_path)
    with h5py.File(source_path, "r") as handle:
        filenames = _decode_string_array(handle["filenames"][:])
        labels = handle["labels"][:]
        patient_ids = handle["patient_ids"][:]
        source_image_paths = handle.get("source_image_paths")
        source_mask_paths = handle.get("source_mask_paths")
        row_count = len(filenames)
        if row_count == 0:
            raise ValueError(f"No rows found in source HDF5 dataset: {source_path}")
        source_row_index = list(range(row_count))
        image_paths = (
            _decode_string_array(source_image_paths[:])
            if source_image_paths is not None
            else [
                _build_logical_hdf5_ref(source_path, "images", index) for index in source_row_index
            ]
        )
        mask_paths = (
            _decode_string_array(source_mask_paths[:])
            if source_mask_paths is not None
            else [
                _build_logical_hdf5_ref(source_path, "masks", index) for index in source_row_index
            ]
        )
    dataset = pd.DataFrame(
        {
            "patient_id": patient_ids.astype(int),
            "image_path": image_paths,
            "mask_path": mask_paths,
            "label": labels.astype(int),
            "filename": filenames,
            "source_row_index": source_row_index,
            "source_hdf5_path": [str(source_path)] * row_count,
        }
    )
    logging.info(
        "Loaded %s HDF5 rows from %s patients.",
        len(dataset),
        dataset["patient_id"].nunique(),
    )
    return dataset


def _load_sqlite_patch_dataset(source_path: Path) -> pd.DataFrame:
    logging.info("Loading Stage 5 accepted rows from SQLite %s", source_path)
    with sqlite3.connect(source_path) as connection:
        dataset = pd.read_sql_query(
            """
            SELECT
                p.patient_id,
                p.source_image_path AS image_path,
                p.source_mask_path AS mask_path,
                p.label,
                p.filename,
                p.source_row_index,
                p.source_hdf5_path
            FROM patches p
            INNER JOIN patch_stage_state s ON s.patch_id = p.patch_id
            WHERE s.is_stage4_accepted = 1
            ORDER BY p.patient_id ASC, p.filename ASC
            """,
            connection,
        )
    if dataset.empty:
        raise ValueError(f"No accepted Stage 3.3 rows found in master manifest: {source_path}")
    dataset["source_hdf5_path"] = dataset["source_hdf5_path"].map(
        lambda value: str(
            resolve_manifest_path_ref(
                str(value),
                source_root=source_path.parent,
                manifest_path=source_path,
            )
        )
    )
    dataset["image_path"] = dataset.apply(
        lambda row: to_runtime_hdf5_ref(
            Path(str(row["source_hdf5_path"])),
            str(row["image_path"]),
            expected_dataset="images",
        ),
        axis=1,
    )
    dataset["mask_path"] = dataset.apply(
        lambda row: to_runtime_hdf5_ref(
            Path(str(row["source_hdf5_path"])),
            str(row["mask_path"]),
            expected_dataset="masks",
        ),
        axis=1,
    )
    logging.info(
        "Loaded %s SQLite-backed rows from %s patients.",
        len(dataset),
        dataset["patient_id"].nunique(),
    )
    return dataset
