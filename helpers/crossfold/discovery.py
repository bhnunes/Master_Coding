from __future__ import annotations

import logging
from pathlib import Path

import h5py
import pandas as pd


def load_patch_dataset(source_path: Path) -> pd.DataFrame:
    """Load the Stage 7 source HDF5 dataset into a Stage 5 row index table."""

    if source_path.suffix.lower() != ".h5":
        raise ValueError(f"Stage 5 requires a source .h5 dataset, got: {source_path}")
    return _load_hdf5_patch_dataset(source_path)


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_hdf5_patch_dataset(source_path: Path) -> pd.DataFrame:
    logging.info("Loading Stage 5 data from source HDF5 %s", source_path)
    rows: list[dict[str, object]] = []
    with h5py.File(source_path, "r") as handle:
        filenames = handle["filenames"]
        labels = handle["labels"]
        patient_ids = handle["patient_ids"]
        source_image_paths = handle.get("source_image_paths")
        source_mask_paths = handle.get("source_mask_paths")
        if source_image_paths is None or source_mask_paths is None:
            raise ValueError(
                "Source HDF5 dataset is missing required 'source_image_paths' "
                "or 'source_mask_paths' datasets."
            )
        for index in range(len(filenames)):
            rows.append(
                {
                    "patient_id": int(patient_ids[index]),
                    "image_path": _decode_string(source_image_paths[index]),
                    "mask_path": _decode_string(source_mask_paths[index]),
                    "label": int(labels[index]),
                    "filename": _decode_string(filenames[index]),
                    "source_row_index": index,
                }
            )
    if not rows:
        raise ValueError(f"No rows found in source HDF5 dataset: {source_path}")
    dataset = pd.DataFrame(rows)
    logging.info(
        "Loaded %s HDF5 rows from %s patients.",
        len(dataset),
        dataset["patient_id"].nunique(),
    )
    return dataset
