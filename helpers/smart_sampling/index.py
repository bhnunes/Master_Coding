from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

REQUIRED_DATASETS = ("images", "masks", "patient_ids", "labels")
FILENAME_DATASET_CANDIDATES = ("filenames", "filename")


def guardrail(handle: h5py.File) -> None:
    missing = [key for key in REQUIRED_DATASETS if key not in handle]
    if missing:
        raise KeyError(
            f"Source HDF5 missing required keys: {missing}. Available: {list(handle.keys())}"
        )


def resolve_filename_key(handle: h5py.File) -> str:
    for candidate in FILENAME_DATASET_CANDIDATES:
        if candidate in handle:
            return candidate
    raise KeyError(f"Source HDF5 missing filename dataset. Available: {list(handle.keys())}")


@dataclass(frozen=True)
class H5MetadataIndex:
    h5_path: Path
    patient_map: dict[int, npt.NDArray[np.int64]]
    total_samples: int

    @classmethod
    def build(cls, h5_path: Path) -> H5MetadataIndex:
        with h5py.File(h5_path, "r") as handle:
            guardrail(handle)
            patient_ids = np.asarray(handle["patient_ids"][:], dtype=np.int64)
        dataframe = pd.DataFrame({"patient_id": patient_ids, "idx": np.arange(len(patient_ids))})
        patient_map = {
            int(patient_id): np.asarray(indices, dtype=np.int64)
            for patient_id, indices in dataframe.groupby("patient_id")["idx"]
        }
        return cls(h5_path=h5_path, patient_map=patient_map, total_samples=len(patient_ids))


def load_dataset_shape(h5_path: Path, dataset_name: str) -> tuple[int, ...]:
    with h5py.File(h5_path, "r") as handle:
        dataset = handle[dataset_name]
        return tuple(int(value) for value in dataset.shape)


def read_dataset_rows(
    h5_path: Path,
    dataset_name: str,
    indices: npt.NDArray[np.int64],
) -> Any:
    with h5py.File(h5_path, "r") as handle:
        return handle[dataset_name][indices]
