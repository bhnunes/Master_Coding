from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

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


@dataclass(frozen=True)
class PatientShardReference:
    patient_id: int
    shard_path: Path
    relative_hdf5_path: str
    row_count: int


@dataclass(frozen=True)
class ShardManifestIndex:
    shard_dir: Path
    manifest_path: Path
    patient_map: dict[int, PatientShardReference]
    total_samples: int

    @classmethod
    def build(cls, shard_dir: Path, manifest_path: Path) -> ShardManifestIndex:
        table = pq.read_table(manifest_path)
        records = table.to_pylist()
        patient_map: dict[int, PatientShardReference] = {}
        total_samples = 0
        for record in records:
            patient_id = int(record["patient_id"])
            relative_hdf5_path = str(record["relative_hdf5_path"])
            row_count = int(record["rows"])
            shard_path = shard_dir.parent / relative_hdf5_path
            if patient_id in patient_map:
                raise ValueError(
                    "Duplicate patient_id "
                    f"{patient_id} found in Stage 7 shard manifest {manifest_path}."
                )
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Stage 7 shard manifest references missing shard: {shard_path}"
                )
            patient_map[patient_id] = PatientShardReference(
                patient_id=patient_id,
                shard_path=shard_path,
                relative_hdf5_path=relative_hdf5_path,
                row_count=row_count,
            )
            total_samples += row_count
        return cls(
            shard_dir=shard_dir,
            manifest_path=manifest_path,
            patient_map=patient_map,
            total_samples=total_samples,
        )


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
