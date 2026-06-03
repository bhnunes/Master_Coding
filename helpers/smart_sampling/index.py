from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from helpers.extraction.manifest_paths import resolve_manifest_path_ref


@dataclass(frozen=True)
class CanonicalPatientRow:
    source_row_index: int
    filename: str
    label: int


@dataclass(frozen=True)
class CanonicalPatientReference:
    patient_id: int
    source_hdf5_path: Path
    source_row_indices: npt.NDArray[np.int64]
    rows: tuple[CanonicalPatientRow, ...]


@dataclass(frozen=True)
class MasterManifestIndex:
    master_manifest_path: Path
    patient_map: dict[int, CanonicalPatientReference]
    total_samples: int

    @classmethod
    def build(cls, master_manifest_path: Path) -> MasterManifestIndex:
        patient_rows: dict[int, list[CanonicalPatientRow]] = {}
        patient_paths: dict[int, Path] = {}
        total_samples = 0
        with sqlite3.connect(master_manifest_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT p.patient_id, p.source_hdf5_path, p.source_row_index, p.filename, p.label
                FROM patches p
                INNER JOIN patch_stage_state s ON s.patch_id = p.patch_id
                WHERE s.split = 'TRAIN' AND s.is_stage4_accepted = 1
                ORDER BY p.patient_id ASC, p.source_row_index ASC
                """
            ).fetchall()

        for row in rows:
            patient_id = int(row["patient_id"])
            source_hdf5_path = resolve_manifest_path_ref(
                str(row["source_hdf5_path"]),
                source_root=master_manifest_path.parent,
                manifest_path=master_manifest_path,
            )
            existing_path = patient_paths.get(patient_id)
            if existing_path is None:
                patient_paths[patient_id] = source_hdf5_path
            elif existing_path != source_hdf5_path:
                raise ValueError(
                    "Stage 6 requires one canonical Stage 2 shard per patient, but patient "
                    f"{patient_id} spans multiple source_hdf5_path values: "
                    f"{existing_path} and {source_hdf5_path}."
                )
            patient_rows.setdefault(patient_id, []).append(
                CanonicalPatientRow(
                    source_row_index=int(row["source_row_index"]),
                    filename=str(row["filename"]),
                    label=int(row["label"]),
                )
            )
            total_samples += 1

        patient_map = {
            patient_id: CanonicalPatientReference(
                patient_id=patient_id,
                source_hdf5_path=patient_paths[patient_id],
                source_row_indices=np.asarray(
                    [row.source_row_index for row in rows_for_patient],
                    dtype=np.int64,
                ),
                rows=tuple(rows_for_patient),
            )
            for patient_id, rows_for_patient in patient_rows.items()
        }
        return cls(
            master_manifest_path=master_manifest_path,
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
