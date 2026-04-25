from __future__ import annotations

import gc
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref
from helpers.training.canonical_dataset import CanonicalDatasetLayout, CanonicalRowHDF5Dataset
from helpers.training.master_manifest_queries import (
    load_lr_finder_training_records,
    load_test_records,
    load_training_records,
    load_validation_records,
)

PATCH_SIDE = 4
RGB_CHANNELS = 3
TRAIN_ROW_VALUE = 1
MASK_CHANNEL_VALUE = 1
TRAIN_RECORD_COUNT = 4
NORMALIZED_ROW_VALUE = 2


def _write_stage2_shard(
    shard_path: Path,
    *,
    patient_id: int,
    labels: list[int],
    filenames: list[str],
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.full(
                        (PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS),
                        row_index + TRAIN_ROW_VALUE,
                        dtype=np.uint8,
                    )
                    for row_index in range(len(labels))
                ]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [np.full((PATCH_SIDE, PATCH_SIDE), label, dtype=np.uint8) for label in labels]
            ),
        )
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.uint8))
        handle.create_dataset(
            "patient_ids",
            data=np.asarray([patient_id] * len(labels), dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.asarray([filename.encode("utf-8") for filename in filenames]),
        )


def _write_master_manifest(master_manifest_path: Path, shard_paths: list[Path]) -> None:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                slide_id TEXT,
                source_signature TEXT,
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL,
                source_slide_path TEXT NOT NULL,
                annotation_path TEXT,
                artifacts_geojson_path TEXT,
                stage2_case_record_id INTEGER NOT NULL,
                stage2_processing_signature TEXT,
                stage2_status TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_hdf5_path, source_row_index)
            );
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        split_by_patient = {1: "TRAIN", 2: "TRAIN", 3: "VALIDATION", 4: "TEST"}
        selected_rows = {(str(shard_paths[0]), 0), (str(shard_paths[1]), 1)}
        for shard_path in shard_paths:
            patient_id = int(shard_path.stem)
            filenames = [f"p{patient_id}_{row_index}.png" for row_index in range(2)]
            labels = [0, 1]
            for row_index, (filename, label) in enumerate(zip(filenames, labels, strict=True)):
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path,
                        source_row_index,
                        filename,
                        patient_id,
                        label,
                        slide_id,
                        source_signature,
                        source_image_path,
                        source_mask_path,
                        source_slide_path,
                        stage2_case_record_id,
                        stage2_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        to_manifest_path_ref(shard_path, manifest_path=master_manifest_path),
                        row_index,
                        filename,
                        patient_id,
                        label,
                        f"slide_{patient_id}",
                        f"sig_{patient_id}",
                        build_hdf5_dataset_ref("images", row_index),
                        build_hdf5_dataset_ref("masks", row_index),
                        f"/slides/{patient_id}.svs",
                        patient_id,
                        "COMPLETED",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO patch_stage_state (
                        patch_id,
                        split,
                        normalization_method,
                        normalization_artifact_id,
                        sampling_decision,
                        is_stage4_accepted,
                        is_stage7_selected,
                        last_updated_stage_name
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, 'STAGE6')
                    """,
                    (
                        cursor.lastrowid,
                        split_by_patient[patient_id],
                        "NOT_NORMALIZED",
                        None,
                        (
                            "sampled_kept"
                            if (str(shard_path), row_index) in selected_rows
                            else "rejected_reducible"
                        ),
                        1 if (str(shard_path), row_index) in selected_rows else 0,
                    ),
                )
        connection.commit()


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(np.asarray(mask)),
        }


class _RecordingNormalizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def normalize_image(
        self,
        image: npt.NDArray[np.uint8],
        *,
        cache_key: object | None = None,
    ) -> npt.NDArray[np.uint8]:
        self.calls.append(str(cache_key))
        return np.asarray(image + 1, dtype=np.uint8)


class _RecordingTransform:
    def __init__(self) -> None:
        self.images: list[npt.NDArray[np.generic]] = []

    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        self.images.append(np.asarray(image))
        return {
            "image": torch.from_numpy(np.moveaxis(np.asarray(image), -1, 0)),
            "mask": torch.from_numpy(np.asarray(mask)),
        }


@pytest.fixture
def canonical_dataset_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[Path]]:
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _IdentityTransform(),
    )
    shard_paths = [tmp_path / "PATCHES" / f"{patient_id}.h5" for patient_id in (1, 2, 3, 4)]
    for shard_path in shard_paths:
        patient_id = int(shard_path.stem)
        _write_stage2_shard(
            shard_path,
            patient_id=patient_id,
            labels=[0, 1],
            filenames=[f"p{patient_id}_0.png", f"p{patient_id}_1.png"],
        )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_master_manifest(master_manifest_path, shard_paths)
    return master_manifest_path, shard_paths


def test_master_manifest_query_helpers_filter_runtime_rows(
    canonical_dataset_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, shard_paths = canonical_dataset_fixture

    lr_records = load_lr_finder_training_records(master_manifest_path, smart_sampling=True)
    train_records = load_training_records(master_manifest_path, smart_sampling=False)
    validation_records = load_validation_records(master_manifest_path)
    test_records = load_test_records(master_manifest_path)

    assert [(record.patient_id, record.source_row_index) for record in lr_records] == [
        ("1", 0),
        ("2", 1),
    ]
    assert len(train_records) == TRAIN_RECORD_COUNT
    assert all(record.split == "TRAIN" for record in train_records)
    assert validation_records[0].source_hdf5_path == shard_paths[2]
    assert test_records[0].source_hdf5_path == shard_paths[3]


def test_master_manifest_query_helpers_preserve_patient_isolation_across_runtime_splits(
    canonical_dataset_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = canonical_dataset_fixture

    train_patients = {
        record.patient_id
        for record in load_training_records(master_manifest_path, smart_sampling=False)
    }
    validation_patients = {
        record.patient_id for record in load_validation_records(master_manifest_path)
    }
    test_patients = {record.patient_id for record in load_test_records(master_manifest_path)}

    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)


def test_master_manifest_query_helpers_preserve_stage2_row_identity(
    canonical_dataset_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = canonical_dataset_fixture

    records = [
        *load_training_records(master_manifest_path, smart_sampling=False),
        *load_validation_records(master_manifest_path),
        *load_test_records(master_manifest_path),
    ]

    for record in records:
        with h5py.File(record.source_hdf5_path, "r") as handle:
            observed_label = int(handle["labels"][record.source_row_index])
            observed_patient_id = str(handle["patient_ids"][record.source_row_index])
            observed_filename = handle["filenames"][record.source_row_index].decode("utf-8")

        assert observed_label == record.label
        assert observed_patient_id == record.patient_id
        assert observed_filename == record.filename


def test_canonical_dataset_reads_rows_without_sqlite_after_startup(
    canonical_dataset_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = canonical_dataset_fixture
    records = load_training_records(master_manifest_path, smart_sampling=False)
    layout = CanonicalDatasetLayout(records=tuple(records), local_cache_dir=None)
    dataset = CanonicalRowHDF5Dataset(
        layout,
        mode="validation",
        mask_mode="raw",
        include_patient_id=True,
        include_filename=True,
    )

    gc.collect()
    master_manifest_path.unlink()
    image, mask, patient_id, filename = dataset[0]

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert patient_id == "1"
    assert filename == "p1_0.png"


def test_canonical_dataset_reopens_handles_for_new_shards_and_processes(
    canonical_dataset_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_manifest_path, shard_paths = canonical_dataset_fixture
    records = load_training_records(master_manifest_path, smart_sampling=False)
    layout = CanonicalDatasetLayout(records=tuple(records), local_cache_dir=None)
    dataset = CanonicalRowHDF5Dataset(layout, mode="validation", mask_mode="binary")

    _ = dataset[0]
    first_opened_path = dataset._opened_shard_path
    assert first_opened_path == str(shard_paths[0])

    _ = dataset[2]
    assert dataset._opened_shard_path == str(shard_paths[1])
    assert dataset._opened_shard_path != first_opened_path

    original_getpid = os.getpid
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.os.getpid",
        lambda: original_getpid() + 1000,
    )
    dataset._open_file(shard_paths[1])
    assert dataset._opened_pid == original_getpid() + 1000


def test_canonical_dataset_builds_two_channel_masks(
    canonical_dataset_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = canonical_dataset_fixture
    records = load_validation_records(master_manifest_path)
    layout = CanonicalDatasetLayout(records=tuple(records), local_cache_dir=None)
    dataset = CanonicalRowHDF5Dataset(layout, mode="validation", mask_mode="two_channel")

    image, mask = dataset[1]

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (2, 4, 4)


def test_canonical_dataset_applies_stain_normalizer_before_transform(
    canonical_dataset_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_manifest_path, _shard_paths = canonical_dataset_fixture
    records = load_training_records(master_manifest_path, smart_sampling=False)
    layout = CanonicalDatasetLayout(records=tuple(records), local_cache_dir=None)
    recording_transform = _RecordingTransform()
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: recording_transform,
    )
    normalizer = _RecordingNormalizer()
    dataset = CanonicalRowHDF5Dataset(
        layout,
        mode="validation",
        mask_mode="raw",
        image_normalizer=normalizer,
    )

    image, _mask = dataset[0]

    assert normalizer.calls == ["p1_0.png"]
    assert int(recording_transform.images[0][0, 0, 0]) == NORMALIZED_ROW_VALUE
    assert int(image[0, 0, 0]) == NORMALIZED_ROW_VALUE
