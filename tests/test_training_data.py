from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref
from helpers.provenance import hash_file_sha256
from helpers.training import data as training_data
from helpers.training.data import (
    ArtifactAwareDatasetView,
    HybridProstateDataset,
    HybridProstateShardDataset,
    PreparedShardTrainingData,
    PreparedTrainingData,
    ProstateCancerDatasetHDF5,
    ProstateCancerShardDataset,
    ShardDatasetLayout,
    SubsetView,
    _decode_filename,
    collate_batch,
    collect_manifest_split_provenance,
    collect_shard_dataset_provenance,
    create_stratified_subset_within_patients,
    get_training_hdf5_filename,
    load_artifact_coverage_lookup,
    prepare_training_data,
    prepare_training_shard_data,
    setup_local_hdf5,
    verify_patient_separation,
)

PATCH_SIDE = 4
RGB_CHANNELS = 3
HDF5_ROW_COUNT = 3
FILTERED_LABEL_COUNT = 2
SHARD_BATCH_VALUE_OFFSET = 1
SUBSET_INDEX_COUNT = 5
FIRST_TRAIN_PIXEL = 11
SECOND_TRAIN_PIXEL = 44
NORMALIZED_PIXEL = 16
MANIFEST_ROW_COUNT = 2


def _write_hdf5(path: Path, patient_ids: list[bytes] | None = None) -> None:
    if patient_ids is None:
        patient_ids = [b"p1", b"p2", b"p3"]

    images = np.arange(
        HDF5_ROW_COUNT * PATCH_SIDE * PATCH_SIDE * RGB_CHANNELS,
        dtype=np.uint8,
    ).reshape(HDF5_ROW_COUNT, PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS)
    masks = np.array(
        [
            np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
            np.ones((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
            np.tri(PATCH_SIDE, PATCH_SIDE, dtype=np.uint8),
        ]
    )
    labels = np.array([0, 1, 0], dtype=np.uint8)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("patient_ids", data=np.array(patient_ids, dtype="S8"))
        handle.create_dataset(
            "filenames",
            data=np.array(
                [
                    b"NOT_CANCER_PATIENT_1_0_0_0001.png",
                    b"CANCER_PATIENT_2_0_0_0002.png",
                    b"NOT_CANCER_PATIENT_3_0_0_0003.png",
                ],
                dtype="S64",
            ),
        )


def _write_hdf5_with_legacy_filename_dataset(path: Path) -> None:
    images = np.arange(
        HDF5_ROW_COUNT * PATCH_SIDE * PATCH_SIDE * RGB_CHANNELS,
        dtype=np.uint8,
    ).reshape(HDF5_ROW_COUNT, PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS)
    masks = np.zeros((HDF5_ROW_COUNT, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
    labels = np.array([0, 1, 0], dtype=np.uint8)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("patient_ids", data=np.array([b"p1", b"p2", b"p3"], dtype="S8"))
        handle.create_dataset(
            "filename",
            data=np.array(
                [
                    b"NOT_CANCER_PATIENT_1_0_0_0001.png",
                    b"CANCER_PATIENT_2_0_0_0002.png",
                    b"NOT_CANCER_PATIENT_3_0_0_0003.png",
                ],
                dtype="S64",
            ),
        )


def _write_shard_split(root: Path, split_name: str, patient_ids: list[str]) -> ShardDatasetLayout:
    shard_dir = root / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for index, patient_id in enumerate(patient_ids):
        relative_hdf5_path = f"{split_name}/{patient_id}.h5"
        with h5py.File(shard_dir / f"{patient_id}.h5", "w") as handle:
            handle.create_dataset(
                "images",
                data=np.full(
                    (1, PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS),
                    index + SHARD_BATCH_VALUE_OFFSET,
                    dtype=np.uint8,
                ),
            )
            handle.create_dataset(
                "masks",
                data=np.full((1, PATCH_SIDE, PATCH_SIDE), index % 2, dtype=np.uint8),
            )
            handle.create_dataset("labels", data=np.array([index % 2], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id.encode()], dtype="S16"))
            handle.create_dataset(
                "filenames",
                data=np.array([f"{patient_id}.png".encode()], dtype="S32"),
            )
            handle.attrs["source_signature"] = f"{split_name}-sig"
            handle.attrs["source_hdf5_sha256"] = f"{split_name}-sha"
        manifest_rows.append(
            {
                "split": split_name.replace("_shards", ""),
                "patient_id": patient_id,
                "relative_hdf5_path": relative_hdf5_path,
                "rows": 1,
                "label_0_count": 1 if index % 2 == 0 else 0,
                "label_1_count": 1 if index % 2 == 1 else 0,
            }
        )
        sample_rows.append(
            {
                "split": split_name.replace("_shards", ""),
                "patient_id": patient_id,
                "relative_hdf5_path": relative_hdf5_path,
                "row_in_shard": 0,
                "label": index % 2,
                "filename": f"{patient_id}.png",
            }
        )
    pq.write_table(pa.Table.from_pylist(manifest_rows), shard_dir / "manifest.parquet")
    pq.write_table(pa.Table.from_pylist(sample_rows), shard_dir / "sample_manifest.parquet")
    return ShardDatasetLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(mask),
        }


def test_get_training_hdf5_filename_prefers_filtered_file_when_enabled(tmp_path: Path) -> None:
    train_path = tmp_path / "TRAIN.h5"
    train_path.write_bytes(b"train")
    with h5py.File(tmp_path / "TRAIN_FILTERED.h5", "w") as handle:
        handle.attrs["source_hdf5_sha256"] = hash_file_sha256(train_path)
        handle.attrs["selection_signature"] = "sig-123"

    chosen = get_training_hdf5_filename(str(tmp_path), smart_sampling=True)

    assert chosen.endswith("TRAIN_FILTERED.h5")


def test_get_training_hdf5_filename_rejects_stale_filtered_file(tmp_path: Path) -> None:
    train_path = tmp_path / "TRAIN.h5"
    train_path.write_bytes(b"train-v2")
    with h5py.File(tmp_path / "TRAIN_FILTERED.h5", "w") as handle:
        handle.attrs["source_hdf5_sha256"] = "stale-source-sha"
        handle.attrs["selection_signature"] = "sig-123"

    with pytest.raises(ValueError, match="TRAIN_FILTERED.h5"):
        get_training_hdf5_filename(str(tmp_path), smart_sampling=True)


def test_decode_filename_handles_bytes_and_other_values() -> None:
    assert _decode_filename(b"file.png") == "file.png"
    assert _decode_filename(7) == "7"


def test_get_training_hdf5_filename_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing"):
        get_training_hdf5_filename(str(tmp_path / "missing"), smart_sampling=False)


def test_setup_local_hdf5_rejects_missing_source_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    source_dir.mkdir()
    (source_dir / "TRAIN.h5").write_bytes(b"train")

    with pytest.raises(FileNotFoundError, match="Critical data missing"):
        setup_local_hdf5(str(source_dir), str(local_dir), smart_sampling=False)


def test_setup_local_hdf5_copies_validation_and_selected_train_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    source_dir.mkdir()
    (source_dir / "VALIDATION.h5").write_bytes(b"validation")
    train_path = source_dir / "TRAIN.h5"
    train_path.write_bytes(b"train")
    with h5py.File(source_dir / "TRAIN_FILTERED.h5", "w") as handle:
        handle.attrs["source_hdf5_sha256"] = hash_file_sha256(train_path)
        handle.attrs["selection_signature"] = "sig-123"

    chosen_name = setup_local_hdf5(str(source_dir), str(local_dir), smart_sampling=True)

    assert chosen_name == "TRAIN_FILTERED.h5"
    assert (local_dir / "VALIDATION.h5").read_bytes() == b"validation"
    with h5py.File(local_dir / "TRAIN_FILTERED.h5", "r") as handle:
        assert handle.attrs["selection_signature"] == "sig-123"


def test_prostate_dataset_reads_items_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = ProstateCancerDatasetHDF5(str(hdf5_path), mode="val", subset_indices=[1, 2])
    image, mask = cast(tuple[torch.Tensor, torch.Tensor], dataset[0])

    assert image is not None
    assert mask is not None
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert dataset.get_labels().tolist() == [1, 0]
    assert dataset.get_patient_ids().tolist() == [b"p2", b"p3"]
    assert dataset.get_class_counts() == {"CANCER": 1, "NOT_CANCER": 1}


def test_prostate_dataset_returns_artifact_covariates_when_lookup_is_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_hdf5(hdf5_path)
    with sqlite3.connect(master_manifest_path) as connection:
        connection.execute(
            """
            CREATE TABLE patches (
                filename TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        connection.execute(
            "INSERT INTO patches VALUES (?, ?, ?, ?, ?, ?)",
            ("CANCER_PATIENT_2_0_0_0002.png", 0.2, 0.1, 0.3, 0.0, 0.4),
        )
        connection.commit()
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = ProstateCancerDatasetHDF5(
        str(hdf5_path),
        mode="val",
        subset_indices=[1],
        artifact_coverage_by_filename=load_artifact_coverage_lookup(str(master_manifest_path)),
    )
    image, mask, artifact_covariates = dataset[0]

    assert image is not None
    assert mask is not None
    assert torch.allclose(artifact_covariates, torch.tensor([0.2, 0.1, 0.3, 0.0, 0.4]))


def test_load_artifact_coverage_lookup_defaults_missing_values_to_zero(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    with sqlite3.connect(master_manifest_path) as connection:
        connection.execute(
            """
            CREATE TABLE patches (
                filename TEXT NOT NULL,
                cov_fold REAL,
                cov_penmarking REAL,
                cov_oof REAL,
                cov_darkspot_foreign REAL,
                cov_edge_airbubble REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO patches VALUES (?, ?, ?, ?, ?, ?)",
            ("patch.png", 0.5, None, None, None, None),
        )
        connection.commit()

    lookup = load_artifact_coverage_lookup(str(master_manifest_path))

    assert lookup["patch.png"] == (0.5, 0.0, 0.0, 0.0, 0.0)


def test_subset_view_exposes_filtered_labels_and_patient_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )
    dataset = ProstateCancerDatasetHDF5(str(hdf5_path), mode="val")

    subset = SubsetView(dataset, [0, 2])

    assert subset.get_labels() == [0, 0]
    assert subset.get_patient_ids() == [b"p1", b"p3"]


def test_prostate_dataset_returns_none_pair_when_transform_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)

    class FailingTransform:
        def __call__(
            self, *, image: npt.NDArray[np.generic], mask: npt.NDArray[np.generic]
        ) -> dict[str, torch.Tensor]:
            del image, mask
            raise RuntimeError("boom")

    monkeypatch.setattr(training_data, "get_transforms", lambda mode, img_size: FailingTransform())
    dataset = ProstateCancerDatasetHDF5(str(hdf5_path), mode="val")

    assert dataset[0] == (None, None)


def test_prostate_dataset_state_resets_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )
    dataset = ProstateCancerDatasetHDF5(str(hdf5_path), mode="val")
    _ = dataset[0]

    state = dataset.__getstate__()
    assert state["h5_file"] is None
    assert state["_atexit_registered"] is False

    dataset.__setstate__(state)
    assert dataset.h5_file is None
    assert dataset._opened_pid is None


def test_hybrid_dataset_supports_ram_cache_and_class_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    class MemoryInfo:
        available = 10**12

    monkeypatch.setattr(psutil, "virtual_memory", lambda: MemoryInfo())
    dataset = HybridProstateDataset(str(hdf5_path), mode="val", subset_indices=[2, 0])

    image, mask = cast(tuple[torch.Tensor, torch.Tensor], dataset[0])

    assert dataset.use_ram_cache is True
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert dataset.get_class_counts() == {"CANCER": 0, "NOT_CANCER": 2}


def test_hybrid_dataset_opens_file_in_disk_mode_and_returns_artifact_covariates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    class MemoryInfo:
        available = 1

    monkeypatch.setattr(psutil, "virtual_memory", lambda: MemoryInfo())
    dataset = HybridProstateDataset(
        str(hdf5_path),
        mode="val",
        subset_indices=[1],
        artifact_coverage_by_filename={"CANCER_PATIENT_2_0_0_0002.png": (0.1, 0.2, 0.3, 0.4, 0.5)},
    )

    image, mask, artifact_covariates = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], dataset[0]
    )

    assert dataset.use_ram_cache is False
    assert dataset.h5_file is not None
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert torch.allclose(artifact_covariates, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    dataset.close()
    assert dataset.h5_file is None


def test_hybrid_dataset_supports_legacy_singular_filename_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN_FILTERED.h5"
    _write_hdf5_with_legacy_filename_dataset(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    class MemoryInfo:
        available = 1

    monkeypatch.setattr(psutil, "virtual_memory", lambda: MemoryInfo())
    dataset = HybridProstateDataset(
        str(hdf5_path),
        mode="val",
        subset_indices=[1],
        artifact_coverage_by_filename={"CANCER_PATIENT_2_0_0_0002.png": (0.1, 0.2, 0.3, 0.4, 0.5)},
    )

    _, _, artifact_covariates = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], dataset[0])

    assert torch.allclose(artifact_covariates, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    dataset.close()


def test_hybrid_dataset_raises_runtime_error_when_transform_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)

    class MemoryInfo:
        available = 1

    class FailingTransform:
        def __call__(
            self, *, image: npt.NDArray[np.generic], mask: npt.NDArray[np.generic]
        ) -> dict[str, torch.Tensor]:
            del image, mask
            raise RuntimeError("boom")

    monkeypatch.setattr(psutil, "virtual_memory", lambda: MemoryInfo())
    monkeypatch.setattr(training_data, "get_transforms", lambda mode, img_size: FailingTransform())
    dataset = HybridProstateDataset(str(hdf5_path), mode="val")

    with pytest.raises(RuntimeError, match="Transform failed"):
        _ = dataset[0]


def test_collate_batch_filters_invalid_entries() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.uint8)
    mask = torch.zeros((4, 4), dtype=torch.long)

    batch = collate_batch([(image, mask), None, (None, mask), (image, None)])

    assert batch is not None
    images, masks = batch
    assert images.shape == (1, 3, 4, 4)
    assert masks.shape == (1, 4, 4)


def test_verify_patient_separation_rejects_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "TRAIN.h5"
    val_path = tmp_path / "VALIDATION.h5"
    _write_hdf5(train_path, patient_ids=[b"p1", b"p2", b"p3"])
    _write_hdf5(val_path, patient_ids=[b"p3", b"p4", b"p5"])
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    train_dataset = ProstateCancerDatasetHDF5(str(train_path), mode="train")
    val_dataset = ProstateCancerDatasetHDF5(str(val_path), mode="val")

    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE"):
        verify_patient_separation(train_dataset, val_dataset)


class _SubsetDataset:
    def __init__(self) -> None:
        self.labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
        self.patient_ids = np.array([b"p1", b"p1", b"p2", b"p2", b"p2", b"p3"])

    def __len__(self) -> int:
        return len(self.labels)


def test_create_stratified_subset_within_patients_preserves_patient_class_groups() -> None:
    dataset = _SubsetDataset()

    subset = create_stratified_subset_within_patients(
        dataset, ratio=0.5, split_name="train", seed=7
    )

    subset_indices = sorted(subset.indices)

    assert len(subset_indices) == SUBSET_INDEX_COUNT
    assert {0, 1, 2, 5}.issubset(subset_indices)
    assert any(index in subset_indices for index in (3, 4))


def test_prepare_training_shard_data_prefers_filtered_train_shards(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_shard_split(data_root, "TRAIN_FILTERED_shards", ["p1"])
    _write_shard_split(data_root, "VALIDATION_shards", ["v1"])

    prepared = prepare_training_shard_data(data_root, tmp_path / "local", smart_sampling=True)

    assert isinstance(prepared, PreparedShardTrainingData)
    assert prepared.source_split_name == "TRAIN_FILTERED_shards"
    assert prepared.train_layout.shard_dir == data_root / "TRAIN_FILTERED_shards"


def test_collect_shard_dataset_provenance_hashes_manifests(tmp_path: Path) -> None:
    layout = _write_shard_split(tmp_path / "data", "TRAIN_shards", ["p1", "p2"])

    provenance = collect_shard_dataset_provenance(layout)

    assert provenance["path"].endswith("sample_manifest.parquet")
    assert provenance["manifest_sha256"] == hash_file_sha256(layout.manifest_path)
    assert provenance["source_signature"] == "TRAIN_shards-sig"


def test_collect_shard_dataset_provenance_includes_filtered_selection_metadata(
    tmp_path: Path,
) -> None:
    layout = _write_shard_split(tmp_path / "data", "TRAIN_FILTERED_shards", ["p1", "p2"])
    for shard_path in layout.shard_dir.glob("*.h5"):
        with h5py.File(shard_path, "a") as handle:
            handle.attrs["stage7_label_aware"] = True
            handle.attrs["stage7_holdout_mode"] = "within_patient_patch_holdout"
            handle.attrs["stage7_selected_count"] = int(handle["labels"].shape[0])
    (layout.shard_dir / "summary.json").write_text(
        '{"selection_signature": "selection-123"}', encoding="utf-8"
    )

    provenance = collect_shard_dataset_provenance(layout)

    assert provenance["selection_signature"] == "selection-123"
    assert provenance["smart_sampling_enabled"] is True
    assert provenance["source_signature"] == "TRAIN_FILTERED_shards-sig"
    assert provenance["summary_path"] is not None
    assert provenance["summary_sha256"] == hash_file_sha256(layout.shard_dir / "summary.json")
    assert provenance["smart_sampling_metadata"] == {
        "stage7_holdout_mode": "within_patient_patch_holdout",
        "stage7_label_aware": True,
        "stage7_selected_count": 1,
    }


def test_prostate_shard_dataset_reads_items_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_shard_split(tmp_path / "data", "VALIDATION_shards", ["p1", "p2"])
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = ProstateCancerShardDataset(layout, mode="val", subset_indices=[1])
    image, mask = cast(tuple[torch.Tensor, torch.Tensor], dataset[0])

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert dataset.get_labels().tolist() == [1]
    assert dataset.get_patient_ids().tolist() == ["p2"]


def test_hybrid_shard_dataset_supports_ram_cache_and_artifact_covariates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_shard_split(tmp_path / "data", "TRAIN_shards", ["p1"])
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    class MemoryInfo:
        available = 10**12

    monkeypatch.setattr(psutil, "virtual_memory", lambda: MemoryInfo())
    dataset = HybridProstateShardDataset(
        layout,
        mode="train",
        artifact_coverage_by_filename={"p1.png": (0.1, 0.2, 0.3, 0.4, 0.5)},
    )

    image, mask, artifact_covariates = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], dataset[0]
    )

    assert dataset.use_ram_cache is True
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert torch.allclose(artifact_covariates, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))


def _write_stage2_manifest_shard(
    shard_path: Path,
    *,
    patient_id: int,
    pixel_values: list[int],
    labels: list[int],
    filenames: list[str],
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [np.full((4, 4, 3), pixel_value, dtype=np.uint8) for pixel_value in pixel_values]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack([np.full((4, 4), label, dtype=np.uint8) for label in labels]),
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


def _write_training_master_manifest(master_manifest_path: Path, shard_paths: list[Path]) -> None:
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
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            """
        )
        selected_rows = {(str(shard_paths[0]), 0), (str(shard_paths[1]), 1)}
        split_by_patient = {1: "TRAIN", 2: "TRAIN", 3: "VALIDATION", 4: "VALIDATION"}
        for shard_path in shard_paths:
            patient_id = int(shard_path.stem.replace("patient_", ""))
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
                        stage2_status,
                        cov_fold,
                        cov_penmarking,
                        cov_oof,
                        cov_darkspot_foreign,
                        cov_edge_airbubble
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        0.2 if filename == "p1_0.png" else 0.0,
                        0.1 if filename == "p1_0.png" else 0.0,
                        0.3 if filename == "p1_0.png" else 0.0,
                        0.4 if filename == "p1_0.png" else 0.0,
                        0.5 if filename == "p1_0.png" else 0.0,
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


@pytest.fixture
def training_manifest_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[Path]]:
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _IdentityTransform(),
    )
    shard_paths = [tmp_path / "PATCHES" / f"patient_{patient_id}.h5" for patient_id in (1, 2, 3, 4)]
    _write_stage2_manifest_shard(
        shard_paths[0],
        patient_id=1,
        pixel_values=[11, 22],
        labels=[0, 1],
        filenames=["p1_0.png", "p1_1.png"],
    )
    _write_stage2_manifest_shard(
        shard_paths[1],
        patient_id=2,
        pixel_values=[33, 44],
        labels=[0, 1],
        filenames=["p2_0.png", "p2_1.png"],
    )
    _write_stage2_manifest_shard(
        shard_paths[2],
        patient_id=3,
        pixel_values=[55, 66],
        labels=[0, 1],
        filenames=["p3_0.png", "p3_1.png"],
    )
    _write_stage2_manifest_shard(
        shard_paths[3],
        patient_id=4,
        pixel_values=[77, 88],
        labels=[0, 1],
        filenames=["p4_0.png", "p4_1.png"],
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_training_master_manifest(master_manifest_path, shard_paths)
    return master_manifest_path, shard_paths


def test_prepare_training_data_uses_sqlite_rows_and_manifest_provenance(
    training_manifest_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture

    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=True,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=False,
    )

    assert isinstance(prepared, PreparedTrainingData)
    assert prepared.source_split_name == "TRAIN_SELECTED"
    train_dataset = cast(Any, prepared.train_dataset)
    validation_dataset = cast(Any, prepared.validation_dataset)
    assert train_dataset.get_labels().tolist() == [0, 1]
    assert train_dataset.get_patient_ids().tolist() == ["1", "2"]
    assert validation_dataset.get_patient_ids().tolist() == ["3", "3", "4", "4"]
    assert prepared.training_provenance["split"] == "TRAIN"
    assert prepared.training_provenance["smart_sampling"] is True
    assert prepared.validation_provenance["split"] == "VALIDATION"


def test_prepare_training_data_reads_canonical_stage2_rows(
    training_manifest_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture

    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=True,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=False,
    )
    first_image, first_mask = prepared.train_dataset[0]
    second_image, second_mask = prepared.train_dataset[1]

    assert int(first_image[0, 0, 0]) == FIRST_TRAIN_PIXEL
    assert int(second_image[0, 0, 0]) == SECOND_TRAIN_PIXEL
    assert int(first_mask[0, 0]) == 0
    assert int(second_mask[0, 0]) == 1


def test_prepare_training_data_preserves_patient_separation(
    training_manifest_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture
    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=False,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=False,
    )

    verify_patient_separation(prepared.train_dataset, prepared.validation_dataset)


def test_prepare_training_data_uses_shared_stain_normalizer(
    training_manifest_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture
    normalizer = type(
        "RecordingNormalizer",
        (),
        {
            "__init__": lambda self: setattr(self, "calls", []),
            "normalize_image": lambda self, image, cache_key=None: (
                self.calls.append(str(cache_key)) or np.asarray(image + 5, dtype=np.uint8)
            ),
        },
    )()
    monkeypatch.setattr(
        training_data, "build_split_stain_normalizer", lambda *args, **kwargs: normalizer
    )

    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=True,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=False,
    )
    image, _mask = prepared.train_dataset[0]

    assert normalizer.calls[0] == "p1_0.png"
    assert int(image[0, 0, 0]) == NORMALIZED_PIXEL


def test_prepare_training_data_wraps_artifact_aware_loss_features(
    training_manifest_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture

    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=True,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=True,
    )
    image, mask, artifact_covariates = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], prepared.train_dataset[0]
    )

    assert isinstance(prepared.train_dataset, ArtifactAwareDatasetView)
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert torch.allclose(artifact_covariates, torch.tensor([0.2, 0.1, 0.3, 0.4, 0.5]))


def test_prepare_training_data_builds_subset_and_weights(
    training_manifest_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture

    def fake_subset(dataset: object, ratio: float, split_name: str, seed: int) -> object:
        del dataset, ratio, split_name, seed
        return type("FakeSubset", (), {"indices": [1]})()

    monkeypatch.setattr(training_data, "create_stratified_subset_within_patients", fake_subset)

    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=False,
        use_subset=True,
        subset_ratio=0.5,
        seed=24,
        use_artifact_aware_loss=False,
    )

    train_dataset = cast(Any, prepared.train_dataset)
    validation_dataset = cast(Any, prepared.validation_dataset)
    assert train_dataset.get_labels().tolist() == [1]
    assert validation_dataset.get_labels().tolist() == [1]
    assert prepared.sample_weights.tolist() == [1.0]


def test_collect_manifest_split_provenance_reports_manifest_metadata(
    training_manifest_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = training_manifest_fixture
    prepared = prepare_training_data(
        master_manifest_path=master_manifest_path,
        local_data_dir=master_manifest_path.parent / "local",
        smart_sampling=True,
        use_subset=False,
        subset_ratio=1.0,
        seed=24,
        use_artifact_aware_loss=False,
    )
    provenance = collect_manifest_split_provenance(
        master_manifest_path,
        records=cast(Any, prepared.train_dataset).records,
        split="TRAIN",
        smart_sampling=True,
    )

    assert provenance["master_manifest_path"] == str(master_manifest_path)
    assert provenance["row_count"] == MANIFEST_ROW_COUNT
    assert provenance["selection_mode"] == "stage7_selected"
