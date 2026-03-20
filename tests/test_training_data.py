from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helpers import training_data
from helpers.training_data import (
    ProstateCancerDatasetHDF5,
    SubsetView,
    collate_batch,
    create_stratified_subset_within_patients,
    get_training_hdf5_filename,
    load_artifact_coverage_lookup,
    setup_local_hdf5,
    verify_patient_separation,
)


def _write_hdf5(path: Path, patient_ids: list[bytes] | None = None) -> None:
    if patient_ids is None:
        patient_ids = [b"p1", b"p2", b"p3"]

    images = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
    masks = np.array(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.tri(4, 4, dtype=np.uint8),
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
    (tmp_path / "TRAIN.h5").write_bytes(b"train")
    (tmp_path / "TRAIN_FILTERED.h5").write_bytes(b"filtered")

    chosen = get_training_hdf5_filename(str(tmp_path), smart_sampling=True)

    assert chosen.endswith("TRAIN_FILTERED.h5")


def test_setup_local_hdf5_copies_validation_and_selected_train_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    source_dir.mkdir()
    (source_dir / "VALIDATION.h5").write_bytes(b"validation")
    (source_dir / "TRAIN.h5").write_bytes(b"train")
    (source_dir / "TRAIN_FILTERED.h5").write_bytes(b"filtered")

    chosen_name = setup_local_hdf5(str(source_dir), str(local_dir), smart_sampling=True)

    assert chosen_name == "TRAIN_FILTERED.h5"
    assert (local_dir / "VALIDATION.h5").read_bytes() == b"validation"
    assert (local_dir / "TRAIN_FILTERED.h5").read_bytes() == b"filtered"


def test_prostate_dataset_reads_items_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    _write_hdf5(hdf5_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = ProstateCancerDatasetHDF5(str(hdf5_path), mode="val", subset_indices=[1, 2])
    image, mask = dataset[0]

    assert image is not None
    assert mask is not None
    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert dataset.get_labels().tolist() == [1, 0]
    assert dataset.get_patient_ids().tolist() == [b"p2", b"p3"]


def test_prostate_dataset_returns_artifact_covariates_when_lookup_is_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TRAIN.h5"
    artifact_path = tmp_path / "artifact_patch_index.parquet"
    _write_hdf5(hdf5_path)
    pq.write_table(
        pa.table(
            {
                "filename": ["CANCER_PATIENT_2_0_0_0002.png"],
                "cov_fold": [0.2],
                "cov_penmarking": [0.1],
                "cov_oof": [0.3],
                "cov_darkspot_foreign": [0.0],
                "cov_edge_airbubble": [0.4],
            }
        ),
        artifact_path,
    )
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = ProstateCancerDatasetHDF5(
        str(hdf5_path),
        mode="val",
        subset_indices=[1],
        artifact_coverage_by_filename=load_artifact_coverage_lookup(str(artifact_path)),
    )
    image, mask, artifact_covariates = dataset[0]

    assert image is not None
    assert mask is not None
    assert torch.allclose(artifact_covariates, torch.tensor([0.2, 0.1, 0.3, 0.0, 0.4]))


def test_load_artifact_coverage_lookup_defaults_missing_values_to_zero(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact_patch_index.parquet"
    pq.write_table(pa.table({"filename": ["patch.png"], "cov_fold": [0.5]}), artifact_path)

    lookup = load_artifact_coverage_lookup(str(artifact_path))

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

    assert len(subset_indices) == 5
    assert {0, 1, 2, 5}.issubset(subset_indices)
    assert any(index in subset_indices for index in (3, 4))
