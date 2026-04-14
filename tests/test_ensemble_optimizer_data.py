from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helpers.ensemble_optimizer import data as optimizer_data


def _write_validation_shards(root: Path) -> optimizer_data.ValidationShardsLayout:
    shard_dir = root / "VALIDATION_shards"
    shard_dir.mkdir(parents=True)
    images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    masks = np.array([np.zeros((4, 4), dtype=np.uint8), np.eye(4, dtype=np.uint8)])
    for patient_id, row_index in (("p1", 0), ("p2", 1)):
        with h5py.File(shard_dir / f"{patient_id}.h5", "w") as handle:
            handle.create_dataset("images", data=images[row_index : row_index + 1])
            handle.create_dataset("masks", data=masks[row_index : row_index + 1])
            handle.create_dataset("labels", data=np.array([row_index], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id.encode()]))
            handle.create_dataset("filenames", data=np.array([f"{patient_id}.png".encode()]))
            handle.attrs["source_hdf5_sha256"] = "stage5-sha"
            handle.attrs["upstream_source_signature"] = "stage2-sig"
            handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "VALIDATION",
                    "patient_id": "p1",
                    "relative_hdf5_path": "VALIDATION_shards/p1.h5",
                    "rows": 1,
                    "label_0_count": 1,
                    "label_1_count": 0,
                },
                {
                    "split": "VALIDATION",
                    "patient_id": "p2",
                    "relative_hdf5_path": "VALIDATION_shards/p2.h5",
                    "rows": 1,
                    "label_0_count": 0,
                    "label_1_count": 1,
                },
            ]
        ),
        shard_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "VALIDATION",
                    "patient_id": "p1",
                    "relative_hdf5_path": "VALIDATION_shards/p1.h5",
                    "row_in_shard": 0,
                    "label": 0,
                    "filename": "p1.png",
                },
                {
                    "split": "VALIDATION",
                    "patient_id": "p2",
                    "relative_hdf5_path": "VALIDATION_shards/p2.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "p2.png",
                },
            ]
        ),
        shard_dir / "sample_manifest.parquet",
    )
    return optimizer_data.ValidationShardsLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


def _write_multirow_validation_shards(root: Path) -> optimizer_data.ValidationShardsLayout:
    shard_dir = root / "VALIDATION_shards"
    shard_dir.mkdir(parents=True)
    with h5py.File(shard_dir / "p1.h5", "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.full((4, 4, 3), 10, dtype=np.uint8),
                    np.full((4, 4, 3), 20, dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [
                    np.zeros((4, 4), dtype=np.uint8),
                    np.eye(4, dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([b"p1", b"p1"], dtype="S16"))
        handle.create_dataset("filenames", data=np.array([b"p1-a.png", b"p1-b.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "stage5-sha"
        handle.attrs["upstream_source_signature"] = "stage2-sig"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    with h5py.File(shard_dir / "p2.h5", "w") as handle:
        handle.create_dataset("images", data=np.full((1, 4, 4, 3), 30, dtype=np.uint8))
        handle.create_dataset("masks", data=np.full((1, 4, 4), 2, dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([b"p2"], dtype="S16"))
        handle.create_dataset("filenames", data=np.array([b"p2-a.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "stage5-sha"
        handle.attrs["upstream_source_signature"] = "stage2-sig"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "VALIDATION",
                    "patient_id": "p1",
                    "relative_hdf5_path": "VALIDATION_shards/p1.h5",
                    "rows": 2,
                    "label_0_count": 1,
                    "label_1_count": 1,
                },
                {
                    "split": "VALIDATION",
                    "patient_id": "p2",
                    "relative_hdf5_path": "VALIDATION_shards/p2.h5",
                    "rows": 1,
                    "label_0_count": 0,
                    "label_1_count": 1,
                },
            ]
        ),
        shard_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "VALIDATION",
                    "patient_id": "p1",
                    "relative_hdf5_path": "VALIDATION_shards/p1.h5",
                    "row_in_shard": 1,
                    "label": 1,
                    "filename": "p1-b.png",
                },
                {
                    "split": "VALIDATION",
                    "patient_id": "p1",
                    "relative_hdf5_path": "VALIDATION_shards/p1.h5",
                    "row_in_shard": 0,
                    "label": 0,
                    "filename": "p1-a.png",
                },
                {
                    "split": "VALIDATION",
                    "patient_id": "p2",
                    "relative_hdf5_path": "VALIDATION_shards/p2.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "p2-a.png",
                },
            ]
        ),
        shard_dir / "sample_manifest.parquet",
    )
    return optimizer_data.ValidationShardsLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


class _ChannelFirstTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(np.moveaxis(mask, -1, 0)),
        }


class _ChannelLastTransform:
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


def test_setup_validation_shards_returns_layout_when_staging_disabled(tmp_path: Path) -> None:
    layout = _write_validation_shards(tmp_path / "source")

    result = optimizer_data.setup_validation_shards(
        tmp_path / "source",
        tmp_path / "local",
        stage_input_locally=False,
    )

    assert result.shard_dir == layout.shard_dir
    assert result.local_cache_dir is None


def test_setup_validation_shards_prepares_local_cache_when_staging_enabled(tmp_path: Path) -> None:
    _write_validation_shards(tmp_path / "source")

    result = optimizer_data.setup_validation_shards(
        tmp_path / "source", tmp_path / "local", stage_input_locally=True
    )

    assert result.local_cache_dir == tmp_path / "local" / "patient_shards"
    assert result.local_cache_dir.exists()


def test_setup_validation_shards_raises_for_missing_shard_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing"):
        optimizer_data.setup_validation_shards(
            source_dir,
            tmp_path / "local",
            stage_input_locally=False,
        )


def test_collect_validation_shard_provenance_reads_shared_lineage(tmp_path: Path) -> None:
    layout = _write_validation_shards(tmp_path / "source")

    provenance = optimizer_data.collect_validation_shard_provenance(layout)

    assert provenance["attrs"]["source_hdf5_sha256"] == "stage5-sha"
    assert provenance["attrs"]["upstream_source_signature"] == "stage2-sig"


def test_decode_patient_id_handles_bytes_and_scalars() -> None:
    assert optimizer_data._decode_patient_id(b"patient-1") == "patient-1"
    assert optimizer_data._decode_patient_id(12) == "12"


def test_validation_hdf5_dataset_builds_two_channel_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(layout)
    image, mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[1])

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (2, 4, 4)
    assert patient_id == "p2"
    assert torch.equal(mask.sum(dim=0), torch.ones((4, 4), dtype=mask.dtype))


def test_validation_hdf5_dataset_can_filter_to_allowed_patients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(layout, allowed_patients={"p2"})

    assert len(dataset) == 1
    _image, _mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[0])
    assert patient_id == "p2"


def test_validation_hdf5_dataset_uses_row_in_shard_for_multirow_patient_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_multirow_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(layout)
    first_image, first_mask, first_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[0]
    )
    second_image, second_mask, second_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[1]
    )
    third_image, third_mask, third_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[2]
    )

    assert int(first_image[0, 0, 0]) == 20
    assert int(second_image[0, 0, 0]) == 10
    assert int(third_image[0, 0, 0]) == 30
    assert first_patient_id == "p1"
    assert second_patient_id == "p1"
    assert third_patient_id == "p2"
    assert float(first_mask[1].sum()) == 4.0
    assert float(second_mask[0].sum()) == 16.0
    assert float(third_mask[1].sum()) == 16.0


def test_validation_hdf5_dataset_permute_branch_handles_hwc_mask_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelLastTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(layout)
    _image, mask, _patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[0])

    assert tuple(mask.shape) == (2, 4, 4)


def test_validation_hdf5_dataset_returns_none_triplet_when_transform_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingTransform:
        def __call__(
            self,
            *,
            image: npt.NDArray[np.generic],
            mask: npt.NDArray[np.generic],
        ) -> dict[str, torch.Tensor]:
            del image, mask
            raise RuntimeError("boom")

    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(optimizer_data, "get_transforms", lambda mode, img_size: FailingTransform())

    dataset = optimizer_data.ValidationHDF5Dataset(layout)

    assert dataset[0] == (None, None, None)


def test_validation_hdf5_dataset_reopens_file_after_pid_change_or_shard_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )
    registered: list[object] = []
    monkeypatch.setattr(optimizer_data.atexit, "register", registered.append)  # type: ignore[attr-defined]
    current_pid = {"value": 101}
    monkeypatch.setattr(optimizer_data.os, "getpid", lambda: current_pid["value"])  # type: ignore[attr-defined]

    dataset = optimizer_data.ValidationHDF5Dataset(layout)
    _ = dataset[0]
    first_handle = dataset.h5_file

    current_pid["value"] = 102
    dataset._open_file("VALIDATION_shards/p2.h5")

    assert first_handle is not dataset.h5_file
    assert dataset._opened_pid == 102
    assert dataset._opened_shard_path is not None
    assert dataset._opened_shard_path.endswith("p2.h5")
    assert len(registered) == 1


def test_validation_hdf5_dataset_state_reset_clears_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(layout)
    _ = dataset[0]

    state = dataset.__getstate__()
    assert state["h5_file"] is None
    assert state["_atexit_registered"] is False

    dataset.__setstate__(state)

    assert dataset.h5_file is None
    assert dataset.masks_dset is None
    assert dataset._opened_pid is None


def test_collate_validation_batch_filters_invalid_items() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.float32)
    mask = torch.zeros((2, 4, 4), dtype=torch.float32)

    batch = optimizer_data.collate_validation_batch([(image, mask, "p1"), (None, None, None), None])

    assert batch is not None
    images, masks, patient_ids = batch
    assert tuple(images.shape) == (1, 3, 4, 4)
    assert tuple(masks.shape) == (1, 2, 4, 4)
    assert patient_ids == ["p1"]


def test_create_validation_dataloader_uses_expected_collate_and_worker_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_validation_shards(tmp_path / "source")
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    loader = optimizer_data.create_validation_dataloader(layout, batch_size=2, workers=0)

    assert loader.batch_size == 2
    assert loader.collate_fn is optimizer_data.collate_validation_batch
    assert loader.worker_init_fn is optimizer_data.worker_init_fn  # type: ignore[attr-defined]


def test_summarize_validation_shards_returns_ordered_patients_and_positive_subset(
    tmp_path: Path,
) -> None:
    layout = _write_validation_shards(tmp_path / "source")

    ordered_patients, positive_patients = optimizer_data.summarize_validation_shards(layout)

    assert ordered_patients == ["p1", "p2"]
    assert positive_patients == {"p2"}
