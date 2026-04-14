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

from helpers.ensemble_inference import data as inference_data


def _write_test_shards(root: Path) -> inference_data.TestShardsLayout:
    shard_dir = root / "TEST_shards"
    shard_dir.mkdir(parents=True)
    images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    masks = np.array([np.zeros((4, 4), dtype=np.uint8), np.tri(4, 4, dtype=np.uint8)])
    for patient_id, row_index in ((1, 0), (2, 1)):
        with h5py.File(shard_dir / f"{patient_id}.h5", "w") as handle:
            handle.create_dataset("images", data=images[row_index : row_index + 1])
            handle.create_dataset("masks", data=masks[row_index : row_index + 1])
            handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id], dtype=np.int32))
            handle.create_dataset("filenames", data=np.array([f"f{patient_id}.png".encode()]))
            handle.attrs["source_hdf5_sha256"] = "stage5-sha"
            handle.attrs["upstream_source_signature"] = "stage2-sig"
            handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "TEST",
                    "patient_id": 1,
                    "relative_hdf5_path": "TEST_shards/1.h5",
                    "rows": 1,
                    "label_0_count": 0,
                    "label_1_count": 1,
                },
                {
                    "split": "TEST",
                    "patient_id": 2,
                    "relative_hdf5_path": "TEST_shards/2.h5",
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
                    "split": "TEST",
                    "patient_id": 1,
                    "relative_hdf5_path": "TEST_shards/1.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "f1.png",
                },
                {
                    "split": "TEST",
                    "patient_id": 2,
                    "relative_hdf5_path": "TEST_shards/2.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "f2.png",
                },
            ]
        ),
        shard_dir / "sample_manifest.parquet",
    )
    return inference_data.TestShardsLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


def _write_multirow_test_shards(root: Path) -> inference_data.TestShardsLayout:
    shard_dir = root / "TEST_shards"
    shard_dir.mkdir(parents=True)
    with h5py.File(shard_dir / "1.h5", "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.full((4, 4, 3), 7, dtype=np.uint8),
                    np.full((4, 4, 3), 8, dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [
                    np.zeros((4, 4), dtype=np.uint8),
                    np.tri(4, 4, dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset("labels", data=np.array([1, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"f1a.png", b"f1b.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "stage5-sha"
        handle.attrs["upstream_source_signature"] = "stage2-sig"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    with h5py.File(shard_dir / "2.h5", "w") as handle:
        handle.create_dataset("images", data=np.full((1, 4, 4, 3), 9, dtype=np.uint8))
        handle.create_dataset("masks", data=np.full((1, 4, 4), 3, dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([2], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"f2a.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "stage5-sha"
        handle.attrs["upstream_source_signature"] = "stage2-sig"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "clean-sha"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "TEST",
                    "patient_id": 1,
                    "relative_hdf5_path": "TEST_shards/1.h5",
                    "rows": 2,
                    "label_0_count": 0,
                    "label_1_count": 2,
                },
                {
                    "split": "TEST",
                    "patient_id": 2,
                    "relative_hdf5_path": "TEST_shards/2.h5",
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
                    "split": "TEST",
                    "patient_id": 1,
                    "relative_hdf5_path": "TEST_shards/1.h5",
                    "row_in_shard": 1,
                    "label": 1,
                    "filename": "f1b.png",
                },
                {
                    "split": "TEST",
                    "patient_id": 1,
                    "relative_hdf5_path": "TEST_shards/1.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "f1a.png",
                },
                {
                    "split": "TEST",
                    "patient_id": 2,
                    "relative_hdf5_path": "TEST_shards/2.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "f2a.png",
                },
            ]
        ),
        shard_dir / "sample_manifest.parquet",
    )
    return inference_data.TestShardsLayout(
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


def test_setup_test_shards_returns_layout_when_staging_disabled(tmp_path: Path) -> None:
    layout = _write_test_shards(tmp_path / "source")

    result = inference_data.setup_test_shards(
        tmp_path / "source",
        tmp_path / "local",
        stage_input_locally=False,
    )

    assert result.shard_dir == layout.shard_dir
    assert result.local_cache_dir is None


def test_setup_test_shards_prepares_local_cache_when_staging_enabled(tmp_path: Path) -> None:
    _write_test_shards(tmp_path / "source")

    result = inference_data.setup_test_shards(
        tmp_path / "source", tmp_path / "local", stage_input_locally=True
    )

    assert result.local_cache_dir == tmp_path / "local" / "patient_shards"
    assert result.local_cache_dir.exists()


def test_setup_test_shards_raises_for_missing_shard_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing"):
        inference_data.setup_test_shards(source_dir, tmp_path / "local", stage_input_locally=False)


def test_decode_patient_id_handles_bytes_and_scalars() -> None:
    assert inference_data._decode_patient_id(b"patient-1") == "patient-1"
    assert inference_data._decode_patient_id(12) == "12"


def test_collect_test_shard_provenance_reads_shared_lineage(tmp_path: Path) -> None:
    layout = _write_test_shards(tmp_path / "source")

    provenance = inference_data.collect_test_shard_provenance(layout)

    assert provenance["attrs"]["source_hdf5_sha256"] == "stage5-sha"
    assert provenance["attrs"]["upstream_source_signature"] == "stage2-sig"


def test_test_hdf5_dataset_reads_item_and_decodes_patient_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_test_shards(tmp_path / "source")
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = inference_data.TestHDF5Dataset(layout)
    image, mask, patient_id, filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str | None], dataset[1]
    )

    assert tuple(image.shape) == (3, 4, 4)
    assert mask.dtype == torch.uint8
    assert tuple(mask.shape) == (4, 4)
    assert patient_id == "2"
    assert filename == "f2.png"


def test_test_hdf5_dataset_uses_row_in_shard_for_multirow_patient_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_multirow_test_shards(tmp_path / "source")
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = inference_data.TestHDF5Dataset(layout)
    first_image, first_mask, first_patient_id, first_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str | None], dataset[0]
    )
    second_image, second_mask, second_patient_id, second_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str | None], dataset[1]
    )
    third_image, third_mask, third_patient_id, third_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str | None], dataset[2]
    )

    assert int(first_image[0, 0, 0]) == 8
    assert int(second_image[0, 0, 0]) == 7
    assert int(third_image[0, 0, 0]) == 9
    assert int(first_mask.sum()) == 10
    assert int(second_mask.sum()) == 0
    assert int(third_mask.sum()) == 16
    assert first_patient_id == "1"
    assert second_patient_id == "1"
    assert third_patient_id == "2"
    assert first_filename == "f1b.png"
    assert second_filename == "f1a.png"
    assert third_filename == "f2a.png"


def test_test_hdf5_dataset_returns_none_triplet_when_transform_fails(
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

    layout = _write_test_shards(tmp_path / "source")
    monkeypatch.setattr(inference_data, "get_transforms", lambda mode, img_size: FailingTransform())

    dataset = inference_data.TestHDF5Dataset(layout)

    assert dataset[0] == (None, None, None, None)


def test_test_hdf5_dataset_reopens_file_after_pid_change_or_shard_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_test_shards(tmp_path / "source")
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )
    registered: list[object] = []
    monkeypatch.setattr(inference_data.atexit, "register", registered.append)  # type: ignore[attr-defined]
    current_pid = {"value": 10}
    monkeypatch.setattr(inference_data.os, "getpid", lambda: current_pid["value"])  # type: ignore[attr-defined]

    dataset = inference_data.TestHDF5Dataset(layout)
    _ = dataset[0]
    first_handle = dataset.h5_file

    current_pid["value"] = 11
    dataset._open_file("TEST_shards/2.h5")

    assert first_handle is not dataset.h5_file
    assert dataset._opened_pid == 11
    assert dataset._opened_shard_path is not None
    assert dataset._opened_shard_path.endswith("2.h5")
    assert len(registered) == 1


def test_test_hdf5_dataset_state_reset_clears_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_test_shards(tmp_path / "source")
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = inference_data.TestHDF5Dataset(layout)
    _ = dataset[0]

    state = dataset.__getstate__()
    assert state["h5_file"] is None
    assert state["_atexit_registered"] is False

    dataset.__setstate__(state)

    assert dataset.h5_file is None
    assert dataset.images_dset is None
    assert dataset._opened_pid is None


def test_collate_test_batch_filters_invalid_items() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.uint8)
    mask = torch.zeros((4, 4), dtype=torch.uint8)

    batch = inference_data.collate_test_batch(
        [(image, mask, "p1", "f1.png"), (None, None, None, None), None]
    )

    assert batch is not None
    images, masks, patient_ids, filenames = batch
    assert tuple(images.shape) == (1, 3, 4, 4)
    assert tuple(masks.shape) == (1, 4, 4)
    assert patient_ids == ["p1"]
    assert filenames == ["f1.png"]


def test_collate_test_batch_returns_none_when_all_items_invalid() -> None:
    assert inference_data.collate_test_batch([(None, None, None, None), None]) is None


def test_create_test_dataloader_sets_worker_dependent_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_test_shards(tmp_path / "source")
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    loader = inference_data.create_test_dataloader(layout, batch_size=2, workers=1)

    assert loader.batch_size == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 4
    assert loader.collate_fn is inference_data.collate_test_batch
    assert loader.worker_init_fn is inference_data.worker_init_fn  # type: ignore[attr-defined]
