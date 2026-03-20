from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.ensemble_inference import data as inference_data


def _write_test_hdf5(path: Path) -> None:
    images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    masks = np.array(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.tri(4, 4, dtype=np.uint8),
        ]
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("patient_ids", data=np.array([b"p1", b"p2"], dtype="S8"))


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


def test_setup_test_hdf5_returns_source_when_staging_disabled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "TEST.h5"
    source_path.write_bytes(b"test")

    result = inference_data.setup_test_hdf5(
        source_dir,
        tmp_path / "local",
        stage_input_locally=False,
    )

    assert result == source_path


def test_setup_test_hdf5_copies_test_file_when_staging_enabled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    source_dir.mkdir()
    (source_dir / "TEST.h5").write_bytes(b"test-payload")

    result = inference_data.setup_test_hdf5(source_dir, local_dir, stage_input_locally=True)

    assert result == local_dir / "TEST.h5"
    assert result.read_bytes() == b"test-payload"


def test_setup_test_hdf5_raises_for_missing_test_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing"):
        inference_data.setup_test_hdf5(source_dir, tmp_path / "local", stage_input_locally=False)


def test_decode_patient_id_handles_bytes_and_scalars() -> None:
    assert inference_data._decode_patient_id(b"patient-1") == "patient-1"
    assert inference_data._decode_patient_id(12) == "12"


def test_test_hdf5_dataset_reads_item_and_decodes_patient_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TEST.h5"
    _write_test_hdf5(hdf5_path)
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = inference_data.TestHDF5Dataset(hdf5_path)
    image, mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[1])

    assert tuple(image.shape) == (3, 4, 4)
    assert mask.dtype == torch.uint8
    assert tuple(mask.shape) == (4, 4)
    assert patient_id == "p2"


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

    hdf5_path = tmp_path / "TEST.h5"
    _write_test_hdf5(hdf5_path)
    monkeypatch.setattr(inference_data, "get_transforms", lambda mode, img_size: FailingTransform())

    dataset = inference_data.TestHDF5Dataset(hdf5_path)

    assert dataset[0] == (None, None, None)


def test_test_hdf5_dataset_reopens_file_after_pid_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TEST.h5"
    _write_test_hdf5(hdf5_path)
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )
    registered: list[object] = []
    monkeypatch.setattr(inference_data.atexit, "register", registered.append)  # type: ignore[attr-defined]
    current_pid = {"value": 10}
    monkeypatch.setattr(inference_data.os, "getpid", lambda: current_pid["value"])  # type: ignore[attr-defined]

    dataset = inference_data.TestHDF5Dataset(hdf5_path)
    _ = dataset[0]
    first_handle = dataset.h5_file

    current_pid["value"] = 11
    dataset._open_file()

    assert first_handle is not dataset.h5_file
    assert dataset._opened_pid == 11
    assert len(registered) == 1


def test_test_hdf5_dataset_state_reset_clears_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TEST.h5"
    _write_test_hdf5(hdf5_path)
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    dataset = inference_data.TestHDF5Dataset(hdf5_path)
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

    batch = inference_data.collate_test_batch([(image, mask, "p1"), (None, None, None), None])

    assert batch is not None
    images, masks, patient_ids = batch
    assert tuple(images.shape) == (1, 3, 4, 4)
    assert tuple(masks.shape) == (1, 4, 4)
    assert patient_ids == ["p1"]


def test_collate_test_batch_returns_none_when_all_items_invalid() -> None:
    assert inference_data.collate_test_batch([(None, None, None), None]) is None


def test_create_test_dataloader_sets_worker_dependent_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "TEST.h5"
    _write_test_hdf5(hdf5_path)
    monkeypatch.setattr(
        inference_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    loader = inference_data.create_test_dataloader(hdf5_path, batch_size=2, workers=1)

    assert loader.batch_size == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 4
    assert loader.collate_fn is inference_data.collate_test_batch
    assert loader.worker_init_fn is inference_data.worker_init_fn  # type: ignore[attr-defined]
