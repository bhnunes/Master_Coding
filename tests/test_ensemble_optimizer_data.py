from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.ensemble_optimizer import data as optimizer_data


def _write_validation_hdf5(path: Path) -> None:
    images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    masks = np.array(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.eye(4, dtype=np.uint8),
        ]
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("patient_ids", data=np.array([b"p1", b"p2"], dtype="S8"))


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


def test_setup_validation_hdf5_returns_source_when_staging_disabled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "VALIDATION.h5"
    source_path.write_bytes(b"validation")

    result = optimizer_data.setup_validation_hdf5(
        source_dir,
        tmp_path / "local",
        stage_input_locally=False,
    )

    assert result == source_path


def test_setup_validation_hdf5_copies_validation_file_when_staging_enabled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    source_dir.mkdir()
    (source_dir / "VALIDATION.h5").write_bytes(b"validation-payload")

    result = optimizer_data.setup_validation_hdf5(source_dir, local_dir, stage_input_locally=True)

    assert result == local_dir / "VALIDATION.h5"
    assert result.read_bytes() == b"validation-payload"


def test_setup_validation_hdf5_raises_for_missing_validation_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing"):
        optimizer_data.setup_validation_hdf5(
            source_dir,
            tmp_path / "local",
            stage_input_locally=False,
        )


def test_validation_hdf5_dataset_builds_two_channel_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path)
    image, mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[1])

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (2, 4, 4)
    assert patient_id == "p2"
    assert torch.equal(mask.sum(dim=0), torch.ones((4, 4), dtype=mask.dtype))


def test_validation_hdf5_dataset_can_filter_to_allowed_patients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path, allowed_patients={"p2"})

    assert len(dataset) == 1
    _image, _mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[0])
    assert patient_id == "p2"


def test_validation_hdf5_dataset_permute_branch_handles_hwc_mask_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelLastTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path)
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

    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(optimizer_data, "get_transforms", lambda mode, img_size: FailingTransform())

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path)

    assert dataset[0] == (None, None, None)


def test_validation_hdf5_dataset_reopens_file_after_pid_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )
    registered: list[object] = []
    monkeypatch.setattr(optimizer_data.atexit, "register", registered.append)  # type: ignore[attr-defined]
    current_pid = {"value": 101}
    monkeypatch.setattr(optimizer_data.os, "getpid", lambda: current_pid["value"])  # type: ignore[attr-defined]

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path)
    _ = dataset[0]
    first_handle = dataset.h5_file

    current_pid["value"] = 102
    dataset._open_file()

    assert first_handle is not dataset.h5_file
    assert dataset._opened_pid == 102
    assert len(registered) == 1


def test_validation_hdf5_dataset_state_reset_clears_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    dataset = optimizer_data.ValidationHDF5Dataset(hdf5_path)
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
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)
    monkeypatch.setattr(
        optimizer_data, "get_transforms", lambda mode, img_size: _ChannelFirstTransform()
    )

    loader = optimizer_data.create_validation_dataloader(hdf5_path, batch_size=2, workers=0)

    assert loader.batch_size == 2
    assert loader.collate_fn is optimizer_data.collate_validation_batch
    assert loader.worker_init_fn is optimizer_data.worker_init_fn  # type: ignore[attr-defined]


def test_summarize_validation_hdf5_returns_ordered_patients_and_positive_subset(
    tmp_path: Path,
) -> None:
    hdf5_path = tmp_path / "VALIDATION.h5"
    _write_validation_hdf5(hdf5_path)

    ordered_patients, positive_patients = optimizer_data.summarize_validation_hdf5(hdf5_path)

    assert ordered_patients == ["p1", "p2"]
    assert positive_patients == {"p2"}
