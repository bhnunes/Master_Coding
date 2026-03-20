from __future__ import annotations

import atexit
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from helpers.training.data import get_transforms
from helpers.training.runtime import worker_init_fn


def setup_validation_hdf5(
    hdf5_drive_dir: Path,
    local_data_dir: Path,
    *,
    stage_input_locally: bool,
) -> Path:
    source_path = hdf5_drive_dir / "VALIDATION.h5"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}")
    if not stage_input_locally:
        return source_path

    if local_data_dir.exists():
        shutil.rmtree(local_data_dir)
    local_data_dir.mkdir(parents=True, exist_ok=True)
    destination_path = local_data_dir / "VALIDATION.h5"
    shutil.copy2(source_path, destination_path)
    return destination_path


class ValidationHDF5Dataset(Dataset[Any]):
    def __init__(self, hdf5_path: Path) -> None:
        self.hdf5_path = str(hdf5_path)
        self.transform = get_transforms(mode="validation", img_size=224)
        with h5py.File(self.hdf5_path, "r") as handle:
            self.full_pids = np.asarray(handle["patient_ids"][:])
            self.total_len = len(self.full_pids)
        self.h5_file: Any = None
        self.images_dset: Any = None
        self.masks_dset: Any = None
        self._opened_pid: int | None = None
        self._atexit_registered = False

    def _open_file(self) -> None:
        pid = os.getpid()
        if self.h5_file is not None and self._opened_pid is not None and self._opened_pid != pid:
            self.close()
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.hdf5_path, "r", libver="latest", rdcc_nbytes=50 * 1024 * 1024
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self._opened_pid = pid
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str] | tuple[None, None, None]:
        if self.h5_file is None:
            self._open_file()

        image = self.images_dset[idx]
        mask = self.masks_dset[idx]
        patient_id = str(self.full_pids[idx])
        two_channel_mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.float32)
        two_channel_mask[mask == 0, 0] = 1.0
        two_channel_mask[mask != 0, 1] = 1.0
        try:
            augmented = self.transform(image=image, mask=two_channel_mask)
            final_image = augmented["image"]
            final_mask = augmented["mask"]
            if final_mask.shape[0] != 2:
                final_mask = final_mask.permute(2, 0, 1)
            return final_image, final_mask, patient_id
        except Exception as error:
            print(f"Error on index {idx}: {error}")
            return None, None, None

    def close(self) -> None:
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass
        finally:
            self.h5_file = None
            self.images_dset = None
            self.masks_dset = None
            self._opened_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._atexit_registered = False


def collate_validation_batch(batch: list[Any]) -> Any:
    filtered = [item for item in batch if item is not None and item[0] is not None]
    if not filtered:
        return None
    images = default_collate([item[0] for item in filtered])
    masks = default_collate([item[1] for item in filtered])
    patient_ids = [item[2] for item in filtered]
    return images, masks, patient_ids


def create_validation_dataloader(
    hdf5_path: Path,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader[Any]:
    dataset = ValidationHDF5Dataset(hdf5_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_validation_batch,
        worker_init_fn=worker_init_fn,
    )
