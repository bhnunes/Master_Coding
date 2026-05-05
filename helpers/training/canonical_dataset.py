from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from helpers.patient_shard_cache import PatientShardCache
from helpers.training.master_manifest_queries import CanonicalRowRecord
from helpers.training.stain_normalization import ImageStainNormalizer

_TWO_CHANNEL_MASK_COUNT = 2
_MASK_IMAGE_NDIM = 3
_SINGLE_CHANNEL_COUNT = 1

MaskMode = Literal["raw", "binary", "two_channel"]


@dataclass(frozen=True)
class CanonicalDatasetLayout:
    records: tuple[CanonicalRowRecord, ...]
    local_cache_dir: Path | None


class CanonicalRowHDF5Dataset(Dataset[Any]):
    """Load canonical Stage 2 rows using startup-cached metadata only."""

    def __init__(
        self,
        layout: CanonicalDatasetLayout,
        *,
        mode: str,
        mask_mode: MaskMode,
        include_patient_id: bool = False,
        include_filename: bool = False,
        image_normalizer: ImageStainNormalizer | None = None,
    ) -> None:
        self.layout = layout
        self.mode = mode
        self.mask_mode = mask_mode
        self.include_patient_id = include_patient_id
        self.include_filename = include_filename
        self.image_normalizer = image_normalizer
        self.transform = get_transforms(mode=mode, img_size=224)
        self.records = list(layout.records)
        self.labels = np.asarray([record.label for record in self.records], dtype=np.int64)
        self.patient_ids = np.asarray([record.patient_id for record in self.records], dtype=str)
        self.filenames = np.asarray([record.filename for record in self.records], dtype=str)
        self.cache = (
            PatientShardCache(layout.local_cache_dir, size_cap_bytes=0)
            if layout.local_cache_dir is not None
            else None
        )
        self.h5_file: Any = None
        self.images_dset: Any = None
        self.masks_dset: Any = None
        self.filenames_dset: Any = None
        self._opened_pid: int | None = None
        self._opened_shard_path: str | None = None
        self._atexit_registered = False

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Any:
        record = self.records[idx]
        resolved_shard_path = self._resolve_shard_path(record.source_hdf5_path)
        resolved_shard_path_str = str(resolved_shard_path)
        if self.h5_file is None or self._opened_shard_path != resolved_shard_path_str:
            self._open_file(resolved_shard_path)

        image = np.asarray(self.images_dset[record.source_row_index], dtype=np.uint8)
        if self.image_normalizer is not None:
            image = self.image_normalizer.normalize_image(image, cache_key=record.filename)
        mask = self._prepare_mask(self.masks_dset[record.source_row_index])
        augmented = self.transform(image=image, mask=mask)
        final_mask = self._finalize_mask(augmented["mask"])

        output: list[Any] = [augmented["image"], final_mask]
        if self.include_patient_id:
            output.append(record.patient_id)
        if self.include_filename:
            filename = _decode_filename(self.filenames_dset[record.source_row_index])
            output.append(filename)
        return tuple(output)

    def get_labels(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.asarray(self.labels)

    def get_patient_ids(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.asarray(self.patient_ids)

    def close(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
            self.images_dset = None
            self.masks_dset = None
            self.filenames_dset = None
            self._opened_pid = None
            self._opened_shard_path = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["filenames_dset"] = None
        state["_opened_pid"] = None
        state["_opened_shard_path"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self.filenames_dset = None
        self._opened_pid = None
        self._opened_shard_path = None
        self._atexit_registered = False

    def _resolve_shard_path(self, source_hdf5_path: Path) -> Path:
        if self.cache is None:
            return source_hdf5_path
        return self.cache.fetch(source_hdf5_path)

    def _open_file(self, shard_path: Path) -> None:
        pid = os.getpid()
        shard_path_str = str(shard_path)
        if (
            self.h5_file is not None
            and self._opened_pid is not None
            and (self._opened_pid != pid or self._opened_shard_path != shard_path_str)
        ):
            self.close()

        if self.h5_file is None:
            self.h5_file = h5py.File(
                shard_path_str,
                "r",
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024,
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self.filenames_dset = _get_filenames_dataset(self.h5_file)
            self._opened_pid = pid
            self._opened_shard_path = shard_path_str
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def _prepare_mask(self, mask: Any) -> Any:
        if self.mask_mode == "binary":
            return (np.asarray(mask) != 0).astype(np.uint8)
        if self.mask_mode == "two_channel":
            mask_array = np.asarray(mask)
            two_channel_mask = np.zeros(
                (mask_array.shape[0], mask_array.shape[1], _TWO_CHANNEL_MASK_COUNT),
                dtype=np.float32,
            )
            two_channel_mask[mask_array == 0, 0] = 1.0
            two_channel_mask[mask_array != 0, 1] = 1.0
            return two_channel_mask
        return mask

    def _finalize_mask(self, mask: Any) -> torch.Tensor:
        if self.mask_mode == "two_channel":
            final_mask = mask
            if final_mask.shape[0] != _TWO_CHANNEL_MASK_COUNT:
                final_mask = final_mask.permute(2, 0, 1)
            return cast(torch.Tensor, final_mask)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.asarray(mask))
        if self.mask_mode == "binary":
            return cast(torch.Tensor, mask.to(torch.uint8))
        if mask.ndim == _MASK_IMAGE_NDIM and mask.shape[-1] == _SINGLE_CHANNEL_COUNT:
            mask = mask.squeeze(-1)
        return cast(torch.Tensor, mask.long())


def _decode_filename(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _get_filenames_dataset(handle: h5py.File) -> Any:
    for dataset_name in ("filenames", "filename"):
        if dataset_name in handle:
            return handle[dataset_name]
    available = ", ".join(handle.keys())
    raise KeyError(
        "HDF5 file is missing the filename dataset. Expected one of "
        f"('filenames', 'filename'). Available: {available}"
    )


def get_transforms(mode: str = "train", img_size: int = 224) -> Any:
    from helpers.training.data import get_transforms as training_get_transforms

    return training_get_transforms(mode=mode, img_size=img_size)
