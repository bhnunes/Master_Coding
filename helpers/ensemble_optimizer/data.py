from __future__ import annotations

import atexit
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from helpers.patient_shard_cache import PatientShardCache
from helpers.provenance import hash_file_sha256
from helpers.training.data import get_transforms
from helpers.training.runtime import worker_init_fn


@dataclass(frozen=True)
class ValidationShardsLayout:
    shard_dir: Path
    manifest_path: Path
    sample_manifest_path: Path
    local_cache_dir: Path | None


@dataclass(frozen=True)
class ValidationSampleRecord:
    patient_id: str
    relative_hdf5_path: str
    row_in_shard: int


def _decode_patient_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def setup_validation_shards(
    hdf5_drive_dir: Path,
    local_data_dir: Path,
    *,
    stage_input_locally: bool,
) -> ValidationShardsLayout:
    shard_dir = hdf5_drive_dir / "VALIDATION_shards"
    manifest_path = shard_dir / "manifest.parquet"
    sample_manifest_path = shard_dir / "sample_manifest.parquet"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing {shard_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not sample_manifest_path.is_file():
        raise FileNotFoundError(f"Missing {sample_manifest_path}")

    local_cache_dir: Path | None = None
    if stage_input_locally:
        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
        local_cache_dir = local_data_dir / "patient_shards"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

    return ValidationShardsLayout(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        sample_manifest_path=sample_manifest_path,
        local_cache_dir=local_cache_dir,
    )


def collect_validation_shard_provenance(layout: ValidationShardsLayout) -> dict[str, Any]:
    manifest_records = pq.read_table(layout.manifest_path).to_pylist()
    lineage_keys = (
        "source_hdf5_sha256",
        "upstream_source_signature",
        "stage4_cleaning_manifest_sha256",
    )
    lineage_attrs: dict[str, Any] | None = None
    for record in manifest_records:
        shard_path = layout.shard_dir.parent / str(record["relative_hdf5_path"])
        with h5py.File(shard_path, "r") as handle:
            observed = {key: handle.attrs.get(key) for key in lineage_keys}
        if lineage_attrs is None:
            lineage_attrs = observed
            continue
        if observed != lineage_attrs:
            raise ValueError(
                f"VALIDATION shard lineage mismatch in '{shard_path}'. Expected {lineage_attrs}, "
                f"got {observed}."
            )
    return {
        "path": str(layout.sample_manifest_path),
        "manifest_path": str(layout.manifest_path),
        "shard_dir": str(layout.shard_dir),
        "manifest_sha256": hash_file_sha256(layout.manifest_path),
        "sample_manifest_sha256": hash_file_sha256(layout.sample_manifest_path),
        "attrs": lineage_attrs or {},
    }


class ValidationHDF5Dataset(Dataset[Any]):
    def __init__(
        self,
        layout: ValidationShardsLayout,
        *,
        allowed_patients: set[str] | None = None,
    ) -> None:
        self.layout = layout
        self.transform = get_transforms(mode="validation", img_size=224)
        sample_records = pq.read_table(layout.sample_manifest_path).to_pylist()
        self.records = [
            ValidationSampleRecord(
                patient_id=_decode_patient_id(record["patient_id"]),
                relative_hdf5_path=str(record["relative_hdf5_path"]),
                row_in_shard=int(record["row_in_shard"]),
            )
            for record in sample_records
            if allowed_patients is None
            or _decode_patient_id(record["patient_id"]) in allowed_patients
        ]
        self.total_len = len(self.records)
        self.cache = (
            PatientShardCache(layout.local_cache_dir, size_cap_bytes=0)
            if layout.local_cache_dir is not None
            else None
        )
        self.h5_file: Any = None
        self.images_dset: Any = None
        self.masks_dset: Any = None
        self._opened_pid: int | None = None
        self._opened_shard_path: str | None = None
        self._atexit_registered = False

    def _resolve_shard_path(self, relative_hdf5_path: str) -> Path:
        source_path = self.layout.shard_dir.parent / relative_hdf5_path
        if self.cache is None:
            return source_path
        return self.cache.fetch(source_path)

    def _open_file(self, relative_hdf5_path: str) -> None:
        pid = os.getpid()
        resolved_shard_path = self._resolve_shard_path(relative_hdf5_path)
        shard_path_str = str(resolved_shard_path)
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
            self._opened_pid = pid
            self._opened_shard_path = shard_path_str
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str] | tuple[None, None, None]:
        record = self.records[idx]
        resolved_shard_path = str(self._resolve_shard_path(record.relative_hdf5_path))
        if self.h5_file is None or self._opened_shard_path != resolved_shard_path:
            self._open_file(record.relative_hdf5_path)

        image = self.images_dset[record.row_in_shard]
        mask = self.masks_dset[record.row_in_shard]
        two_channel_mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.float32)
        two_channel_mask[mask == 0, 0] = 1.0
        two_channel_mask[mask != 0, 1] = 1.0
        try:
            augmented = self.transform(image=image, mask=two_channel_mask)
            final_image = augmented["image"]
            final_mask = augmented["mask"]
            if final_mask.shape[0] != 2:
                final_mask = final_mask.permute(2, 0, 1)
            return final_image, final_mask, record.patient_id
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
            self._opened_shard_path = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["_opened_pid"] = None
        state["_opened_shard_path"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._opened_shard_path = None
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
    layout: ValidationShardsLayout,
    *,
    batch_size: int,
    workers: int,
    allowed_patients: set[str] | None = None,
) -> DataLoader[Any]:
    dataset = ValidationHDF5Dataset(layout, allowed_patients=allowed_patients)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_validation_batch,
        worker_init_fn=worker_init_fn,
    )


def summarize_validation_shards(layout: ValidationShardsLayout) -> tuple[list[str], set[str]]:
    ordered_patients: list[str] = []
    positive_patients: set[str] = set()
    seen_patients: set[str] = set()

    for record in pq.read_table(layout.sample_manifest_path).to_pylist():
        patient_id = _decode_patient_id(record["patient_id"])
        if patient_id not in seen_patients:
            ordered_patients.append(patient_id)
            seen_patients.add(patient_id)
        if int(record.get("label", 0)) != 0:
            positive_patients.add(patient_id)

    return ordered_patients, positive_patients
