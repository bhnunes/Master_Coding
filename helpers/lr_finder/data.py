from __future__ import annotations

import atexit
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from helpers.lr_finder.config import LRFinderConfig
from helpers.patient_shard_cache import PatientShardCache
from helpers.provenance import hash_file_sha256
from helpers.training.data import (
    collate_batch,
    create_stratified_subset_within_patients,
    get_transforms,
)
from helpers.training.runtime import worker_init_fn


@dataclass(frozen=True)
class ShardDatasetLayout:
    shard_dir: Path
    manifest_path: Path
    sample_manifest_path: Path
    local_cache_dir: Path | None


@dataclass(frozen=True)
class TrainingSampleRecord:
    patient_id: str
    label: int
    relative_hdf5_path: str
    row_in_shard: int


@dataclass(frozen=True)
class PreparedTrainingData:
    source_layout: ShardDatasetLayout
    validation_layout: ShardDatasetLayout
    source_split_name: str
    dataset: Dataset[Any]
    sample_weights: torch.Tensor
    training_provenance: dict[str, Any]
    validation_provenance: dict[str, Any]


def _decode_patient_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_filename(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_shard_layout(
    drive_dir: Path,
    split_dir_name: str,
    *,
    local_data_dir: Path,
    stage_input_locally: bool,
) -> ShardDatasetLayout:
    shard_dir = drive_dir / split_dir_name
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
        local_cache_dir = local_data_dir / split_dir_name
        local_cache_dir.mkdir(parents=True, exist_ok=True)

    return ShardDatasetLayout(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        sample_manifest_path=sample_manifest_path,
        local_cache_dir=local_cache_dir,
    )


def setup_lr_finder_data(
    config: LRFinderConfig,
) -> tuple[ShardDatasetLayout, ShardDatasetLayout, str]:
    if config.stage_input_locally and config.local_data_dir.exists():
        shutil.rmtree(config.local_data_dir)
    if config.stage_input_locally:
        config.local_data_dir.mkdir(parents=True, exist_ok=True)

    filtered_shard_dir = config.hdf5_drive_dir / "TRAIN_FILTERED_shards"
    source_split_name = (
        "TRAIN_FILTERED_shards"
        if config.smart_sampling and filtered_shard_dir.is_dir()
        else "TRAIN_shards"
    )
    source_layout = _load_shard_layout(
        config.hdf5_drive_dir,
        source_split_name,
        local_data_dir=config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    validation_layout = _load_shard_layout(
        config.hdf5_drive_dir,
        "VALIDATION_shards",
        local_data_dir=config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    return source_layout, validation_layout, source_split_name


def collect_shard_layout_provenance(layout: ShardDatasetLayout) -> dict[str, Any]:
    manifest_records = pq.read_table(layout.manifest_path).to_pylist()
    lineage_keys = (
        "source_hdf5_sha256",
        "upstream_source_signature",
        "stage4_cleaning_manifest_sha256",
        "selection_signature",
    )
    lineage_attrs: dict[str, Any] | None = None
    for record in manifest_records:
        shard_path = layout.shard_dir.parent / str(record["relative_hdf5_path"])
        with h5py.File(shard_path, "r") as handle:
            observed = {key: handle.attrs.get(key) for key in lineage_keys if key in handle.attrs}
        if lineage_attrs is None:
            lineage_attrs = observed
            continue
        if observed != lineage_attrs:
            raise ValueError(
                f"Shard lineage mismatch in '{shard_path}'. "
                f"Expected {lineage_attrs}, got {observed}."
            )
    return {
        "path": str(layout.sample_manifest_path),
        "manifest_path": str(layout.manifest_path),
        "shard_dir": str(layout.shard_dir),
        "manifest_sha256": hash_file_sha256(layout.manifest_path),
        "sample_manifest_sha256": hash_file_sha256(layout.sample_manifest_path),
        "attrs": lineage_attrs or {},
    }


class TrainingShardDataset(Dataset[Any]):
    def __init__(
        self,
        layout: ShardDatasetLayout,
        *,
        mode: str = "train",
        subset_indices: list[int] | None = None,
    ) -> None:
        self.layout = layout
        self.mode = mode
        self.transform = get_transforms(mode=mode, img_size=224)
        records = [
            TrainingSampleRecord(
                patient_id=_decode_patient_id(record["patient_id"]),
                label=int(record["label"]),
                relative_hdf5_path=str(record["relative_hdf5_path"]),
                row_in_shard=int(record["row_in_shard"]),
            )
            for record in pq.read_table(layout.sample_manifest_path).to_pylist()
        ]
        if subset_indices is not None:
            self.records = [records[index] for index in subset_indices]
        else:
            self.records = records
        self.indices = np.arange(len(self.records), dtype=np.int64)
        self.labels = np.asarray([record.label for record in self.records], dtype=np.int64)
        self.patient_ids = np.asarray([record.patient_id for record in self.records], dtype=str)
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
            self.filenames_dset = self.h5_file["filenames"]
            self._opened_pid = pid
            self._opened_shard_path = shard_path_str
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        resolved_shard_path = str(self._resolve_shard_path(record.relative_hdf5_path))
        if self.h5_file is None or self._opened_shard_path != resolved_shard_path:
            self._open_file(record.relative_hdf5_path)

        image = self.images_dset[record.row_in_shard]
        mask = self.masks_dset[record.row_in_shard]
        _filename = _decode_filename(self.filenames_dset[record.row_in_shard])
        try:
            augmented = self.transform(image=image, mask=mask)
            transformed_mask = augmented["mask"]
            if transformed_mask.ndim == 3 and transformed_mask.shape[-1] == 1:
                transformed_mask = transformed_mask.squeeze(-1)
            return augmented["image"], transformed_mask.long()
        except Exception as error:
            raise RuntimeError(f"Transform failed at idx={idx}") from error

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

    def __del__(self) -> None:
        self.close()

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


def prepare_training_data(config: LRFinderConfig) -> PreparedTrainingData:
    source_layout, validation_layout, source_split_name = setup_lr_finder_data(config)
    base_dataset = TrainingShardDataset(source_layout, mode="train")
    subset_indices: list[int] | None = None
    if config.use_subset and config.subset_ratio < 1.0:
        subset = create_stratified_subset_within_patients(
            base_dataset,
            config.subset_ratio,
            split_name="train",
            seed=config.seed,
        )
        subset_indices = [int(index) for index in subset.indices]
        base_dataset.close()
        dataset = TrainingShardDataset(source_layout, mode="train", subset_indices=subset_indices)
    else:
        dataset = base_dataset

    labels = np.asarray(dataset.get_labels())
    class_counts = np.bincount(labels)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return PreparedTrainingData(
        source_layout=source_layout,
        validation_layout=validation_layout,
        source_split_name=source_split_name,
        dataset=dataset,
        sample_weights=sample_weights,
        training_provenance=collect_shard_layout_provenance(source_layout),
        validation_provenance=collect_shard_layout_provenance(validation_layout),
    )


def build_train_loader(
    dataset: Dataset[Any],
    sample_weights: torch.Tensor,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[object]:
    sampler = WeightedRandomSampler(
        weights=cast(Sequence[float], sample_weights.numpy()),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
