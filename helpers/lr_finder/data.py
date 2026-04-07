from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from helpers.lr_finder.config import LRFinderConfig
from helpers.training.data import (
    HybridProstateDataset,
    collate_batch,
    create_stratified_subset_within_patients,
    get_training_hdf5_filename,
)
from helpers.training.runtime import worker_init_fn


@dataclass(frozen=True)
class PreparedTrainingData:
    source_h5_path: Path
    dataset: HybridProstateDataset
    sample_weights: torch.Tensor


def prepare_source_h5(config: LRFinderConfig) -> Path:
    source_h5_path = Path(
        get_training_hdf5_filename(str(config.hdf5_drive_dir), config.smart_sampling)
    )
    if not config.stage_input_locally:
        return source_h5_path

    config.local_data_dir.mkdir(parents=True, exist_ok=True)
    local_path = config.local_data_dir / source_h5_path.name
    shutil.copy2(source_h5_path, local_path)
    return local_path


def prepare_training_data(config: LRFinderConfig) -> PreparedTrainingData:
    source_h5_path = prepare_source_h5(config)
    base_dataset = HybridProstateDataset(str(source_h5_path), mode="train")
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
        dataset = HybridProstateDataset(
            str(source_h5_path), mode="train", subset_indices=subset_indices
        )
    else:
        dataset = base_dataset

    labels = np.asarray(dataset.get_labels())
    class_counts = np.bincount(labels)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return PreparedTrainingData(
        source_h5_path=source_h5_path,
        dataset=dataset,
        sample_weights=sample_weights,
    )


def build_train_loader(
    dataset: HybridProstateDataset,
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
