from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from helpers.lr_finder.config import LRFinderConfig
from helpers.training.canonical_dataset import CanonicalDatasetLayout, CanonicalRowHDF5Dataset
from helpers.training.compact_train_selected import remap_records_to_compact_train_selected
from helpers.training.data import collect_manifest_split_provenance
from helpers.training.master_manifest_queries import (
    format_split_record_availability,
    load_lr_finder_training_records,
    load_validation_records,
    summarize_split_record_availability,
)
from helpers.training.runtime import worker_init_fn
from helpers.training.stain_normalization import (
    build_split_stain_normalizer,
)


@dataclass(frozen=True)
class PreparedTrainingData:
    source_split_name: str
    dataset: Dataset[Any]
    sample_weights: torch.Tensor
    training_provenance: dict[str, Any]
    validation_provenance: dict[str, Any]


def prepare_training_data(
    config: LRFinderConfig,
    *,
    normalizer_device: torch.device | str = "cpu",
) -> PreparedTrainingData:
    use_compact_train_selected = config.smart_sampling and config.use_compact_train_selected
    use_local_cache = config.stage_input_locally or use_compact_train_selected
    if use_local_cache and config.local_data_dir.exists():
        import shutil

        shutil.rmtree(config.local_data_dir)
    if use_local_cache:
        config.local_data_dir.mkdir(parents=True, exist_ok=True)

    training_records = load_lr_finder_training_records(
        config.master_manifest_path,
        smart_sampling=config.smart_sampling,
    )
    if not training_records:
        _raise_empty_lr_finder_training_records(config)
    original_training_records = tuple(training_records)
    compact_provenance: dict[str, Any] | None = None
    if use_compact_train_selected:
        compact_result = remap_records_to_compact_train_selected(
            original_training_records,
            compact_dir=config.compact_train_selected_dir,
            master_manifest_path=config.master_manifest_path,
        )
        training_records = list(compact_result.records)
        compact_provenance = compact_result.provenance
    validation_records = load_validation_records(config.master_manifest_path)
    source_split_name = "TRAIN_SELECTED" if config.smart_sampling else "TRAIN"
    local_cache_dir = (
        config.local_data_dir / "TRAIN_SELECTED_COMPACT"
        if use_compact_train_selected
        else config.local_data_dir
        if config.stage_input_locally
        else None
    )
    image_normalizer = build_split_stain_normalizer(
        config.master_manifest_path,
        training_records,
        runtime_normalization_method=config.runtime_normalization_method,
        runtime_vahadane_backend=config.runtime_vahadane_backend,
        device=normalizer_device,
        source_matrix_cache_path=config.stain_matrix_cache_path,
    )

    base_dataset = CanonicalRowHDF5Dataset(
        CanonicalDatasetLayout(records=tuple(training_records), local_cache_dir=local_cache_dir),
        mode="train",
        mask_mode="raw",
        image_normalizer=image_normalizer,
    )
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
        if not subset_indices:
            _raise_empty_lr_finder_subset(config, source_row_count=len(training_records))
        subset_records = tuple(training_records[index] for index in subset_indices)
        dataset = CanonicalRowHDF5Dataset(
            CanonicalDatasetLayout(records=subset_records, local_cache_dir=local_cache_dir),
            mode="train",
            mask_mode="raw",
            image_normalizer=image_normalizer,
        )
        provenance_records = tuple(original_training_records[index] for index in subset_indices)
    else:
        dataset = base_dataset
        provenance_records = original_training_records

    labels = np.asarray(dataset.get_labels())
    class_counts = np.bincount(labels)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return PreparedTrainingData(
        source_split_name=source_split_name,
        dataset=dataset,
        sample_weights=sample_weights,
        training_provenance=_with_compact_provenance(
            collect_manifest_split_provenance(
                config.master_manifest_path,
                records=provenance_records,
                split="TRAIN",
                smart_sampling=config.smart_sampling,
                runtime_normalization_method=config.runtime_normalization_method,
                runtime_vahadane_backend=config.runtime_vahadane_backend,
            ),
            compact_provenance=compact_provenance,
        ),
        validation_provenance=collect_manifest_split_provenance(
            config.master_manifest_path,
            records=validation_records,
            split="VALIDATION",
            smart_sampling=False,
            runtime_normalization_method=config.runtime_normalization_method,
            runtime_vahadane_backend=config.runtime_vahadane_backend,
        ),
    )


def _raise_empty_lr_finder_training_records(config: LRFinderConfig) -> None:
    availability = format_split_record_availability(
        summarize_split_record_availability(config.master_manifest_path)
    )
    selection = (
        "Stage 6-selected TRAIN rows (is_stage7_selected=1)"
        if config.smart_sampling
        else "Stage 4-accepted TRAIN rows"
    )
    guidance = (
        "LR_FINDER_SMART_SAMPLING=True makes Stage 7 require a completed Stage 6 "
        "selection for TRAIN. Rerun 6_smart_sampler.py for this manifest and confirm it "
        "finishes the manifest update, or set LR_FINDER_SMART_SAMPLING=False to use all "
        "Stage 4-accepted TRAIN rows."
        if config.smart_sampling
        else "Check that 4_crossfold.py assigned accepted rows to split=TRAIN before "
        "running Stage 7."
    )
    raise ValueError(
        "No LR finder training rows are available for "
        f"{selection}. master_manifest_path={config.master_manifest_path}. "
        f"Stage 4-accepted manifest counts: {availability}. {guidance}"
    )


def _raise_empty_lr_finder_subset(config: LRFinderConfig, *, source_row_count: int) -> None:
    raise ValueError(
        "LR finder subset selection produced zero TRAIN rows. "
        f"source_row_count={source_row_count}, "
        f"LR_FINDER_SUBSET_RATIO={config.subset_ratio}. Increase LR_FINDER_SUBSET_RATIO "
        "or disable LR_FINDER_USE_SUBSET."
    )


def _with_compact_provenance(
    provenance: dict[str, Any],
    *,
    compact_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    if compact_provenance is None:
        return provenance
    enriched = dict(provenance)
    enriched["storage_backend"] = compact_provenance
    return enriched


def build_train_loader(
    dataset: Dataset[Any],
    sample_weights: torch.Tensor,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[object]:
    if len(sample_weights) <= 0:
        raise ValueError(
            "Cannot build LR finder training loader with zero sample weights. "
            "The training dataset is empty; check LR_FINDER_SMART_SAMPLING, Stage 6 "
            "selection, and LR_FINDER_SUBSET_RATIO."
        )
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


def collate_batch(batch: list[Any]) -> Any:
    from helpers.training.data import collate_batch as training_collate_batch

    return training_collate_batch(batch)


def create_stratified_subset_within_patients(
    full_dataset: Any,
    ratio: float,
    split_name: str = "Unknown",
    seed: int = 42,
) -> Any:
    from helpers.training.data import (
        create_stratified_subset_within_patients as training_create_subset,
    )

    return training_create_subset(full_dataset, ratio, split_name=split_name, seed=seed)
