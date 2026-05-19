from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from helpers.provenance import hash_file_sha256
from helpers.runtime_normalization import runtime_vahadane_backend_for_provenance
from helpers.training.canonical_dataset import CanonicalDatasetLayout, CanonicalRowHDF5Dataset
from helpers.training.master_manifest_queries import CanonicalRowRecord, load_test_records
from helpers.training.runtime import worker_init_fn
from helpers.training.stain_normalization import (
    build_split_stain_normalizer,
    resolve_runtime_normalization_selection,
    resolve_stage4_split_bundle_id,
)

_BATCH_WITH_LABELS_LENGTH = 5


@dataclass(frozen=True)
class TestDatasetLayout:
    records: tuple[CanonicalRowRecord, ...]
    local_cache_dir: Path | None
    master_manifest_path: Path
    runtime_normalization_method: str = "NOT_NORMALIZED"
    runtime_vahadane_backend: str = "fixed_source"
    normalizer_device: torch.device | str = "cpu"


def setup_test_data(
    master_manifest_path: Path,
    local_data_dir: Path,
    *,
    stage_input_locally: bool,
    runtime_normalization_method: str = "NOT_NORMALIZED",
    runtime_vahadane_backend: str = "fixed_source",
    normalizer_device: torch.device | str = "cpu",
) -> TestDatasetLayout:
    if not master_manifest_path.is_file():
        raise FileNotFoundError(f"Missing {master_manifest_path}")

    local_cache_dir: Path | None = None
    if stage_input_locally:
        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
        local_cache_dir = local_data_dir / "patient_shards"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

    return TestDatasetLayout(
        records=tuple(load_test_records(master_manifest_path)),
        local_cache_dir=local_cache_dir,
        master_manifest_path=master_manifest_path,
        runtime_normalization_method=runtime_normalization_method,
        runtime_vahadane_backend=runtime_vahadane_backend,
        normalizer_device=normalizer_device,
    )


def _build_manifest_runtime_lineage_attrs(layout: TestDatasetLayout) -> dict[str, Any]:
    stage4_split_bundle_id = resolve_stage4_split_bundle_id(layout.records)
    normalization_method, normalization_artifact_id = resolve_runtime_normalization_selection(
        layout.master_manifest_path,
        layout.records,
        runtime_normalization_method=layout.runtime_normalization_method,
    )
    active_vahadane_backend = runtime_vahadane_backend_for_provenance(
        runtime_normalization_method=normalization_method,
        runtime_vahadane_backend=layout.runtime_vahadane_backend,
    )

    return {
        "master_manifest_sha256": hash_file_sha256(layout.master_manifest_path),
        "stage4_split_bundle_id": stage4_split_bundle_id,
        "runtime_normalization_method": normalization_method,
        "runtime_vahadane_backend": active_vahadane_backend,
        "normalization_method": normalization_method,
        "normalization_artifact_id": normalization_artifact_id,
    }


def collect_test_dataset_provenance(layout: TestDatasetLayout) -> dict[str, Any]:
    attrs = _build_manifest_runtime_lineage_attrs(layout)
    return {
        "path": str(layout.master_manifest_path),
        "master_manifest_path": str(layout.master_manifest_path),
        "master_manifest_sha256": hash_file_sha256(layout.master_manifest_path),
        "split": "TEST",
        "row_count": len(layout.records),
        "shard_count": len({str(record.source_hdf5_path) for record in layout.records}),
        "stage4_split_bundle_id": attrs["stage4_split_bundle_id"],
        "runtime_normalization_method": attrs["runtime_normalization_method"],
        "runtime_vahadane_backend": attrs["runtime_vahadane_backend"],
        "normalization_methods": [str(attrs["normalization_method"])],
        "attrs": attrs,
    }


class TestDataset(Dataset[Any]):
    def __init__(self, layout: TestDatasetLayout) -> None:
        self.layout = layout
        self.records = list(layout.records)
        self.base_dataset = CanonicalRowHDF5Dataset(
            CanonicalDatasetLayout(
                records=layout.records,
                local_cache_dir=layout.local_cache_dir,
            ),
            mode="val",
            mask_mode="binary",
            include_patient_id=True,
            include_filename=True,
            image_normalizer=build_split_stain_normalizer(
                layout.master_manifest_path,
                self.records,
                runtime_normalization_method=layout.runtime_normalization_method,
                runtime_vahadane_backend=layout.runtime_vahadane_backend,
                device=layout.normalizer_device,
            ),
        )
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid: int | None = None
        self._opened_shard_path: str | None = None
        self._atexit_registered = False

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, str, str | None, int]
        | tuple[None, None, None, None, None]
    ):
        try:
            image, mask, patient_id, filename = cast(
                tuple[torch.Tensor, torch.Tensor, str, str],
                self.base_dataset[idx],
            )
            self.h5_file = self.base_dataset.h5_file
            self.images_dset = self.base_dataset.images_dset
            self.masks_dset = self.base_dataset.masks_dset
            self._opened_pid = self.base_dataset._opened_pid
            self._opened_shard_path = self.base_dataset._opened_shard_path
            self._atexit_registered = self.base_dataset._atexit_registered
            return image, mask, patient_id, filename, int(self.records[idx].label)
        except Exception as error:
            print(f"Error on test index {idx}: {error}")
            return None, None, None, None, None

    def close(self) -> None:
        self.base_dataset.close()
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._opened_shard_path = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["base_dataset"] = self.base_dataset.__getstate__()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["_opened_pid"] = None
        state["_opened_shard_path"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        base_state = cast(dict[str, Any], state.pop("base_dataset"))
        self.__dict__.update(state)
        self.base_dataset = object.__new__(CanonicalRowHDF5Dataset)
        self.base_dataset.__setstate__(base_state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._opened_shard_path = None
        self._atexit_registered = False


def collate_test_batch(batch: list[Any]) -> Any:
    filtered = [item for item in batch if item is not None and item[0] is not None]
    if not filtered:
        return None
    images = default_collate([item[0] for item in filtered])
    masks = default_collate([item[1] for item in filtered])
    patient_ids = [item[2] for item in filtered]
    filenames = [item[3] for item in filtered]
    if len(filtered[0]) < _BATCH_WITH_LABELS_LENGTH:
        return images, masks, patient_ids, filenames
    labels = default_collate([int(item[4]) for item in filtered])
    return images, masks, patient_ids, filenames, labels


def create_test_dataloader(
    layout: TestDatasetLayout,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader[Any]:
    dataset = TestDataset(layout)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_test_batch,
        worker_init_fn=worker_init_fn,
    )
