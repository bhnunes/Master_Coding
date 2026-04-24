from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from helpers.provenance import hash_file_sha256
from helpers.training.canonical_dataset import CanonicalDatasetLayout, CanonicalRowHDF5Dataset
from helpers.training.master_manifest_queries import CanonicalRowRecord, load_validation_records
from helpers.training.runtime import worker_init_fn
from helpers.training.stain_normalization import (
    build_split_stain_normalizer,
    resolve_runtime_normalization_selection,
    resolve_stage4_split_bundle_id,
)


@dataclass(frozen=True)
class ValidationDatasetLayout:
    records: tuple[CanonicalRowRecord, ...]
    local_cache_dir: Path | None
    master_manifest_path: Path
    runtime_normalization_method: str = "NOT_NORMALIZED"


def setup_validation_data(
    master_manifest_path: Path,
    local_data_dir: Path,
    *,
    stage_input_locally: bool,
    runtime_normalization_method: str = "NOT_NORMALIZED",
) -> ValidationDatasetLayout:
    local_cache_dir = local_data_dir / "patient_shards" if stage_input_locally else None
    if local_cache_dir is not None:
        import shutil

        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
        local_cache_dir.mkdir(parents=True, exist_ok=True)

    return ValidationDatasetLayout(
        records=tuple(load_validation_records(master_manifest_path)),
        local_cache_dir=local_cache_dir,
        master_manifest_path=master_manifest_path,
        runtime_normalization_method=runtime_normalization_method,
    )


def collect_validation_provenance(layout: ValidationDatasetLayout) -> dict[str, Any]:
    stage4_split_bundle_id = resolve_stage4_split_bundle_id(layout.records)
    normalization_method, normalization_artifact_id = resolve_runtime_normalization_selection(
        layout.master_manifest_path,
        layout.records,
        runtime_normalization_method=layout.runtime_normalization_method,
    )
    return {
        "path": str(layout.master_manifest_path),
        "master_manifest_path": str(layout.master_manifest_path),
        "master_manifest_sha256": hash_file_sha256(layout.master_manifest_path),
        "row_count": len(layout.records),
        "shard_count": len({str(record.source_hdf5_path) for record in layout.records}),
        "stage4_split_bundle_id": stage4_split_bundle_id,
        "runtime_normalization_method": normalization_method,
        "normalization_methods": [normalization_method],
        "attrs": {
            "master_manifest_sha256": hash_file_sha256(layout.master_manifest_path),
            "stage4_split_bundle_id": stage4_split_bundle_id,
            "runtime_normalization_method": normalization_method,
            "normalization_method": normalization_method,
            "normalization_artifact_id": normalization_artifact_id,
        },
    }


class ValidationDataset(Dataset[Any]):
    def __init__(
        self,
        layout: ValidationDatasetLayout,
        *,
        allowed_patients: set[str] | None = None,
    ) -> None:
        self.layout = layout
        self.records = [
            record
            for record in layout.records
            if allowed_patients is None or record.patient_id in allowed_patients
        ]
        self.base_dataset = CanonicalRowHDF5Dataset(
            CanonicalDatasetLayout(
                records=tuple(self.records),
                local_cache_dir=layout.local_cache_dir,
            ),
            mode="val",
            mask_mode="two_channel",
            include_patient_id=True,
            image_normalizer=build_split_stain_normalizer(
                layout.master_manifest_path,
                self.records,
                runtime_normalization_method=layout.runtime_normalization_method,
                device="cpu",
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

    def __getitem__(self, idx: int) -> tuple[Any, Any, str] | tuple[None, None, None]:
        try:
            image, mask, patient_id = cast(tuple[Any, Any, str], self.base_dataset[idx])
            self.h5_file = self.base_dataset.h5_file
            self.images_dset = self.base_dataset.images_dset
            self.masks_dset = self.base_dataset.masks_dset
            self._opened_pid = self.base_dataset._opened_pid
            self._opened_shard_path = self.base_dataset._opened_shard_path
            self._atexit_registered = self.base_dataset._atexit_registered
            return image, mask, patient_id
        except Exception as error:
            print(f"Error on index {idx}: {error}")
            return None, None, None

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


def collate_validation_batch(batch: list[Any]) -> Any:
    filtered = [item for item in batch if item is not None and item[0] is not None]
    if not filtered:
        return None
    images = default_collate([item[0] for item in filtered])
    masks = default_collate([item[1] for item in filtered])
    patient_ids = [item[2] for item in filtered]
    return images, masks, patient_ids


def create_validation_dataloader(
    layout: ValidationDatasetLayout,
    *,
    batch_size: int,
    workers: int,
    allowed_patients: set[str] | None = None,
) -> DataLoader[Any]:
    dataset = ValidationDataset(layout, allowed_patients=allowed_patients)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_validation_batch,
        worker_init_fn=worker_init_fn,
    )


def summarize_validation_data(layout: ValidationDatasetLayout) -> tuple[list[str], set[str]]:
    ordered_patients: list[str] = []
    positive_patients: set[str] = set()
    seen_patients: set[str] = set()

    for record in layout.records:
        patient_id = record.patient_id
        if patient_id not in seen_patients:
            ordered_patients.append(patient_id)
            seen_patients.add(patient_id)
        if record.label != 0:
            positive_patients.add(patient_id)

    return ordered_patients, positive_patients
