from __future__ import annotations

import atexit
import os
import shutil
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import psutil
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import default_collate

from helpers.provenance import hash_file_sha256, hash_json_payload
from helpers.runtime_normalization import runtime_vahadane_backend_for_provenance
from helpers.training.canonical_dataset import CanonicalDatasetLayout, CanonicalRowHDF5Dataset
from helpers.training.master_manifest_queries import (
    CanonicalRowRecord,
    load_training_records,
    load_validation_records,
)
from helpers.training.stain_normalization import (
    build_split_stain_normalizer,
    resolve_runtime_normalization_selection,
    resolve_stage4_split_bundle_id,
)

NumericArray = npt.NDArray[np.generic]
ArtifactCoverageLookup = dict[str, tuple[float, float, float, float, float]]
ZERO_ARTIFACT_COVERAGE = (0.0, 0.0, 0.0, 0.0, 0.0)
_MASK_IMAGE_NDIM = 3
_SINGLE_CHANNEL_COUNT = 1
FILENAME_DATASET_CANDIDATES = ("filenames", "filename")
ARTIFACT_COVERAGE_COLUMNS = (
    "cov_fold",
    "cov_penmarking",
    "cov_oof",
    "cov_darkspot_foreign",
    "cov_edge_airbubble",
)


def _decode_filename(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _squeeze_single_channel_mask(mask: Any) -> Any:
    if mask.ndim == _MASK_IMAGE_NDIM and mask.shape[-1] == _SINGLE_CHANNEL_COUNT:
        return mask.squeeze(-1)
    return mask


def _get_filenames_dataset(handle: h5py.File) -> Any:
    for dataset_name in FILENAME_DATASET_CANDIDATES:
        if dataset_name in handle:
            return handle[dataset_name]
    available = ", ".join(handle.keys())
    raise KeyError(
        "HDF5 file is missing the filename dataset. Expected one of "
        f"{FILENAME_DATASET_CANDIDATES}. Available: {available}"
    )


def load_artifact_coverage_lookup(master_manifest_path: str) -> ArtifactCoverageLookup:
    """Load filename-keyed artifact coverage vectors from `master_manifest.sqlite`."""

    manifest_path = Path(master_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Master manifest does not exist: {manifest_path}")

    with sqlite3.connect(manifest_path) as connection:
        rows = connection.execute(
            "SELECT filename, cov_fold, cov_penmarking, cov_oof, "
            "cov_darkspot_foreign, cov_edge_airbubble FROM patches"
        ).fetchall()

    lookup: ArtifactCoverageLookup = {}
    for row in rows:
        filename, cov_fold, cov_penmarking, cov_oof, cov_darkspot_foreign, cov_edge_airbubble = row
        lookup[str(filename)] = (
            float(cov_fold or 0.0),
            float(cov_penmarking or 0.0),
            float(cov_oof or 0.0),
            float(cov_darkspot_foreign or 0.0),
            float(cov_edge_airbubble or 0.0),
        )
    return lookup


def get_transforms(mode: str = "train", img_size: int = 224) -> Any:
    """Return the existing augmentation pipeline for train or validation."""

    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-45, 45),
                    fill=0,
                    fill_mask=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    ],
                    p=0.8,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1.0),
                    ],
                    p=0.14,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(1, int(img_size / 10)),
                    hole_width_range=(1, int(img_size / 10)),
                    fill=0,
                    fill_mask=0,
                    p=0.2,
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose([ToTensorV2()])


def collate_batch(batch: list[Any]) -> Any:
    """Drop invalid samples before default PyTorch collation."""

    filtered_batch = [
        item for item in batch if item is not None and item[0] is not None and item[1] is not None
    ]
    return default_collate(filtered_batch) if filtered_batch else None


def get_training_hdf5_filename(drive_dir: str, smart_sampling: bool) -> str:
    """Select the training HDF5 file according to smart-sampling settings."""

    original_src = os.path.join(drive_dir, "TRAIN.h5")
    filtered_src = os.path.join(drive_dir, "TRAIN_FILTERED.h5")
    if not os.path.exists(drive_dir):
        raise FileNotFoundError(f"Missing {drive_dir}")
    if smart_sampling and os.path.exists(original_src) and os.path.exists(filtered_src):
        with h5py.File(filtered_src, "r") as handle:
            expected_source_sha = handle.attrs.get("source_hdf5_sha256")
            selection_signature = handle.attrs.get("selection_signature")
        if isinstance(expected_source_sha, bytes):
            expected_source_sha = expected_source_sha.decode("utf-8")
        if isinstance(selection_signature, bytes):
            selection_signature = selection_signature.decode("utf-8")
        if not expected_source_sha or not selection_signature:
            raise ValueError(
                "TRAIN_FILTERED.h5 is missing fail-closed smart-sampling provenance metadata."
            )
        observed_source_sha = hash_file_sha256(original_src)
        if observed_source_sha != expected_source_sha:
            raise ValueError(
                "TRAIN_FILTERED.h5 does not match the current TRAIN.h5 content. "
                "Regenerate TRAIN_FILTERED.h5 before continuing."
            )
        return filtered_src
    return original_src


def setup_local_hdf5(drive_dir: str, local_dir: str, smart_sampling: bool) -> str:
    """Copy required HDF5 files to the local fast-storage directory."""

    print(f"\n{'=' * 25} Setting up HDF5 Data {'=' * 25}")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    chosen_file_name = os.path.basename(get_training_hdf5_filename(drive_dir, smart_sampling))
    required_files = ["VALIDATION.h5", chosen_file_name]

    print(f"Source Directory: {drive_dir}")
    start_time = time.time()
    for filename in required_files:
        src = os.path.join(drive_dir, filename)
        dst = os.path.join(local_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Critical data missing: {src}")
        print(f"Copying {filename} to local disk...")
        shutil.copy2(src, dst)
        if os.path.getsize(src) != os.path.getsize(dst):
            raise RuntimeError(f"Copy failed for {filename}: Size mismatch.")

    elapsed = time.time() - start_time
    print(f"Data transfer complete in {elapsed:.2f} seconds.")
    return chosen_file_name


class HybridProstateDataset(Dataset[Any]):
    """HDF5 dataset that optionally caches a subset in RAM."""

    def __init__(
        self,
        hdf5_path: str,
        mode: str = "train",
        subset_indices: list[int] | None = None,
        artifact_coverage_by_filename: ArtifactCoverageLookup | None = None,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.transform = get_transforms(mode=mode, img_size=224)

        print(f"Opening {hdf5_path}...")
        with h5py.File(self.hdf5_path, "r") as handle:
            images = cast(Any, handle["images"])
            masks = cast(Any, handle["masks"])
            labels = cast(Any, handle["labels"])
            patient_ids = cast(Any, handle["patient_ids"])
            filenames = cast(Any, _get_filenames_dataset(handle))
            self.full_labels = np.asarray(labels[:])
            self.full_pids = np.asarray(patient_ids[:])
            self.full_filenames = np.asarray(filenames[:])
            img_shape = tuple(images.shape)
            mask_shape = tuple(masks.shape)

        if subset_indices is not None:
            self.indices = np.asarray(subset_indices)
        else:
            self.indices = np.arange(len(self.full_labels))

        bytes_per_sample = (img_shape[1] * img_shape[2] * img_shape[3]) + (
            mask_shape[1] * mask_shape[2]
        )
        total_bytes_needed = len(self.indices) * bytes_per_sample
        available_ram = psutil.virtual_memory().available
        self.use_ram_cache = total_bytes_needed < (available_ram * 0.70)
        self.images_cache: NumericArray | None = None
        self.masks_cache: NumericArray | None = None
        self.h5_file: Any = None
        self.images_dset: Any = None
        self.masks_dset: Any = None
        self.filenames_dset: Any = None
        self._opened_pid: int | None = None
        self._atexit_registered = False
        self.artifact_coverage_by_filename = artifact_coverage_by_filename

        if self.use_ram_cache:
            print(
                f"RAM Sufficient ({available_ram / 1e9:.1f}GB avail). "
                f"Loading {total_bytes_needed / 1e9:.2f}GB into memory..."
            )
            self._load_to_ram()
        else:
            print(
                f"Dataset too large ({total_bytes_needed / 1e9:.2f}GB) for RAM "
                f"({available_ram / 1e9:.1f}GB). Using disk mode."
            )
            self.labels = self.full_labels[self.indices]
            self.patient_ids = self.full_pids[self.indices]

    def _load_to_ram(self) -> None:
        start_time = time.time()
        sorted_indices = np.sort(self.indices)
        with h5py.File(self.hdf5_path, "r") as handle:
            self.images_cache = np.asarray(cast(Any, handle["images"])[sorted_indices])
            self.masks_cache = np.asarray(cast(Any, handle["masks"])[sorted_indices])
        self.labels = self.full_labels[sorted_indices]
        self.patient_ids = self.full_pids[sorted_indices]
        self.filenames = self.full_filenames[sorted_indices]
        self.indices = np.arange(len(sorted_indices))
        print(f"Loaded {len(self.indices)} samples in {time.time() - start_time:.2f}s.")

    def _open_file(self) -> None:
        pid = os.getpid()
        if self.h5_file is not None and self._opened_pid != pid:
            self.close()

        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.hdf5_path, "r", libver="latest", rdcc_nbytes=50 * 1024 * 1024
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self.filenames_dset = _get_filenames_dataset(self.h5_file)
            self._opened_pid = pid
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_ram_cache:
            assert self.images_cache is not None
            assert self.masks_cache is not None
            image = self.images_cache[idx]
            mask = self.masks_cache[idx]
            filename = _decode_filename(self.filenames[idx])
        else:
            if self.h5_file is None:
                self._open_file()
            real_idx = int(self.indices[idx])
            image = self.images_dset[real_idx]
            mask = self.masks_dset[real_idx]
            filename = _decode_filename(self.filenames_dset[real_idx])

        try:
            augmented = self.transform(image=image, mask=mask)
            transformed_mask = _squeeze_single_channel_mask(augmented["mask"])
            if self.artifact_coverage_by_filename is None:
                return augmented["image"], transformed_mask.long()
            artifact_covariates = torch.tensor(
                self.artifact_coverage_by_filename.get(filename, ZERO_ARTIFACT_COVERAGE),
                dtype=torch.float32,
            )
            return augmented["image"], transformed_mask.long(), artifact_covariates
        except Exception as error:
            raise RuntimeError(f"Transform failed at idx={idx}") from error

    def get_labels(self) -> NumericArray:
        return np.asarray(self.labels)

    def get_patient_ids(self) -> NumericArray:
        return np.asarray(self.patient_ids)

    def get_class_counts(self) -> dict[str, int]:
        counts = np.bincount(np.asarray(self.labels))
        return {
            "CANCER": int(counts[1]) if len(counts) > 1 else 0,
            "NOT_CANCER": int(counts[0]) if len(counts) > 0 else 0,
        }

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

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["filenames_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self.filenames_dset = None
        self._opened_pid = None
        self._atexit_registered = False


class ProstateCancerDatasetHDF5(Dataset[Any]):
    """Lazy HDF5 dataset used by validation and training paths."""

    def __init__(
        self,
        hdf5_path: str,
        mode: str = "train",
        subset_indices: list[int] | None = None,
        artifact_coverage_by_filename: ArtifactCoverageLookup | None = None,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.transform = get_transforms(mode=mode, img_size=224)

        with h5py.File(self.hdf5_path, "r") as handle:
            self.full_labels = np.asarray(cast(Any, handle["labels"])[:])
            self.full_pids = np.asarray(cast(Any, handle["patient_ids"])[:])
            self.full_filenames = np.asarray(cast(Any, _get_filenames_dataset(handle))[:])
            self.total_len = len(self.full_labels)

        if subset_indices is not None:
            self.indices = np.asarray(subset_indices)
        else:
            self.indices = np.arange(self.total_len)

        self.labels = self.full_labels[self.indices]
        self.patient_ids = self.full_pids[self.indices]
        self.filenames = self.full_filenames[self.indices]
        self.h5_file: Any = None
        self.images_dset: Any = None
        self.masks_dset: Any = None
        self.filenames_dset: Any = None
        self._opened_pid: int | None = None
        self._atexit_registered = False
        self.artifact_coverage_by_filename = artifact_coverage_by_filename

    def _open_file(self) -> None:
        pid = os.getpid()
        if self.h5_file is not None and self._opened_pid is not None and self._opened_pid != pid:
            self.close()

        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.hdf5_path,
                "r",
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024,
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self.filenames_dset = _get_filenames_dataset(self.h5_file)
            self._opened_pid = pid
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        if self.h5_file is None:
            self._open_file()
        real_idx = int(self.indices[idx])
        image = self.images_dset[real_idx]
        mask = self.masks_dset[real_idx]
        filename = _decode_filename(self.filenames_dset[real_idx])
        try:
            augmented = self.transform(image=image, mask=mask)
            transformed_mask = augmented["mask"].long()
            if self.artifact_coverage_by_filename is None:
                return augmented["image"], transformed_mask
            artifact_covariates = torch.tensor(
                self.artifact_coverage_by_filename.get(filename, ZERO_ARTIFACT_COVERAGE),
                dtype=torch.float32,
            )
            return augmented["image"], transformed_mask, artifact_covariates
        except Exception as error:
            print(f"Error on index {idx}: {error}")
            return None, None

    def get_labels(self) -> NumericArray:
        return np.asarray(self.labels)

    def get_patient_ids(self) -> NumericArray:
        return np.asarray(self.patient_ids)

    def get_class_counts(self) -> dict[str, int]:
        counts = np.bincount(np.asarray(self.labels))
        return {
            "CANCER": int(counts[1]) if len(counts) > 1 else 0,
            "NOT_CANCER": int(counts[0]) if len(counts) > 0 else 0,
        }

    def close(self) -> None:
        try:
            if self.h5_file is not None:
                try:
                    self.h5_file.close()
                except Exception:
                    pass
        finally:
            self.h5_file = None
            self.images_dset = None
            self.masks_dset = None
            self.filenames_dset = None
            self._opened_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["filenames_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self.filenames_dset = None
        self._opened_pid = None
        self._atexit_registered = False


class SubsetView(Dataset[Any]):
    """Thin view over an existing dataset preserving helper accessors."""

    def __init__(self, base_ds: Any, indices: list[int] | NumericArray) -> None:
        self.base_ds = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.base_ds[int(self.indices[index])]

    def get_labels(self) -> list[Any]:
        labels = np.asarray(self.base_ds.get_labels())
        return list(labels[self.indices].tolist())

    def get_patient_ids(self) -> list[Any]:
        patient_ids = np.asarray(self.base_ds.get_patient_ids())
        return list(patient_ids[self.indices].tolist())


@dataclass(frozen=True)
class PreparedTrainingData:
    train_dataset: Dataset[Any]
    validation_dataset: Dataset[Any]
    sample_weights: torch.Tensor
    source_split_name: str
    training_provenance: dict[str, Any]
    validation_provenance: dict[str, Any]


class ArtifactAwareDatasetView(Dataset[Any]):
    """Append filename-keyed artifact covariates to a base dataset item."""

    def __init__(
        self,
        base_dataset: CanonicalRowHDF5Dataset,
        artifact_coverage_by_filename: ArtifactCoverageLookup,
    ) -> None:
        self.base_dataset = base_dataset
        self.artifact_coverage_by_filename = artifact_coverage_by_filename

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        image, mask = self.base_dataset[index]
        filename = str(self.base_dataset.filenames[index])
        artifact_covariates = torch.tensor(
            self.artifact_coverage_by_filename.get(filename, ZERO_ARTIFACT_COVERAGE),
            dtype=torch.float32,
        )
        return image, mask, artifact_covariates

    def get_labels(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return self.base_dataset.get_labels()

    def get_patient_ids(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        return self.base_dataset.get_patient_ids()

    def close(self) -> None:
        self.base_dataset.close()

    @property
    def records(self) -> list[CanonicalRowRecord]:
        return self.base_dataset.records


def verify_patient_separation(train_dataset: Any, val_dataset: Any) -> None:
    """Ensure train and validation splits do not share patients."""

    train_patient_ids = set(train_dataset.get_patient_ids())
    val_patient_ids = set(val_dataset.get_patient_ids())
    intersection = train_patient_ids.intersection(val_patient_ids)
    print(
        f"Patient Check: Train={len(train_patient_ids)}, Val={len(val_patient_ids)}, "
        f"Intersection={len(intersection)}"
    )
    if intersection:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE: Patients {intersection} found in both Train and Val!"
        )
    print("Patient separation verified.")


def collect_manifest_split_provenance(
    master_manifest_path: Path,
    *,
    records: Sequence[CanonicalRowRecord],
    split: str,
    smart_sampling: bool,
    runtime_normalization_method: str = "NOT_NORMALIZED",
    runtime_vahadane_backend: str = "fixed_source",
) -> dict[str, Any]:
    stage4_split_bundle_id = resolve_stage4_split_bundle_id(records)
    normalization_method, normalization_artifact_id = resolve_runtime_normalization_selection(
        master_manifest_path,
        records,
        runtime_normalization_method=runtime_normalization_method,
    )
    active_vahadane_backend = runtime_vahadane_backend_for_provenance(
        runtime_normalization_method=normalization_method,
        runtime_vahadane_backend=runtime_vahadane_backend,
    )
    manifest_sha256 = hash_file_sha256(master_manifest_path)
    source_signature = _build_manifest_record_source_signature(records)
    selection_signature = source_signature if smart_sampling else None
    selection_mode = "stage7_selected" if smart_sampling else "all_stage4_accepted"
    return {
        "path": str(master_manifest_path),
        "master_manifest_path": str(master_manifest_path),
        "sha256": manifest_sha256,
        "master_manifest_sha256": manifest_sha256,
        "source_signature": source_signature,
        "selection_signature": selection_signature,
        "split": split,
        "row_count": len(records),
        "shard_count": len({str(record.source_hdf5_path) for record in records}),
        "smart_sampling": smart_sampling,
        "smart_sampling_enabled": smart_sampling,
        "smart_sampling_metadata": {"selection_mode": selection_mode},
        "stage4_split_bundle_id": stage4_split_bundle_id,
        "runtime_normalization_method": normalization_method,
        "runtime_vahadane_backend": active_vahadane_backend,
        "selection_mode": selection_mode,
        "normalization_methods": [normalization_method],
        "attrs": {
            "master_manifest_sha256": manifest_sha256,
            "stage4_split_bundle_id": stage4_split_bundle_id,
            "runtime_normalization_method": normalization_method,
            "runtime_vahadane_backend": active_vahadane_backend,
            "normalization_method": normalization_method,
            "normalization_artifact_id": normalization_artifact_id,
        },
    }


def _build_manifest_record_source_signature(records: Sequence[CanonicalRowRecord]) -> str:
    return hash_json_payload(
        {
            "records": [
                {
                    "source_hdf5_path": str(record.source_hdf5_path),
                    "source_row_index": record.source_row_index,
                    "patient_id": record.patient_id,
                    "label": record.label,
                    "filename": record.filename,
                    "split": record.split,
                    "stage4_split_bundle_id": record.stage4_split_bundle_id,
                    "normalization_method": record.normalization_method,
                    "normalization_artifact_id": record.normalization_artifact_id,
                    "sampling_decision": record.sampling_decision,
                    "is_stage7_selected": record.is_stage7_selected,
                }
                for record in records
            ]
        }
    )


def prepare_training_data(  # noqa: PLR0913
    *,
    master_manifest_path: Path,
    local_data_dir: Path,
    smart_sampling: bool,
    use_subset: bool,
    subset_ratio: float,
    seed: int,
    use_artifact_aware_loss: bool,
    runtime_normalization_method: str = "NOT_NORMALIZED",
    runtime_vahadane_backend: str = "fixed_source",
    normalizer_device: torch.device | str = "cpu",
) -> PreparedTrainingData:
    if local_data_dir.exists():
        shutil.rmtree(local_data_dir)
    local_data_dir.mkdir(parents=True, exist_ok=True)

    training_records = load_training_records(
        master_manifest_path,
        smart_sampling=smart_sampling,
    )
    validation_records = load_validation_records(master_manifest_path)
    source_split_name = "TRAIN_SELECTED" if smart_sampling else "TRAIN"
    artifact_coverage_by_filename = (
        load_artifact_coverage_lookup(str(master_manifest_path))
        if use_artifact_aware_loss
        else None
    )

    train_records_for_dataset = tuple(training_records)
    validation_records_for_dataset = tuple(validation_records)
    if use_subset and subset_ratio < 1.0:
        train_subset_source = CanonicalRowHDF5Dataset(
            CanonicalDatasetLayout(records=train_records_for_dataset, local_cache_dir=None),
            mode="train",
            mask_mode="raw",
        )
        validation_subset_source = CanonicalRowHDF5Dataset(
            CanonicalDatasetLayout(records=validation_records_for_dataset, local_cache_dir=None),
            mode="val",
            mask_mode="raw",
        )
        train_subset = create_stratified_subset_within_patients(
            train_subset_source,
            subset_ratio,
            "train",
            seed=seed,
        )
        validation_subset = create_stratified_subset_within_patients(
            validation_subset_source,
            subset_ratio,
            "val",
            seed=seed,
        )
        train_records_for_dataset = tuple(
            training_records[int(index)] for index in list(train_subset.indices)
        )
        validation_records_for_dataset = tuple(
            validation_records[int(index)] for index in list(validation_subset.indices)
        )

    train_dataset_base = CanonicalRowHDF5Dataset(
        CanonicalDatasetLayout(
            records=train_records_for_dataset,
            local_cache_dir=local_data_dir / "TRAIN",
        ),
        mode="train",
        mask_mode="raw",
        image_normalizer=build_split_stain_normalizer(
            master_manifest_path,
            train_records_for_dataset,
            runtime_normalization_method=runtime_normalization_method,
            runtime_vahadane_backend=runtime_vahadane_backend,
            device=normalizer_device,
        ),
    )
    validation_dataset_base = CanonicalRowHDF5Dataset(
        CanonicalDatasetLayout(
            records=validation_records_for_dataset,
            local_cache_dir=local_data_dir / "VALIDATION",
        ),
        mode="val",
        mask_mode="raw",
        image_normalizer=build_split_stain_normalizer(
            master_manifest_path,
            validation_records_for_dataset,
            runtime_normalization_method=runtime_normalization_method,
            runtime_vahadane_backend=runtime_vahadane_backend,
            device=normalizer_device,
        ),
    )
    train_dataset: Dataset[Any]
    validation_dataset: Dataset[Any]
    if artifact_coverage_by_filename is None:
        train_dataset = train_dataset_base
        validation_dataset = validation_dataset_base
    else:
        train_dataset = ArtifactAwareDatasetView(train_dataset_base, artifact_coverage_by_filename)
        validation_dataset = ArtifactAwareDatasetView(
            validation_dataset_base,
            artifact_coverage_by_filename,
        )

    labels = np.asarray(cast(Any, train_dataset).get_labels())
    class_counts = np.bincount(labels)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return PreparedTrainingData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        sample_weights=sample_weights,
        source_split_name=source_split_name,
        training_provenance=collect_manifest_split_provenance(
            master_manifest_path,
            records=train_records_for_dataset,
            split="TRAIN",
            smart_sampling=smart_sampling,
            runtime_normalization_method=runtime_normalization_method,
            runtime_vahadane_backend=runtime_vahadane_backend,
        ),
        validation_provenance=collect_manifest_split_provenance(
            master_manifest_path,
            records=validation_records_for_dataset,
            split="VALIDATION",
            smart_sampling=False,
            runtime_normalization_method=runtime_normalization_method,
            runtime_vahadane_backend=runtime_vahadane_backend,
        ),
    )


def create_stratified_subset_within_patients(
    full_dataset: Any,
    ratio: float,
    split_name: str = "Unknown",
    seed: int = 42,
) -> Subset[Any]:
    """Create a deterministic per-patient, per-class stratified subset."""

    print(
        f"  Creating a {ratio:.0%} two-level stratified subsample for "
        f"{split_name.upper()} (by patient and class)..."
    )

    indices_by_patient_and_class: dict[Any, dict[Any, list[int]]] = {}
    for idx in range(len(full_dataset)):
        patient_id = full_dataset.patient_ids[idx]
        label = full_dataset.labels[idx]
        patient_groups = indices_by_patient_and_class.setdefault(patient_id, {})
        patient_groups.setdefault(label, []).append(idx)

    print(
        f"    Found {len(indices_by_patient_and_class)} unique patients in the {split_name} split."
    )

    subset_indices: list[int] = []
    generator = torch.Generator().manual_seed(seed)
    for class_groups in indices_by_patient_and_class.values():
        for indices in class_groups.values():
            num_to_sample = int(np.ceil(len(indices) * ratio))
            shuffled_indices = torch.randperm(len(indices), generator=generator).tolist()
            sampled_local_indices = shuffled_indices[:num_to_sample]
            subset_indices.extend(indices[i] for i in sampled_local_indices)

    subset_labels = [full_dataset.labels[idx] for idx in subset_indices]
    if subset_labels:
        subset_counts = np.bincount(subset_labels)
        not_cancer_count = subset_counts[0] if len(subset_counts) > 0 else 0
        cancer_count = subset_counts[1] if len(subset_counts) > 1 else 0
    else:
        not_cancer_count, cancer_count = 0, 0

    print(
        f"    {split_name.title()} subset class counts -> CANCER: {cancer_count}, "
        f"NOT_CANCER: {not_cancer_count}"
    )
    return Subset(full_dataset, subset_indices)
