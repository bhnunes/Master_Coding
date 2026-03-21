from __future__ import annotations

import atexit
import os
import shutil
import time
from typing import Any, cast

import albumentations as A
import cv2
import h5py
import numpy as np
import numpy.typing as npt
import psutil
import pyarrow.parquet as pq
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import default_collate

from helpers.provenance import hash_file_sha256

NumericArray = npt.NDArray[np.generic]
ArtifactCoverageLookup = dict[str, tuple[float, float, float, float, float]]
ZERO_ARTIFACT_COVERAGE = (0.0, 0.0, 0.0, 0.0, 0.0)
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


def _get_filenames_dataset(handle: h5py.File) -> Any:
    for dataset_name in FILENAME_DATASET_CANDIDATES:
        if dataset_name in handle:
            return handle[dataset_name]
    available = ", ".join(handle.keys())
    raise KeyError(
        "HDF5 file is missing the filename dataset. Expected one of "
        f"{FILENAME_DATASET_CANDIDATES}. Available: {available}"
    )


def load_artifact_coverage_lookup(parquet_path: str) -> ArtifactCoverageLookup:
    """Load filename-keyed artifact coverage vectors from Parquet."""

    parquet_file = pq.ParquetFile(parquet_path)
    available_columns = set(parquet_file.schema.names)
    selected_columns = [
        "filename",
        *[c for c in ARTIFACT_COVERAGE_COLUMNS if c in available_columns],
    ]
    table = parquet_file.read(columns=selected_columns)
    data = table.to_pydict()
    filenames = data.pop("filename", [])
    lookup: ArtifactCoverageLookup = {}
    for index, filename in enumerate(filenames):
        lookup[str(filename)] = cast(
            tuple[float, float, float, float, float],
            tuple(
                float((data.get(column_name) or [0.0] * len(filenames))[index] or 0.0)
                for column_name in ARTIFACT_COVERAGE_COLUMNS
            ),
        )
    return lookup


def get_transforms(mode: str = "train", img_size: int = 224) -> A.Compose:
    """Return the existing augmentation pipeline for train or validation."""

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
            transformed_mask = augmented["mask"]
            if transformed_mask.ndim == 3 and transformed_mask.shape[-1] == 1:
                transformed_mask = transformed_mask.squeeze(-1)
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
