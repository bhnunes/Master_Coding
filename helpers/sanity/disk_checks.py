from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from helpers.sanity.models import EXPECTED_MASK_VALUES, LABEL_DIR, MASK_DIR, CheckResult


class PairInspection(TypedDict):
    image_shape: tuple[int, ...] | None
    mask_shape: tuple[int, ...] | None
    mask_unique_values: tuple[int, ...] | None
    mask_has_positive_pixels: bool | None


_MASK_INSPECTION_CACHE: dict[str, PairInspection] = {}


def _is_hdf5_manifest(manifest_split: pd.DataFrame) -> bool:
    return {"relative_hdf5_path", "hdf5_row_index"}.issubset(manifest_split.columns)


@lru_cache(maxsize=8192)
def inspect_image_mask_pair(image_path: str, mask_path: str) -> PairInspection:
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        result: PairInspection = {
            "image_shape": None,
            "mask_shape": None,
            "mask_unique_values": None,
            "mask_has_positive_pixels": None,
        }
        _MASK_INSPECTION_CACHE[mask_path] = {
            "image_shape": None,
            "mask_shape": None,
            "mask_unique_values": None,
            "mask_has_positive_pixels": None,
        }
        return result
    mask_inspection: PairInspection = {
        "image_shape": None,
        "mask_shape": tuple(int(value) for value in mask.shape[:2]),
        "mask_unique_values": tuple(int(value) for value in np.unique(mask).tolist()),
        "mask_has_positive_pixels": bool((mask > 0).any()),
    }
    _MASK_INSPECTION_CACHE[mask_path] = mask_inspection
    return {
        "image_shape": tuple(int(value) for value in image.shape[:2]),
        "mask_shape": mask_inspection["mask_shape"],
        "mask_unique_values": mask_inspection["mask_unique_values"],
        "mask_has_positive_pixels": mask_inspection["mask_has_positive_pixels"],
    }


@lru_cache(maxsize=8192)
def inspect_mask(mask_path: str) -> PairInspection:
    cached = _MASK_INSPECTION_CACHE.get(mask_path)
    if cached is not None:
        return cached
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        result: PairInspection = {
            "image_shape": None,
            "mask_shape": None,
            "mask_unique_values": None,
            "mask_has_positive_pixels": None,
        }
        _MASK_INSPECTION_CACHE[mask_path] = result
        return result
    result = {
        "image_shape": None,
        "mask_shape": tuple(int(value) for value in mask.shape[:2]),
        "mask_unique_values": tuple(int(value) for value in np.unique(mask).tolist()),
        "mask_has_positive_pixels": bool((mask > 0).any()),
    }
    _MASK_INSPECTION_CACHE[mask_path] = result
    return result


@lru_cache(maxsize=8192)
def inspect_hdf5_row(hdf5_path: str, row_index: int) -> PairInspection:
    with h5py.File(hdf5_path, "r") as handle:
        if row_index < 0 or row_index >= len(handle["labels"]):
            return {
                "image_shape": None,
                "mask_shape": None,
                "mask_unique_values": None,
                "mask_has_positive_pixels": None,
            }
        image = np.asarray(handle["images"][row_index])
        mask = np.asarray(handle["masks"][row_index])
    return {
        "image_shape": tuple(int(value) for value in image.shape[:2]),
        "mask_shape": tuple(int(value) for value in mask.shape[:2]),
        "mask_unique_values": tuple(int(value) for value in np.unique(mask).tolist()),
        "mask_has_positive_pixels": bool((mask > 0).any()),
    }


def _list_png_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.name for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png"
    }


def _sample_df(manifest_split: pd.DataFrame, sample_n: int, *, random_state: int) -> pd.DataFrame:
    if manifest_split.empty or len(manifest_split) <= sample_n:
        return manifest_split
    return manifest_split.sample(n=sample_n, random_state=random_state)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_manifest_disk_parity(
    manifest_split: pd.DataFrame, base_dir: Path, split: str
) -> CheckResult:
    if _is_hdf5_manifest(manifest_split):
        return _check_hdf5_manifest_disk_parity(manifest_split, base_dir)
    errors: list[str] = []
    for label in (0, 1):
        manifest_filenames = set(
            manifest_split.loc[manifest_split["label"] == label, "filename"].astype(str).tolist()
        )
        image_dir = base_dir / split / LABEL_DIR[label]
        mask_dir = base_dir / split / MASK_DIR[label]
        image_filenames = _list_png_names(image_dir)
        mask_filenames = _list_png_names(mask_dir)
        if image_filenames != manifest_filenames:
            manifest_only = sorted(manifest_filenames - image_filenames)[:10]
            disk_only = sorted(image_filenames - manifest_filenames)[:10]
            errors.append(
                f"{LABEL_DIR[label]} mismatch: manifest_only={manifest_only} disk_only={disk_only}"
            )
        if mask_filenames != manifest_filenames:
            manifest_only = sorted(manifest_filenames - mask_filenames)[:10]
            disk_only = sorted(mask_filenames - manifest_filenames)[:10]
            errors.append(
                f"{MASK_DIR[label]} mismatch: manifest_only={manifest_only} disk_only={disk_only}"
            )
        if image_filenames != mask_filenames:
            errors.append(
                f"Filename parity mismatch between {LABEL_DIR[label]} and {MASK_DIR[label]}: "
                f"image_only={sorted(image_filenames - mask_filenames)[:10]} "
                f"mask_only={sorted(mask_filenames - image_filenames)[:10]}"
            )
    if errors:
        return CheckResult("FAIL", "; ".join(errors))
    return CheckResult("PASS", "Manifest and disk filenames match exactly for images and masks.")


def _check_hdf5_manifest_disk_parity(manifest_split: pd.DataFrame, base_dir: Path) -> CheckResult:
    errors: list[str] = []
    grouped = manifest_split.groupby("relative_hdf5_path", sort=True)
    for relative_hdf5_path, group in grouped:
        hdf5_path = base_dir / str(relative_hdf5_path)
        if not hdf5_path.is_file():
            errors.append(f"Missing HDF5 file referenced by manifest: {relative_hdf5_path}")
            continue
        ordered_group = group.sort_values("hdf5_row_index", kind="mergesort").reset_index(drop=True)
        with h5py.File(hdf5_path, "r") as handle:
            if len(handle["labels"]) != len(ordered_group):
                errors.append(
                    "HDF5 row count mismatch for "
                    f"{relative_hdf5_path}: manifest={len(ordered_group)} "
                    f"hdf5={len(handle['labels'])}"
                )
                continue
            for row in ordered_group.itertuples(index=False):
                row_index = int(row.hdf5_row_index)
                filename = handle["filenames"][row_index]
                decoded_filename = (
                    filename.decode("utf-8") if isinstance(filename, bytes) else str(filename)
                )
                if decoded_filename != str(row.filename):
                    errors.append(
                        "Filename mismatch at "
                        f"{relative_hdf5_path}[{row_index}]: "
                        f"manifest={row.filename} hdf5={decoded_filename}"
                    )
                if int(handle["labels"][row_index]) != int(row.label):
                    errors.append(
                        f"Label mismatch at {relative_hdf5_path}[{row_index}] for {row.filename}"
                    )
                if "patient_id" in ordered_group.columns and int(
                    handle["patient_ids"][row_index]
                ) != int(row.patient_id):
                    errors.append(
                        f"Patient mismatch at {relative_hdf5_path}[{row_index}] for {row.filename}"
                    )
    if errors:
        return CheckResult("FAIL", "; ".join(errors[:10]))
    return CheckResult("PASS", "Manifest and HDF5 row metadata match exactly.")


def check_paths_exist_and_relative(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    sample_n: int = 1000,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    if _is_hdf5_manifest(manifest_split):
        sample_df = _sample_df(manifest_split, sample_n, random_state=42)
        hdf5_missing: list[str] = []
        hdf5_absolute_paths: list[str] = []
        for row in sample_df.itertuples(index=False):
            relative_hdf5_path = str(row.relative_hdf5_path)
            if Path(relative_hdf5_path).is_absolute():
                hdf5_absolute_paths.append(str(row.filename))
            if not (base_dir / relative_hdf5_path).is_file():
                hdf5_missing.append(str(row.filename))
        if hdf5_absolute_paths:
            return CheckResult(
                "FAIL",
                f"Found absolute HDF5 paths in manifest: {hdf5_absolute_paths[:10]}",
            )
        if hdf5_missing:
            return CheckResult(
                "FAIL",
                f"Missing HDF5 files referenced by manifest: {hdf5_missing[:20]}",
            )
        return CheckResult("PASS", f"Sampled {len(sample_df)} rows: relative HDF5 paths exist.")
    sample_df = _sample_df(manifest_split, sample_n, random_state=42)
    missing: list[str] = []
    absolute_paths: list[str] = []
    for row in sample_df.itertuples(index=False):
        image_rel = str(row.relative_path_image)
        mask_rel = str(row.relative_path_mask)
        if Path(image_rel).is_absolute() or Path(mask_rel).is_absolute():
            absolute_paths.append(str(row.filename))
        if not (base_dir / image_rel).is_file() or not (base_dir / mask_rel).is_file():
            missing.append(str(row.filename))
    if absolute_paths:
        return CheckResult("FAIL", f"Found absolute paths in manifest: {absolute_paths[:10]}")
    if missing:
        return CheckResult("FAIL", f"Missing files referenced by manifest: {missing[:20]}")
    return CheckResult(
        "PASS", f"Sampled {len(sample_df)} rows: relative paths exist for image and mask."
    )


def check_decode_and_shapes(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
    sample_n: int = 1000,
    *,
    full_scan: bool = False,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    if _is_hdf5_manifest(manifest_split):
        scan_df = (
            manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=123)
        )
        hdf5_mismatches: list[str] = []
        hdf5_unreadable: list[str] = []
        for row in tqdm(
            scan_df.itertuples(index=False),
            total=len(scan_df),
            desc=f"{split}: decoding",
            leave=False,
        ):
            inspection = inspect_hdf5_row(
                str(base_dir / str(row.relative_hdf5_path)),
                int(row.hdf5_row_index),
            )
            if inspection["image_shape"] is None or inspection["mask_shape"] is None:
                hdf5_unreadable.append(str(row.filename))
                continue
            if inspection["image_shape"] != inspection["mask_shape"]:
                hdf5_mismatches.append(str(row.filename))
        if hdf5_unreadable:
            return CheckResult(
                "FAIL",
                f"Unreadable HDF5 image/mask rows: {hdf5_unreadable[:10]}",
            )
        if hdf5_mismatches:
            return CheckResult(
                "FAIL",
                f"HDF5 image/mask shape mismatches: {hdf5_mismatches[:10]}",
            )
        return CheckResult(
            "PASS",
            (
                f"{'Full-scan' if full_scan else 'Sampled'} {len(scan_df)} HDF5 rows: "
                "decodable and shapes match."
            ),
        )
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=123)
    )
    mismatches: list[str] = []
    unreadable: list[str] = []
    for row in tqdm(
        scan_df.itertuples(index=False), total=len(scan_df), desc=f"{split}: decoding", leave=False
    ):
        inspection = inspect_image_mask_pair(
            str(base_dir / str(row.relative_path_image)),
            str(base_dir / str(row.relative_path_mask)),
        )
        if inspection["image_shape"] is None or inspection["mask_shape"] is None:
            unreadable.append(str(row.filename))
            continue
        if inspection["image_shape"] != inspection["mask_shape"]:
            mismatches.append(str(row.filename))
    if unreadable:
        return CheckResult("FAIL", f"Unreadable image/mask pairs: {unreadable[:10]}")
    if mismatches:
        return CheckResult("FAIL", f"Image/mask shape mismatches: {mismatches[:10]}")
    return CheckResult(
        "PASS",
        (
            f"{'Full-scan' if full_scan else 'Sampled'} {len(scan_df)} pairs: "
            "decodable and shapes match."
        ),
    )


def check_mask_pixel_values(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
    sample_n: int = 1000,
    *,
    full_scan: bool = False,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    if _is_hdf5_manifest(manifest_split):
        scan_df = (
            manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=999)
        )
        expected_values = {0, 1}
        hdf5_unexpected: set[int] = set()
        for row in tqdm(
            scan_df.itertuples(index=False),
            total=len(scan_df),
            desc=f"{split}: mask values",
            leave=False,
        ):
            inspection = inspect_hdf5_row(
                str(base_dir / str(row.relative_hdf5_path)),
                int(row.hdf5_row_index),
            )
            if inspection["mask_unique_values"] is None:
                return CheckResult("FAIL", f"Unreadable HDF5 mask encountered for {row.filename}.")
            hdf5_unexpected |= set(inspection["mask_unique_values"]) - expected_values
            if hdf5_unexpected and not full_scan:
                break
        if hdf5_unexpected:
            return CheckResult(
                "FAIL",
                (
                    "Unexpected HDF5 mask pixel values found: "
                    f"{sorted(hdf5_unexpected)}. "
                    f"Expected only {sorted(expected_values)}."
                ),
            )
        return CheckResult(
            "PASS",
            (
                f"{'Full-scan' if full_scan else 'Sampled'} HDF5 masks: "
                f"pixel values are only {sorted(expected_values)}."
            ),
        )
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=999)
    )
    unexpected: set[int] = set()
    for row in tqdm(
        scan_df.itertuples(index=False),
        total=len(scan_df),
        desc=f"{split}: mask values",
        leave=False,
    ):
        inspection = inspect_mask(str(base_dir / str(row.relative_path_mask)))
        if inspection["mask_unique_values"] is None:
            return CheckResult("FAIL", f"Unreadable mask encountered for {row.filename}.")
        unexpected |= set(inspection["mask_unique_values"]) - EXPECTED_MASK_VALUES
        if unexpected and not full_scan:
            break
    if unexpected:
        return CheckResult(
            "FAIL",
            (
                f"Unexpected mask pixel values found: {sorted(unexpected)}. "
                f"Expected only {sorted(EXPECTED_MASK_VALUES)}."
            ),
        )
    return CheckResult(
        "PASS",
        (
            f"{'Full-scan' if full_scan else 'Sampled'} masks: pixel values are only "
            f"{sorted(EXPECTED_MASK_VALUES)}."
        ),
    )


def check_checksums(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
    mode: str = "SAMPLE",
    sample_n: int = 200,
) -> CheckResult:
    if _is_hdf5_manifest(manifest_split):
        return CheckResult(
            "N/A", "Per-row checksum verification is not defined for HDF5-native manifests."
        )
    if "sha256_image" not in manifest_split.columns or "sha256_mask" not in manifest_split.columns:
        return CheckResult(
            "N/A", "sha256 columns not present in manifest; skipping checksum verification."
        )
    if mode == "OFF":
        return CheckResult("N/A", "Checksum verification disabled.")
    scan_df = (
        manifest_split
        if mode == "FULL"
        else _sample_df(manifest_split, sample_n, random_state=2026)
    )
    mismatches: list[str] = []
    for row in tqdm(
        scan_df.itertuples(index=False), total=len(scan_df), desc=f"{split}: checksums", leave=False
    ):
        image_hash = row.sha256_image
        mask_hash = row.sha256_mask
        if pd.notna(image_hash) and str(image_hash) != sha256_file(
            base_dir / str(row.relative_path_image)
        ):
            mismatches.append(f"{row.filename} (image)")
        if pd.notna(mask_hash) and str(mask_hash) != sha256_file(
            base_dir / str(row.relative_path_mask)
        ):
            mismatches.append(f"{row.filename} (mask)")
    if mismatches:
        return CheckResult(
            "FAIL", f"Checksum mismatch for {len(mismatches)} items. Examples: {mismatches[:10]}"
        )
    return CheckResult(
        "PASS", f"Checksum verification OK ({mode.lower()}): checked {len(scan_df)} pairs."
    )
