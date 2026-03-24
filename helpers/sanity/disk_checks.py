from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from helpers.sanity.models import CheckResult


class PairInspection(TypedDict):
    image_shape: tuple[int, ...] | None
    mask_shape: tuple[int, ...] | None
    mask_unique_values: tuple[int, ...] | None
    mask_has_positive_pixels: bool | None


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


def _sample_df(
    manifest_split: pd.DataFrame,
    sample_n: int,
    *,
    random_state: int,
) -> pd.DataFrame:
    if manifest_split.empty or len(manifest_split) <= sample_n:
        return manifest_split
    return manifest_split.sample(n=sample_n, random_state=random_state)


def check_manifest_disk_parity(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
) -> CheckResult:
    del split
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
                        f"{relative_hdf5_path}[{row_index}]: manifest={row.filename} "
                        f"hdf5={decoded_filename}"
                    )
                if int(handle["labels"][row_index]) != int(row.label):
                    errors.append(
                        f"Label mismatch at {relative_hdf5_path}[{row_index}] for {row.filename}"
                    )
                if int(handle["patient_ids"][row_index]) != int(row.patient_id):
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
    sample_df = _sample_df(manifest_split, sample_n, random_state=42)
    missing: list[str] = []
    absolute_paths: list[str] = []
    for row in sample_df.itertuples(index=False):
        relative_hdf5_path = str(row.relative_hdf5_path)
        if Path(relative_hdf5_path).is_absolute():
            absolute_paths.append(str(row.filename))
        if not (base_dir / relative_hdf5_path).is_file():
            missing.append(str(row.filename))
    if absolute_paths:
        return CheckResult("FAIL", f"Found absolute HDF5 paths in manifest: {absolute_paths[:10]}")
    if missing:
        return CheckResult("FAIL", f"Missing HDF5 files referenced by manifest: {missing[:20]}")
    return CheckResult("PASS", f"Sampled {len(sample_df)} rows: relative HDF5 paths exist.")


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
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=123)
    )
    mismatches: list[str] = []
    unreadable: list[str] = []
    for row in tqdm(
        scan_df.itertuples(index=False), total=len(scan_df), desc=f"{split}: decoding", leave=False
    ):
        inspection = inspect_hdf5_row(
            str(base_dir / str(row.relative_hdf5_path)), int(row.hdf5_row_index)
        )
        if inspection["image_shape"] is None or inspection["mask_shape"] is None:
            unreadable.append(str(row.filename))
            continue
        if inspection["image_shape"] != inspection["mask_shape"]:
            mismatches.append(str(row.filename))
    if unreadable:
        return CheckResult("FAIL", f"Unreadable HDF5 image/mask rows: {unreadable[:10]}")
    if mismatches:
        return CheckResult("FAIL", f"HDF5 image/mask shape mismatches: {mismatches[:10]}")
    return CheckResult(
        "PASS",
        (
            f"{'Full-scan' if full_scan else 'Sampled'} {len(scan_df)} HDF5 rows: "
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
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=999)
    )
    expected_values = {0, 1}
    unexpected: set[int] = set()
    for row in tqdm(
        scan_df.itertuples(index=False),
        total=len(scan_df),
        desc=f"{split}: mask values",
        leave=False,
    ):
        inspection = inspect_hdf5_row(
            str(base_dir / str(row.relative_hdf5_path)), int(row.hdf5_row_index)
        )
        if inspection["mask_unique_values"] is None:
            return CheckResult("FAIL", f"Unreadable HDF5 mask encountered for {row.filename}.")
        unexpected |= set(inspection["mask_unique_values"]) - expected_values
        if unexpected and not full_scan:
            break
    if unexpected:
        return CheckResult(
            "FAIL",
            (
                "Unexpected HDF5 mask pixel values found: "
                f"{sorted(unexpected)}. Expected only {sorted(expected_values)}."
            ),
        )
    return CheckResult(
        "PASS",
        (
            f"{'Full-scan' if full_scan else 'Sampled'} HDF5 masks: "
            f"pixel values are only {sorted(expected_values)}."
        ),
    )
