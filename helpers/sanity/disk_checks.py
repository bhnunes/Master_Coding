from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from helpers.sanity.models import EXPECTED_MASK_VALUES, LABEL_DIR, MASK_DIR, CheckResult


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
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=123)
    )
    mismatches: list[str] = []
    unreadable: list[str] = []
    for row in tqdm(
        scan_df.itertuples(index=False), total=len(scan_df), desc=f"{split}: decoding", leave=False
    ):
        image = cv2.imread(str(base_dir / str(row.relative_path_image)))
        mask = cv2.imread(str(base_dir / str(row.relative_path_mask)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            unreadable.append(str(row.filename))
            continue
        if image.shape[:2] != mask.shape[:2]:
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
        mask = cv2.imread(str(base_dir / str(row.relative_path_mask)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return CheckResult("FAIL", f"Unreadable mask encountered for {row.filename}.")
        unexpected |= {int(value) for value in np.unique(mask).tolist()} - EXPECTED_MASK_VALUES
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
