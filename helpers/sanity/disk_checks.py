from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict, cast

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from helpers.sanity.models import CheckResult


class PairInspection(TypedDict):
    image_shape: tuple[int, ...] | None
    mask_shape: tuple[int, ...] | None
    mask_unique_values: tuple[int, ...] | None
    mask_has_positive_pixels: bool | None


@dataclass(frozen=True)
class IndexedInspection:
    manifest_index: int
    filename: str
    inspection: PairInspection


_HDF5_SCAN_BATCH_SIZE = 512


def _resolve_source_hdf5_path(raw_path: object) -> Path:
    return Path(str(raw_path))


@lru_cache(maxsize=8192)
def inspect_hdf5_row(hdf5_path: str, row_index: int) -> PairInspection:
    with h5py.File(hdf5_path, "r") as handle:
        labels_dataset = cast(Any, handle["labels"])
        images_dataset = cast(Any, handle["images"])
        masks_dataset = cast(Any, handle["masks"])
        if row_index < 0 or row_index >= len(labels_dataset):
            return {
                "image_shape": None,
                "mask_shape": None,
                "mask_unique_values": None,
                "mask_has_positive_pixels": None,
            }
        image = np.asarray(images_dataset[row_index])
        mask = np.asarray(masks_dataset[row_index])
    return {
        "image_shape": tuple(int(value) for value in image.shape[:2]),
        "mask_shape": tuple(int(value) for value in mask.shape[:2]),
        "mask_unique_values": tuple(int(value) for value in np.unique(mask).tolist()),
        "mask_has_positive_pixels": bool((mask > 0).any()),
    }


def _empty_inspection() -> PairInspection:
    return {
        "image_shape": None,
        "mask_shape": None,
        "mask_unique_values": None,
        "mask_has_positive_pixels": None,
    }


def _build_inspections_for_batch(
    image_batch: NDArray[np.uint8],
    mask_batch: NDArray[np.uint8],
    manifest_indexes: list[int],
    filenames: list[str],
) -> list[IndexedInspection]:
    results: list[IndexedInspection] = []
    for offset, (manifest_index, filename) in enumerate(
        zip(manifest_indexes, filenames, strict=True)
    ):
        image = image_batch[offset]
        mask = mask_batch[offset]
        results.append(
            IndexedInspection(
                manifest_index=manifest_index,
                filename=filename,
                inspection={
                    "image_shape": tuple(int(value) for value in image.shape[:2]),
                    "mask_shape": tuple(int(value) for value in mask.shape[:2]),
                    "mask_unique_values": tuple(int(value) for value in np.unique(mask).tolist()),
                    "mask_has_positive_pixels": bool((mask > 0).any()),
                },
            )
        )
    return results


def collect_hdf5_row_inspections(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
) -> dict[int, IndexedInspection]:
    del split
    if manifest_split.empty:
        return {}
    grouped = manifest_split.groupby("source_hdf5_path", sort=True)
    collected: dict[int, IndexedInspection] = {}
    for raw_hdf5_path, group in grouped:
        hdf5_path = _resolve_source_hdf5_path(raw_hdf5_path)
        if not hdf5_path.is_file():
            for manifest_index, filename in zip(
                group.index.tolist(), group["filename"].astype(str), strict=True
            ):
                collected[int(manifest_index)] = IndexedInspection(
                    manifest_index=int(manifest_index),
                    filename=filename,
                    inspection=_empty_inspection(),
                )
            continue
        ordered_group = group.sort_values("source_row_index", kind="mergesort")
        with h5py.File(hdf5_path, "r") as handle:
            image_dataset = cast(Any, handle["images"])
            mask_dataset = cast(Any, handle["masks"])
            labels_dataset = cast(Any, handle["labels"])
            label_count = len(labels_dataset)
            rows = ordered_group["source_row_index"].astype(int).tolist()
            manifest_indexes = [int(value) for value in ordered_group.index.tolist()]
            filenames = ordered_group["filename"].astype(str).tolist()
            start = 0
            while start < len(rows):
                batch_start = rows[start]
                batch_end = batch_start + 1
                end = start + 1
                while (
                    end < len(rows)
                    and rows[end] == batch_end
                    and (end - start) < _HDF5_SCAN_BATCH_SIZE
                ):
                    batch_end += 1
                    end += 1
                batch_rows = rows[start:end]
                batch_indexes = manifest_indexes[start:end]
                batch_filenames = filenames[start:end]
                if batch_rows[0] < 0 or batch_rows[-1] >= label_count:
                    for manifest_index, filename in zip(
                        batch_indexes,
                        batch_filenames,
                        strict=True,
                    ):
                        collected[manifest_index] = IndexedInspection(
                            manifest_index=manifest_index,
                            filename=filename,
                            inspection=_empty_inspection(),
                        )
                    start = end
                    continue
                image_batch = np.asarray(image_dataset[batch_rows[0] : batch_rows[-1] + 1])
                mask_batch = np.asarray(mask_dataset[batch_rows[0] : batch_rows[-1] + 1])
                batch_results = _build_inspections_for_batch(
                    image_batch,
                    mask_batch,
                    batch_indexes,
                    batch_filenames,
                )
                for result in batch_results:
                    collected[result.manifest_index] = result
                start = end
    return collected


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
    grouped = manifest_split.groupby("source_hdf5_path", sort=True)
    for raw_hdf5_path, group in grouped:
        hdf5_path = _resolve_source_hdf5_path(raw_hdf5_path)
        hdf5_label = str(raw_hdf5_path)
        if not hdf5_path.is_file():
            errors.append(f"Missing HDF5 file referenced by manifest: {hdf5_label}")
            continue
        ordered_group = group.sort_values("source_row_index", kind="mergesort")
        with h5py.File(hdf5_path, "r") as handle:
            labels_dataset = cast(Any, handle["labels"])
            patient_ids_dataset = cast(Any, handle["patient_ids"])
            filenames_dataset = cast(Any, handle["filenames"])
            for record in ordered_group.to_dict("records"):
                row_index = int(record["source_row_index"])
                if row_index < 0 or row_index >= len(labels_dataset):
                    errors.append(f"Row index out of bounds at {hdf5_label}[{row_index}]")
                    continue
                filename = filenames_dataset[row_index]
                decoded_filename = (
                    filename.decode("utf-8") if isinstance(filename, bytes) else str(filename)
                )
                if decoded_filename != str(record["filename"]):
                    errors.append(
                        "Filename mismatch at "
                        f"{hdf5_label}[{row_index}]: manifest={record['filename']} "
                        f"hdf5={decoded_filename}"
                    )
                if int(labels_dataset[row_index]) != int(record["label"]):
                    errors.append(
                        f"Label mismatch at {hdf5_label}[{row_index}] for {record['filename']}"
                    )
                if int(patient_ids_dataset[row_index]) != int(record["patient_id"]):
                    errors.append(
                        f"Patient mismatch at {hdf5_label}[{row_index}] for {record['filename']}"
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
    for record in sample_df.to_dict("records"):
        hdf5_path = _resolve_source_hdf5_path(record["source_hdf5_path"])
        if not hdf5_path.is_file():
            missing.append(str(record["filename"]))
    if missing:
        return CheckResult("FAIL", f"Missing HDF5 files referenced by manifest: {missing[:20]}")
    return CheckResult("PASS", f"Sampled {len(sample_df)} rows: referenced HDF5 paths exist.")


def check_decode_and_shapes(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
    sample_n: int = 1000,
    *,
    full_scan: bool = False,
    row_inspections: dict[int, IndexedInspection] | None = None,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=123)
    ).reset_index()
    mismatches: list[str] = []
    unreadable: list[str] = []
    for record in tqdm(
        scan_df.to_dict("records"), total=len(scan_df), desc=f"{split}: decoding", leave=False
    ):
        if row_inspections is not None:
            inspection = row_inspections[int(record["index"])].inspection
        else:
            inspection = inspect_hdf5_row(
                str(_resolve_source_hdf5_path(record["source_hdf5_path"])),
                int(record["source_row_index"]),
            )
        if inspection["image_shape"] is None or inspection["mask_shape"] is None:
            unreadable.append(str(record["filename"]))
            continue
        if inspection["image_shape"] != inspection["mask_shape"]:
            mismatches.append(str(record["filename"]))
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
    row_inspections: dict[int, IndexedInspection] | None = None,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    scan_df = (
        manifest_split if full_scan else _sample_df(manifest_split, sample_n, random_state=999)
    ).reset_index()
    expected_values = {0, 1}
    unexpected: set[int] = set()
    for record in tqdm(
        scan_df.to_dict("records"),
        total=len(scan_df),
        desc=f"{split}: mask values",
        leave=False,
    ):
        if row_inspections is not None:
            inspection = row_inspections[int(record["index"])].inspection
        else:
            inspection = inspect_hdf5_row(
                str(_resolve_source_hdf5_path(record["source_hdf5_path"])),
                int(record["source_row_index"]),
            )
        if inspection["mask_unique_values"] is None:
            return CheckResult(
                "FAIL", f"Unreadable HDF5 mask encountered for {record['filename']}."
            )
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
