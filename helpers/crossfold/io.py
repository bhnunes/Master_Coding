from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from helpers.provenance import collect_hdf5_provenance


class NormalizerProtocol(Protocol):
    def transform(self, image_rgb: Any) -> Any: ...


def ensure_split_output_directories(output_dir: Path) -> None:
    """Create the required Stage 5 split folder structure."""

    for split_name in ("TRAIN", "VALIDATION", "TEST"):
        for label_name in ("CANCER", "NOT_CANCER"):
            (output_dir / split_name / label_name).mkdir(parents=True, exist_ok=True)
            (output_dir / split_name / f"{label_name}_MASK").mkdir(parents=True, exist_ok=True)


def process_and_write_image(
    src_path: str,
    dest_path: str,
    normalizer: NormalizerProtocol | None,
) -> tuple[bool, str | None]:
    """Read, optionally normalize, and write one image file."""

    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        if normalizer is None:
            shutil.copy2(src_path, dest_path)
            return True, None

        image_bgr = cv2.imread(src_path)
        if image_bgr is None:
            raise OSError(f"Could not read image: {src_path}")
        if normalizer is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            normalized_rgb = normalizer.transform(image_rgb)
            output_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
        else:
            output_bgr = image_bgr
        success = cv2.imwrite(dest_path, output_bgr)
        if not success:
            raise OSError(f"Failed writing to: {dest_path}")
        return True, None
    except Exception as error:
        return False, f"{Path(src_path).name}: {error}"


def move_file(src_path: str, dest_path: str) -> tuple[bool, str | None, bool]:
    """Move one file and report whether a shutil fallback was required."""

    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            Path(src_path).replace(dest_path)
            return True, None, False
        except OSError:
            shutil.move(src_path, dest_path)
            return True, None, True
    except Exception as error:
        return False, f"{Path(src_path).name}: {error}", False


def process_and_write_split_files(
    split_df: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    normalizer: NormalizerProtocol | None,
    normalization_method: str,
    *,
    allow_destructive_move: bool = False,
) -> int:
    """Write one split to disk while preserving image/mask pairing."""

    if split_df.empty:
        logging.info("Skipping empty split: %s", split_name)
        return 0
    destructive_mode = normalization_method == "NOT_NORMALIZED" and allow_destructive_move
    if destructive_mode:
        logging.warning(
            "[DESTRUCTIVE MODE] NOT_NORMALIZED => moving files into split folders for %s. "
            "Original dataset folders will be modified.",
            split_name,
        )
    logging.info("Writing %s items for %s...", len(split_df), split_name)
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    rows_iter = split_df.itertuples(index=False)
    errors: list[str] = []
    move_fallbacks = 0

    def process_one(row: Any) -> tuple[bool, str | None, bool]:
        try:
            label_dir = "CANCER" if int(row.label) == 1 else "NOT_CANCER"
            image_dest = output_dir / split_name / label_dir / row.filename
            mask_dest = output_dir / split_name / f"{label_dir}_MASK" / row.filename
            if destructive_mode:
                image_ok, image_error, image_fallback = move_file(row.image_path, str(image_dest))
                if not image_ok:
                    return False, f"IMG move failed | {image_error}", image_fallback
                mask_ok, mask_error, mask_fallback = move_file(row.mask_path, str(mask_dest))
                if not mask_ok:
                    return False, f"MSK move failed | {mask_error}", image_fallback or mask_fallback
                return True, None, image_fallback or mask_fallback

            image_ok, image_error = process_and_write_image(
                row.image_path, str(image_dest), normalizer
            )
            if not image_ok:
                return False, f"IMG write failed | {image_error}", False
            mask_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.mask_path, mask_dest)
            return True, None, False
        except Exception as error:
            return False, f"{row.filename}: {error}", False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_one, rows_iter, chunksize=256)
        for ok, message, used_fallback in tqdm(
            results,
            total=len(split_df),
            desc=f"Write {split_name}",
            leave=False,
        ):
            if used_fallback:
                move_fallbacks += 1
            if not ok and message:
                errors.append(message)

    if errors:
        for error in errors[:30]:
            logging.error("Write error: %s", error)
        raise RuntimeError(f"Failed to process {len(errors)} items for split {split_name}.")
    return move_fallbacks


def verify_split_integrity(output_dir: Path, split_name: str) -> None:
    """Ensure each written split still has exact image/mask filename parity."""

    split_dir = output_dir / split_name
    if not split_dir.is_dir():
        logging.warning("Split dir missing (skip integrity): %s", split_dir)
        return
    for label_name in ("CANCER", "NOT_CANCER"):
        image_dir = split_dir / label_name
        mask_dir = split_dir / f"{label_name}_MASK"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        image_files = {path.name for path in image_dir.iterdir() if path.suffix.lower() == ".png"}
        mask_files = {path.name for path in mask_dir.iterdir() if path.suffix.lower() == ".png"}
        if image_files != mask_files:
            raise ValueError(
                f"Integrity FAILED for {split_name}/{label_name}: image/mask filenames differ."
            )
    logging.info("Integrity PASSED: %s", split_name)


def _build_hdf5_split_signature(
    split_df: pd.DataFrame,
    *,
    source_hdf5_path: Path,
    normalization_method: str,
) -> str:
    payload = {
        "source_hdf5": collect_hdf5_provenance(source_hdf5_path),
        "normalization_method": normalization_method,
        "rows": split_df[["filename", "patient_id", "label", "source_row_index"]].to_dict(
            "records"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_existing_split_hdf5(output_path: Path, expected_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        if handle.attrs.get("source_signature") != expected_signature:
            raise ValueError(
                f"Existing split HDF5 '{output_path}' does not match the current split inputs. "
                "Enable overwrite or remove the stale file."
            )
    return output_path


def _normalize_image_array(
    image: npt.NDArray[np.uint8],
    normalizer: NormalizerProtocol | None,
) -> npt.NDArray[np.uint8]:
    if normalizer is None:
        return image.astype(np.uint8, copy=False)
    transformed = normalizer.transform(image)
    return np.asarray(transformed, dtype=np.uint8)


def write_split_hdf5(
    *,
    split_df: pd.DataFrame,
    source_hdf5_path: Path,
    output_path: Path,
    normalizer: NormalizerProtocol | None,
    normalization_method: str,
    overwrite: bool,
) -> Path:
    if split_df.empty:
        raise ValueError(f"Cannot create '{output_path.name}' from an empty split dataframe.")

    source_signature = _build_hdf5_split_signature(
        split_df,
        source_hdf5_path=source_hdf5_path,
        normalization_method=normalization_method,
    )
    if output_path.exists() and not overwrite:
        return _validate_existing_split_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    ordered_split_df = split_df.reset_index(drop=True)

    with (
        h5py.File(source_hdf5_path, "r") as source_handle,
        h5py.File(output_path, "w") as dest_handle,
    ):
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["source_hdf5_sha256"] = collect_hdf5_provenance(source_hdf5_path)[
            "sha256"
        ]

        first_index = int(ordered_split_df.iloc[0]["source_row_index"])
        first_image = np.asarray(source_handle["images"][first_index])
        first_mask = np.asarray(source_handle["masks"][first_index])
        images = dest_handle.create_dataset(
            "images",
            shape=(len(ordered_split_df),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(ordered_split_df),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = dest_handle.create_dataset("labels", shape=(len(ordered_split_df),), dtype="uint8")
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(ordered_split_df),), dtype="int32"
        )
        filenames = dest_handle.create_dataset(
            "filenames", shape=(len(ordered_split_df),), dtype=str_dtype
        )

        for output_index, row in enumerate(ordered_split_df.itertuples(index=False)):
            source_index = int(row.source_row_index)
            image = np.asarray(source_handle["images"][source_index], dtype=np.uint8)
            mask = np.asarray(source_handle["masks"][source_index], dtype=np.uint8)
            images[output_index] = _normalize_image_array(image, normalizer)
            masks[output_index] = mask
            labels[output_index] = int(row.label)
            patient_ids[output_index] = int(row.patient_id)
            filenames[output_index] = str(row.filename)

    return output_path


def verify_split_hdf5_integrity(output_path: Path, split_df: pd.DataFrame) -> None:
    ordered_split_df = split_df.reset_index(drop=True)
    with h5py.File(output_path, "r") as handle:
        if len(handle["labels"]) != len(ordered_split_df):
            raise ValueError(
                "Integrity FAILED for "
                f"{output_path.name}: row count does not match split dataframe."
            )
        observed = [
            {
                "label": int(handle["labels"][index]),
                "patient_id": int(handle["patient_ids"][index]),
                "filename": (
                    handle["filenames"][index].decode("utf-8")
                    if isinstance(handle["filenames"][index], bytes)
                    else str(handle["filenames"][index])
                ),
            }
            for index in range(len(ordered_split_df))
        ]
    expected = ordered_split_df[["label", "patient_id", "filename"]].to_dict("records")
    if observed != expected:
        raise ValueError(
            "Integrity FAILED for "
            f"{output_path.name}: HDF5 row metadata does not match split dataframe."
        )
