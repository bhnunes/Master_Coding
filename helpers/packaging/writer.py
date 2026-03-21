from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt

from helpers.packaging.discovery import SampleRecord
from helpers.provenance import hash_file_sha256


def _build_packaging_signature(samples: list[SampleRecord], img_size: int) -> str:
    payload = {
        "img_size": int(img_size),
        "samples": [
            {
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
                "image_sha256": hash_file_sha256(sample.image_path),
                "mask_sha256": hash_file_sha256(sample.mask_path),
                "filename": sample.filename,
                "label": int(sample.label),
                "patient_id": int(sample.patient_id),
            }
            for sample in samples
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_existing_hdf5(output_path: Path, expected_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        existing_signature = handle.attrs.get("source_signature")
        if existing_signature != expected_signature:
            raise ValueError(
                f"Existing HDF5 '{output_path}' does not match the current split inputs. "
                "Enable overwrite or remove the stale file."
            )
    return output_path


def _load_image(image_path: Path, img_size: int) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
    )


def _load_mask(mask_path: Path, img_size: int) -> npt.NDArray[np.uint8]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask file: {mask_path}")
    resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return cast(npt.NDArray[np.uint8], np.where(resized > 0, 1, 0).astype(np.uint8))


def write_split_hdf5(
    output_path: Path,
    samples: list[SampleRecord],
    *,
    img_size: int,
    overwrite: bool,
) -> Path:
    if not samples:
        raise ValueError(f"Cannot create '{output_path.name}' without any samples.")

    source_signature = _build_packaging_signature(samples, img_size)
    if output_path.exists() and not overwrite:
        return _validate_existing_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        handle.attrs["source_signature"] = source_signature
        images = handle.create_dataset(
            "images",
            shape=(len(samples), img_size, img_size, 3),
            dtype="uint8",
        )
        masks = handle.create_dataset(
            "masks",
            shape=(len(samples), img_size, img_size),
            dtype="uint8",
        )
        labels = handle.create_dataset("labels", shape=(len(samples),), dtype="uint8")
        patient_ids = handle.create_dataset("patient_ids", shape=(len(samples),), dtype="int32")
        filenames = handle.create_dataset("filenames", shape=(len(samples),), dtype=str_dtype)

        for index, sample in enumerate(samples):
            images[index] = _load_image(sample.image_path, img_size)
            masks[index] = _load_mask(sample.mask_path, img_size)
            labels[index] = sample.label
            patient_ids[index] = sample.patient_id
            filenames[index] = sample.filename

    return output_path
