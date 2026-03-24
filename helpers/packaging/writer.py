from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt

from helpers.packaging.discovery import SampleRecord
from helpers.provenance import hash_file_sha256


def _signature_hexdigest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    return _signature_hexdigest(payload)


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


def _read_file_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Could not read file bytes: {path}")
    return data


def _decode_image_bytes(
    image_bytes: bytes, image_path: Path, img_size: int
) -> npt.NDArray[np.uint8]:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
    )


def _decode_mask_bytes(mask_bytes: bytes, mask_path: Path, img_size: int) -> npt.NDArray[np.uint8]:
    encoded = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask file: {mask_path}")
    resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return cast(npt.NDArray[np.uint8], np.where(resized > 0, 1, 0).astype(np.uint8))


def _sample_signature_entry(
    sample: SampleRecord, image_sha256: str, mask_sha256: str
) -> dict[str, Any]:
    return {
        "image_path": str(sample.image_path),
        "mask_path": str(sample.mask_path),
        "image_sha256": image_sha256,
        "mask_sha256": mask_sha256,
        "filename": sample.filename,
        "label": int(sample.label),
        "patient_id": int(sample.patient_id),
    }


def _build_packaging_signature_from_rows(
    rows: list[dict[str, Any]],
    *,
    img_size: int,
) -> str:
    return _signature_hexdigest({"img_size": int(img_size), "samples": rows})


def write_patch_dataset_hdf5(
    output_path: Path,
    samples: list[SampleRecord],
    *,
    img_size: int,
    overwrite: bool,
) -> Path:
    if not samples:
        raise ValueError(f"Cannot create '{output_path.name}' without any samples.")

    if output_path.exists() and not overwrite:
        source_signature = _build_packaging_signature(samples, img_size)
        return _validate_existing_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
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
        source_image_paths = handle.create_dataset(
            "source_image_paths", shape=(len(samples),), dtype=str_dtype
        )
        source_mask_paths = handle.create_dataset(
            "source_mask_paths", shape=(len(samples),), dtype=str_dtype
        )
        signature_rows: list[dict[str, Any]] = []

        for index, sample in enumerate(samples):
            image_bytes = _read_file_bytes(sample.image_path)
            mask_bytes = _read_file_bytes(sample.mask_path)
            images[index] = _decode_image_bytes(image_bytes, sample.image_path, img_size)
            masks[index] = _decode_mask_bytes(mask_bytes, sample.mask_path, img_size)
            labels[index] = sample.label
            patient_ids[index] = sample.patient_id
            filenames[index] = sample.filename
            source_image_paths[index] = str(sample.image_path)
            source_mask_paths[index] = str(sample.mask_path)
            signature_rows.append(
                _sample_signature_entry(
                    sample,
                    hashlib.sha256(image_bytes).hexdigest(),
                    hashlib.sha256(mask_bytes).hexdigest(),
                )
            )

        handle.attrs["source_signature"] = _build_packaging_signature_from_rows(
            signature_rows,
            img_size=img_size,
        )

    return output_path
