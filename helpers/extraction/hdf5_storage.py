from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt


def _signature_hexdigest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_rgb_image(image_path: Path) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read patch image: {image_path}")
    return cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _read_binary_mask(mask_path: Path) -> npt.NDArray[np.uint8]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read patch mask: {mask_path}")
    return cast(npt.NDArray[np.uint8], np.where(mask > 0, 1, 0).astype(np.uint8))


def _paths_for_record(
    record: dict[str, Any],
    *,
    cancer_folder: Path,
    not_cancer_folder: Path,
    cancer_mask_folder: Path,
    not_cancer_mask_folder: Path,
) -> tuple[Path, Path]:
    filename = str(record["filename"])
    label = int(record["label"])
    if label == 1:
        return cancer_folder / filename, cancer_mask_folder / filename
    return not_cancer_folder / filename, not_cancer_mask_folder / filename


def _resolve_patch_arrays(
    record: dict[str, Any],
    *,
    image_path: Path,
    mask_path: Path,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    image_array = record.get("_image_array")
    mask_array = record.get("_mask_array")
    if image_array is not None and mask_array is not None:
        return (
            cast(npt.NDArray[np.uint8], np.asarray(image_array, dtype=np.uint8)),
            cast(npt.NDArray[np.uint8], np.asarray(mask_array, dtype=np.uint8)),
        )
    return _read_rgb_image(image_path), _read_binary_mask(mask_path)


def _logical_hdf5_ref(output_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{output_path}::{dataset_name}[{row_index}]"


def write_slide_patch_dataset_hdf5(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    cancer_folder: Path,
    not_cancer_folder: Path,
    cancer_mask_folder: Path,
    not_cancer_mask_folder: Path,
) -> Path | None:
    if not records:
        if output_path.exists():
            output_path.unlink()
        return None

    resolved_rows: list[dict[str, Any]] = []
    for record in records:
        image_path, mask_path = _paths_for_record(
            record,
            cancer_folder=cancer_folder,
            not_cancer_folder=not_cancer_folder,
            cancer_mask_folder=cancer_mask_folder,
            not_cancer_mask_folder=not_cancer_mask_folder,
        )
        image_array, mask_array = _resolve_patch_arrays(
            record,
            image_path=image_path,
            mask_path=mask_path,
        )
        resolved_rows.append(
            {
                "record": record,
                "image_path": image_path,
                "mask_path": mask_path,
                "image": image_array,
                "mask": mask_array,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    first_image = resolved_rows[0]["image"]
    first_mask = resolved_rows[0]["mask"]

    with h5py.File(output_path, "w") as handle:
        images = handle.create_dataset(
            "images",
            shape=(len(resolved_rows),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = handle.create_dataset(
            "masks",
            shape=(len(resolved_rows),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = handle.create_dataset("labels", shape=(len(resolved_rows),), dtype="uint8")
        patient_ids = handle.create_dataset(
            "patient_ids", shape=(len(resolved_rows),), dtype="int32"
        )
        filenames = handle.create_dataset("filenames", shape=(len(resolved_rows),), dtype=str_dtype)
        slide_ids = handle.create_dataset("slide_ids", shape=(len(resolved_rows),), dtype=str_dtype)
        source_image_paths = handle.create_dataset(
            "source_image_paths", shape=(len(resolved_rows),), dtype=str_dtype
        )
        source_mask_paths = handle.create_dataset(
            "source_mask_paths", shape=(len(resolved_rows),), dtype=str_dtype
        )
        signature_rows: list[dict[str, Any]] = []

        for index, row in enumerate(resolved_rows):
            record = row["record"]
            image_path = cast(Path, row["image_path"])
            mask_path = cast(Path, row["mask_path"])
            image = cast(npt.NDArray[np.uint8], row["image"])
            mask = cast(npt.NDArray[np.uint8], row["mask"])
            images[index] = image
            masks[index] = mask
            labels[index] = int(record["label"])
            patient_ids[index] = int(record["patient_id"])
            filenames[index] = str(record["filename"])
            slide_ids[index] = str(record["slide_id"])
            source_image_paths[index] = _logical_hdf5_ref(output_path, "images", index)
            source_mask_paths[index] = _logical_hdf5_ref(output_path, "masks", index)
            signature_rows.append(
                {
                    "filename": str(record["filename"]),
                    "label": int(record["label"]),
                    "patient_id": int(record["patient_id"]),
                    "slide_id": str(record["slide_id"]),
                    "source_image_path": _logical_hdf5_ref(output_path, "images", index),
                    "source_mask_path": _logical_hdf5_ref(output_path, "masks", index),
                    "png_image_export_path": record.get("_png_image_export_path"),
                    "png_mask_export_path": record.get("_png_mask_export_path"),
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
            )

        handle.attrs["source_signature"] = _signature_hexdigest({"rows": signature_rows})

    return output_path
