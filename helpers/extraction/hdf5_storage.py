from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt


def _signature_hexdigest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_patch_arrays(
    record: dict[str, Any],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    image_array = record.get("_image_array")
    mask_array = record.get("_mask_array")
    if image_array is None or mask_array is None:
        raise ValueError(
            "Stage 2 HDF5 writing requires in-memory '_image_array' and '_mask_array' records."
        )
    return (
        cast(npt.NDArray[np.uint8], np.asarray(image_array, dtype=np.uint8)),
        cast(npt.NDArray[np.uint8], np.asarray(mask_array, dtype=np.uint8)),
    )


def _logical_hdf5_ref(output_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{output_path}::{dataset_name}[{row_index}]"


def _require_matching_shape(
    *,
    name: str,
    value: npt.NDArray[np.uint8],
    expected_shape: tuple[int, ...],
) -> None:
    if value.shape != expected_shape:
        raise ValueError(
            f"Stage 2 HDF5 writing requires consistent {name} shapes; "
            f"expected {expected_shape}, got {value.shape}."
        )


def write_slide_patch_dataset_hdf5(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    compression: str | None = "gzip",
) -> Path | None:
    if not records:
        if output_path.exists():
            output_path.unlink()
        return None

    first_image, first_mask = _resolve_patch_arrays(records[0])
    row_count = len(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        images_dataset = handle.create_dataset(
            "images",
            shape=(row_count, *first_image.shape),
            dtype=np.uint8,
            compression=compression,
            chunks=True,
        )
        masks_dataset = handle.create_dataset(
            "masks",
            shape=(row_count, *first_mask.shape),
            dtype=np.uint8,
            compression=compression,
            chunks=True,
        )
        labels_dataset = handle.create_dataset("labels", shape=(row_count,), dtype=np.uint8)
        patient_ids_dataset = handle.create_dataset(
            "patient_ids", shape=(row_count,), dtype=np.int32
        )
        filenames_dataset = handle.create_dataset("filenames", shape=(row_count,), dtype=str_dtype)
        slide_ids_dataset = handle.create_dataset("slide_ids", shape=(row_count,), dtype=str_dtype)
        source_image_paths_dataset = handle.create_dataset(
            "source_image_paths", shape=(row_count,), dtype=str_dtype
        )
        source_mask_paths_dataset = handle.create_dataset(
            "source_mask_paths", shape=(row_count,), dtype=str_dtype
        )
        signature_rows: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            image, mask = _resolve_patch_arrays(record)
            _require_matching_shape(name="image", value=image, expected_shape=first_image.shape)
            _require_matching_shape(name="mask", value=mask, expected_shape=first_mask.shape)
            source_image_path = _logical_hdf5_ref(output_path, "images", index)
            source_mask_path = _logical_hdf5_ref(output_path, "masks", index)

            images_dataset[index] = image
            masks_dataset[index] = mask
            labels_dataset[index] = int(record["label"])
            patient_ids_dataset[index] = int(record["patient_id"])
            filenames_dataset[index] = str(record["filename"])
            slide_ids_dataset[index] = str(record["slide_id"])
            source_image_paths_dataset[index] = source_image_path
            source_mask_paths_dataset[index] = source_mask_path

            signature_rows.append(
                {
                    "filename": str(record["filename"]),
                    "label": int(record["label"]),
                    "patient_id": int(record["patient_id"]),
                    "slide_id": str(record["slide_id"]),
                    "source_image_path": source_image_path,
                    "source_mask_path": source_mask_path,
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
            )

        handle.attrs["source_signature"] = _signature_hexdigest({"rows": signature_rows})

    return output_path
