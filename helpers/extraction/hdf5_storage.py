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

    resolved_rows: list[dict[str, Any]] = []
    for record in records:
        image_array, mask_array = _resolve_patch_arrays(record)
        resolved_rows.append(
            {
                "record": record,
                "image": image_array,
                "mask": mask_array,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    images_data = np.stack(
        [cast(npt.NDArray[np.uint8], row["image"]) for row in resolved_rows],
        axis=0,
    )
    masks_data = np.stack(
        [cast(npt.NDArray[np.uint8], row["mask"]) for row in resolved_rows],
        axis=0,
    )
    labels_data = np.fromiter(
        (int(row["record"]["label"]) for row in resolved_rows),
        dtype=np.uint8,
        count=len(resolved_rows),
    )
    patient_ids_data = np.fromiter(
        (int(row["record"]["patient_id"]) for row in resolved_rows),
        dtype=np.int32,
        count=len(resolved_rows),
    )
    filenames_data = np.asarray(
        [str(row["record"]["filename"]) for row in resolved_rows],
        dtype=object,
    )
    slide_ids_data = np.asarray(
        [str(row["record"]["slide_id"]) for row in resolved_rows],
        dtype=object,
    )
    source_image_paths_data = np.asarray(
        [_logical_hdf5_ref(output_path, "images", index) for index in range(len(resolved_rows))],
        dtype=object,
    )
    source_mask_paths_data = np.asarray(
        [_logical_hdf5_ref(output_path, "masks", index) for index in range(len(resolved_rows))],
        dtype=object,
    )

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=images_data,
            compression=compression,
            chunks=True,
        )
        handle.create_dataset(
            "masks",
            data=masks_data,
            compression=compression,
            chunks=True,
        )
        handle.create_dataset("labels", data=labels_data)
        handle.create_dataset("patient_ids", data=patient_ids_data)
        handle.create_dataset("filenames", data=filenames_data, dtype=str_dtype)
        handle.create_dataset("slide_ids", data=slide_ids_data, dtype=str_dtype)
        handle.create_dataset("source_image_paths", data=source_image_paths_data, dtype=str_dtype)
        handle.create_dataset("source_mask_paths", data=source_mask_paths_data, dtype=str_dtype)
        signature_rows: list[dict[str, Any]] = []

        for index, row in enumerate(resolved_rows):
            record = row["record"]
            image = cast(npt.NDArray[np.uint8], row["image"])
            mask = cast(npt.NDArray[np.uint8], row["mask"])
            signature_rows.append(
                {
                    "filename": str(record["filename"]),
                    "label": int(record["label"]),
                    "patient_id": int(record["patient_id"]),
                    "slide_id": str(record["slide_id"]),
                    "source_image_path": source_image_paths_data[index],
                    "source_mask_path": source_mask_paths_data[index],
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
            )

        handle.attrs["source_signature"] = _signature_hexdigest({"rows": signature_rows})

    return output_path
