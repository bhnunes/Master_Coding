from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt

HDF5_WRITE_TARGET_BATCH_BYTES = 64 * 1024 * 1024
HDF5_WRITE_MAX_BATCH_ROWS = 1024


@dataclass(frozen=True)
class _PreparedHdf5Batch:
    output_slice: slice
    image_batch: npt.NDArray[np.uint8]
    mask_batch: npt.NDArray[np.uint8]
    label_batch: npt.NDArray[np.uint8]
    patient_id_batch: npt.NDArray[np.int32]
    filename_batch: list[str]
    slide_id_batch: list[str]
    source_image_path_batch: list[str]
    source_mask_path_batch: list[str]
    signature_rows: list[dict[str, Any]]


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


def _sha256_uint8_array(value: npt.NDArray[np.uint8]) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous).hexdigest()


def _logical_hdf5_ref(output_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{output_path}::{dataset_name}[{row_index}]"


def _resolve_write_batch_size(
    *,
    first_image: npt.NDArray[np.uint8],
    first_mask: npt.NDArray[np.uint8],
    requested_batch_size: int | None,
) -> int:
    if requested_batch_size is not None:
        return max(1, int(requested_batch_size))

    row_bytes = max(1, int(first_image.nbytes) + int(first_mask.nbytes))
    rows_per_target = max(1, HDF5_WRITE_TARGET_BATCH_BYTES // row_bytes)
    return min(HDF5_WRITE_MAX_BATCH_ROWS, rows_per_target)


def _prepare_hdf5_batch(
    *,
    output_path: Path,
    batch_records: list[dict[str, Any]],
    batch_start: int,
    first_image_shape: tuple[int, ...],
    first_mask_shape: tuple[int, ...],
) -> _PreparedHdf5Batch:
    batch_len = len(batch_records)
    batch_stop = batch_start + batch_len
    image_batch = np.empty((batch_len, *first_image_shape), dtype=np.uint8)
    mask_batch = np.empty((batch_len, *first_mask_shape), dtype=np.uint8)
    label_batch = np.empty((batch_len,), dtype=np.uint8)
    patient_id_batch = np.empty((batch_len,), dtype=np.int32)
    filename_batch: list[str] = []
    slide_id_batch: list[str] = []
    source_image_path_batch: list[str] = []
    source_mask_path_batch: list[str] = []
    signature_rows: list[dict[str, Any]] = []

    for offset, record in enumerate(batch_records):
        index = batch_start + offset
        image, mask = _resolve_patch_arrays(record)
        _require_matching_shape(name="image", value=image, expected_shape=first_image_shape)
        _require_matching_shape(name="mask", value=mask, expected_shape=first_mask_shape)
        source_image_path = _logical_hdf5_ref(output_path, "images", index)
        source_mask_path = _logical_hdf5_ref(output_path, "masks", index)

        image_batch[offset] = image
        mask_batch[offset] = mask
        label_batch[offset] = int(record["label"])
        patient_id_batch[offset] = int(record["patient_id"])
        filename = str(record["filename"])
        slide_id = str(record["slide_id"])
        filename_batch.append(filename)
        slide_id_batch.append(slide_id)
        source_image_path_batch.append(source_image_path)
        source_mask_path_batch.append(source_mask_path)

        signature_rows.append(
            {
                "filename": filename,
                "label": int(record["label"]),
                "patient_id": int(record["patient_id"]),
                "slide_id": slide_id,
                "source_image_path": source_image_path,
                "source_mask_path": source_mask_path,
                "image_sha256": _sha256_uint8_array(image),
                "mask_sha256": _sha256_uint8_array(mask),
            }
        )

    return _PreparedHdf5Batch(
        output_slice=slice(batch_start, batch_stop),
        image_batch=image_batch,
        mask_batch=mask_batch,
        label_batch=label_batch,
        patient_id_batch=patient_id_batch,
        filename_batch=filename_batch,
        slide_id_batch=slide_id_batch,
        source_image_path_batch=source_image_path_batch,
        source_mask_path_batch=source_mask_path_batch,
        signature_rows=signature_rows,
    )


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
    batch_size: int | None = None,
) -> Path | None:
    if not records:
        if output_path.exists():
            output_path.unlink()
        return None

    first_image, first_mask = _resolve_patch_arrays(records[0])
    row_count = len(records)
    resolved_batch_size = _resolve_write_batch_size(
        first_image=first_image,
        first_mask=first_mask,
        requested_batch_size=batch_size,
    )

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

        for batch_start in range(0, row_count, resolved_batch_size):
            prepared_batch = _prepare_hdf5_batch(
                output_path=output_path,
                batch_records=records[batch_start : batch_start + resolved_batch_size],
                batch_start=batch_start,
                first_image_shape=first_image.shape,
                first_mask_shape=first_mask.shape,
            )
            images_dataset[prepared_batch.output_slice] = prepared_batch.image_batch
            masks_dataset[prepared_batch.output_slice] = prepared_batch.mask_batch
            labels_dataset[prepared_batch.output_slice] = prepared_batch.label_batch
            patient_ids_dataset[prepared_batch.output_slice] = prepared_batch.patient_id_batch
            filenames_dataset[prepared_batch.output_slice] = prepared_batch.filename_batch
            slide_ids_dataset[prepared_batch.output_slice] = prepared_batch.slide_id_batch
            source_image_paths_dataset[prepared_batch.output_slice] = (
                prepared_batch.source_image_path_batch
            )
            source_mask_paths_dataset[prepared_batch.output_slice] = (
                prepared_batch.source_mask_path_batch
            )
            signature_rows.extend(prepared_batch.signature_rows)

        handle.attrs["source_signature"] = _signature_hexdigest({"rows": signature_rows})

    return output_path
