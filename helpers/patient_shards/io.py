from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from helpers.crossfold.logging import ProgressReporter
from helpers.provenance import collect_hdf5_provenance, hash_json_payload

_REQUIRED_DATASETS = ("images", "masks", "labels", "patient_ids", "filenames")
_LINEAGE_ATTRS = (
    "source_signature",
    "source_hdf5_sha256",
    "upstream_source_signature",
    "stage4_cleaning_manifest_path",
    "stage4_cleaning_manifest_sha256",
    "stage4_cleaning_selected_rows",
)
_SPLIT_MANIFEST_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("patient_id", pa.int32()),
        ("relative_hdf5_path", pa.string()),
        ("rows", pa.int64()),
        ("label_0_count", pa.int64()),
        ("label_1_count", pa.int64()),
    ]
)
_SAMPLE_MANIFEST_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("patient_id", pa.int32()),
        ("relative_hdf5_path", pa.string()),
        ("row_in_shard", pa.int64()),
        ("label", pa.int8()),
        ("filename", pa.string()),
    ]
)


def _resolve_hdf5_compression(compression: str) -> str | None:
    normalized = compression.strip().lower()
    if normalized == "none":
        return None
    if normalized in {"gzip", "lzf"}:
        return normalized
    raise ValueError(f"Unsupported Stage 6.5 HDF5 compression: {compression}")


def _normalize_hdf5_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_rows_by_source_index(
    dataset: Any, source_indices: list[int]
) -> dict[int, npt.NDArray[np.uint8]]:
    rows_by_index: dict[int, npt.NDArray[np.uint8]] = {}
    if not source_indices:
        return rows_by_index

    span_start = source_indices[0]
    span_end = span_start + 1

    def flush(current_start: int, current_end: int) -> None:
        batch = np.asarray(dataset[current_start:current_end], dtype=np.uint8)
        for batch_offset, source_index in enumerate(range(current_start, current_end)):
            rows_by_index[source_index] = batch[batch_offset]

    for source_index in source_indices[1:]:
        if source_index == span_end:
            span_end += 1
            continue
        flush(span_start, span_end)
        span_start = source_index
        span_end = source_index + 1
    flush(span_start, span_end)
    return rows_by_index


def _build_selection_signature(
    *,
    source_path: Path,
    split_name: str,
    patient_id: int,
    source_signature: str | None,
    row_indices: list[int],
    filenames: list[str],
) -> str:
    payload = {
        "source_path": str(source_path),
        "split_name": split_name,
        "patient_id": patient_id,
        "source_signature": source_signature,
        "row_indices": row_indices,
        "filenames": filenames,
    }
    return hash_json_payload(payload)


def _validate_existing_patient_shard(output_path: Path, expected_selection_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        if handle.attrs.get("selection_signature") != expected_selection_signature:
            raise ValueError(
                f"Existing patient shard '{output_path}' does not match the current split inputs. "
                "Enable overwrite or remove the stale shard."
            )
    return output_path


def _validate_source_split_contract(source_path: Path) -> None:
    with h5py.File(source_path, "r") as handle:
        missing = [name for name in _REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"Stage 5 split HDF5 '{source_path}' is missing datasets: {missing}.")
        row_count = len(cast(Any, handle["filenames"]))
        for dataset_name in _REQUIRED_DATASETS:
            if len(cast(Any, handle[dataset_name])) != row_count:
                raise ValueError(
                    f"Stage 5 split HDF5 '{source_path}' has mismatched row counts for "
                    f"'{dataset_name}'."
                )


def write_patient_shard(
    *,
    source_path: Path,
    output_path: Path,
    split_name: str,
    patient_id: int,
    source_row_indices: list[int],
    hdf5_compression: str = "NONE",
    copy_batch_size: int = 256,
    overwrite: bool,
) -> Path:
    if not source_row_indices:
        raise ValueError(f"Cannot write patient shard '{output_path.name}' with no source rows.")

    _validate_source_split_contract(source_path)
    ordered_row_indices = sorted(int(index) for index in source_row_indices)
    resolved_compression = _resolve_hdf5_compression(hdf5_compression)
    source_split_provenance = collect_hdf5_provenance(source_path)

    with h5py.File(source_path, "r") as source_handle:
        source_patient_ids = np.asarray(
            source_handle["patient_ids"][ordered_row_indices], dtype=np.int32
        )
        unique_patient_ids = {int(value) for value in source_patient_ids.tolist()}
        if unique_patient_ids != {patient_id}:
            raise ValueError(
                f"Shard rows for patient {patient_id} in '{source_path.name}' contain "
                f"unexpected patient ids: {sorted(unique_patient_ids)}."
            )
        filenames_raw = source_handle["filenames"][ordered_row_indices]
        filenames = [_normalize_hdf5_string(value) for value in filenames_raw.tolist()]
        source_signature_attr = source_handle.attrs.get("source_signature")
        source_signature = (
            source_signature_attr.decode("utf-8")
            if isinstance(source_signature_attr, bytes)
            else cast(str | None, source_signature_attr)
        )
        selection_signature = _build_selection_signature(
            source_path=source_path,
            split_name=split_name,
            patient_id=patient_id,
            source_signature=source_signature,
            row_indices=ordered_row_indices,
            filenames=filenames,
        )

    if output_path.exists() and not overwrite:
        return _validate_existing_patient_shard(output_path, selection_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(
        "Stage 6.5 shard write",
        len(ordered_row_indices),
        "rows",
        context=output_path.name,
    )
    reporter.log_start(
        f"split={split_name} | patient_id={patient_id} | rows={len(ordered_row_indices)} "
        f"| compression={resolved_compression or 'none'} | batch_size={copy_batch_size}"
    )
    with (
        h5py.File(source_path, "r") as source_handle,
        h5py.File(output_path, "w") as dest_handle,
    ):
        for attr_name in _LINEAGE_ATTRS:
            attr_value = source_handle.attrs.get(attr_name)
            if attr_value is not None:
                dest_handle.attrs[attr_name] = attr_value
        dest_handle.attrs["source_split_hdf5_path"] = str(source_path)
        dest_handle.attrs["source_split_hdf5_sha256"] = source_split_provenance["sha256"]
        dest_handle.attrs["source_split_row_indices_json"] = json.dumps(ordered_row_indices)
        dest_handle.attrs["selection_signature"] = selection_signature
        dest_handle.attrs["split_name"] = split_name
        dest_handle.attrs["patient_id"] = patient_id
        first_index = ordered_row_indices[0]
        first_image = np.asarray(source_handle["images"][first_index], dtype=np.uint8)
        first_mask = np.asarray(source_handle["masks"][first_index], dtype=np.uint8)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        images = dest_handle.create_dataset(
            "images",
            shape=(len(ordered_row_indices),) + first_image.shape,
            dtype="uint8",
            compression=resolved_compression,
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(ordered_row_indices),) + first_mask.shape,
            dtype="uint8",
            compression=resolved_compression,
            chunks=True,
        )
        labels = dest_handle.create_dataset(
            "labels", shape=(len(ordered_row_indices),), dtype="uint8"
        )
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(ordered_row_indices),), dtype="int32"
        )
        filenames_dataset = dest_handle.create_dataset(
            "filenames", shape=(len(ordered_row_indices),), dtype=str_dtype
        )

        for batch_start in range(0, len(ordered_row_indices), copy_batch_size):
            batch_stop = min(batch_start + copy_batch_size, len(ordered_row_indices))
            batch_indices = ordered_row_indices[batch_start:batch_stop]
            images_by_index = _load_rows_by_source_index(source_handle["images"], batch_indices)
            masks_by_index = _load_rows_by_source_index(source_handle["masks"], batch_indices)
            image_batch = np.empty((len(batch_indices),) + first_image.shape, dtype=np.uint8)
            mask_batch = np.empty((len(batch_indices),) + first_mask.shape, dtype=np.uint8)
            for output_offset, source_index in enumerate(batch_indices):
                image_batch[output_offset] = images_by_index[source_index]
                mask_batch[output_offset] = masks_by_index[source_index]

            images[batch_start:batch_stop] = image_batch
            masks[batch_start:batch_stop] = mask_batch
            labels[batch_start:batch_stop] = np.asarray(
                source_handle["labels"][batch_indices],
                dtype=np.uint8,
            )
            patient_ids[batch_start:batch_stop] = np.asarray(
                source_handle["patient_ids"][batch_indices],
                dtype=np.int32,
            )
            filenames_dataset[batch_start:batch_stop] = np.asarray(
                [
                    _normalize_hdf5_string(value)
                    for value in source_handle["filenames"][batch_indices]
                ],
                dtype=object,
            )
            reporter.log(
                completed_units=batch_stop,
                extra_parts=[f"remaining={len(ordered_row_indices) - batch_stop} rows"],
            )
    logging.info(
        "Stage 6.5 shard verify: %s | split=%s | patient_id=%s | rows=%s",
        output_path.name,
        split_name,
        patient_id,
        len(ordered_row_indices),
    )
    return output_path


def verify_patient_shard_integrity(
    output_path: Path,
    *,
    patient_id: int,
    expected_rows: int,
    expected_labels: list[int] | None = None,
    expected_filenames: list[str] | None = None,
    expected_source_row_indices: list[int] | None = None,
    expected_source_split_path: Path | None = None,
) -> None:
    with h5py.File(output_path, "r") as handle:
        observed_patient_ids = np.asarray(handle["patient_ids"][:], dtype=np.int32)
        observed_labels = np.asarray(handle["labels"][:], dtype=np.uint8)
        observed_filenames = [
            _normalize_hdf5_string(value) for value in handle["filenames"][:].tolist()
        ]
        selection_signature = handle.attrs.get("selection_signature")
        shard_patient_id = int(handle.attrs.get("patient_id"))
        source_split_hdf5_path = _normalize_hdf5_string(
            handle.attrs.get("source_split_hdf5_path", "")
        )
        source_split_row_indices_json = _normalize_hdf5_string(
            handle.attrs.get("source_split_row_indices_json", "")
        )

    if len(observed_patient_ids) != expected_rows:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: row count {len(observed_patient_ids)} "
            f"does not match expected {expected_rows}."
        )
    unique_patient_ids = {int(value) for value in observed_patient_ids.tolist()}
    if unique_patient_ids != {patient_id}:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: expected one patient_id {patient_id}, "
            f"observed {sorted(unique_patient_ids)}."
        )
    if shard_patient_id != patient_id:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: patient_id attr {shard_patient_id} "
            f"does not match expected {patient_id}."
        )
    if selection_signature in {None, ""}:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: missing selection_signature attr."
        )
    if len(observed_filenames) != expected_rows:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: filenames row count does not match expected."
        )
    if expected_labels is not None and observed_labels.tolist() != expected_labels:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: label order does not match source rows."
        )
    if expected_filenames is not None and observed_filenames != expected_filenames:
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: filename order does not match source rows."
        )
    if expected_source_row_indices is not None:
        observed_source_row_indices = cast(list[int], json.loads(source_split_row_indices_json))
        if observed_source_row_indices != expected_source_row_indices:
            raise ValueError(
                f"Integrity FAILED for {output_path.name}: source row index lineage does not match."
            )
    if expected_source_split_path is not None and source_split_hdf5_path != str(
        expected_source_split_path
    ):
        raise ValueError(
            f"Integrity FAILED for {output_path.name}: source split path lineage does not match."
        )


def prepare_output_shard_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_shard_manifest_rows(split_name: str, shard_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.glob("*.h5")):
        with h5py.File(shard_path, "r") as handle:
            row_count = len(cast(Any, handle["patient_ids"]))
            patient_id = int(handle.attrs["patient_id"])
            labels = np.asarray(handle["labels"][:], dtype=np.uint8)
            rows.append(
                {
                    "split": split_name,
                    "patient_id": patient_id,
                    "relative_hdf5_path": str(shard_path.relative_to(shard_dir.parent)),
                    "rows": row_count,
                    "label_0_count": int((labels == 0).sum()),
                    "label_1_count": int((labels == 1).sum()),
                }
            )
    return rows


def build_sample_manifest_rows(split_name: str, shard_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.glob("*.h5")):
        relative_hdf5_path = str(shard_path.relative_to(shard_dir.parent))
        with h5py.File(shard_path, "r") as handle:
            patient_id = int(handle.attrs["patient_id"])
            labels = np.asarray(handle["labels"][:], dtype=np.uint8).tolist()
            filenames = [_normalize_hdf5_string(value) for value in handle["filenames"][:].tolist()]
        for row_in_shard, (label, filename) in enumerate(zip(labels, filenames, strict=True)):
            rows.append(
                {
                    "split": split_name,
                    "patient_id": patient_id,
                    "relative_hdf5_path": relative_hdf5_path,
                    "row_in_shard": row_in_shard,
                    "label": int(label),
                    "filename": filename,
                }
            )
    return rows


def write_split_manifest_parquet(output_path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=_SPLIT_MANIFEST_SCHEMA)
    pq.write_table(table, output_path)


def write_sample_manifest_parquet(output_path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=_SAMPLE_MANIFEST_SCHEMA)
    pq.write_table(table, output_path)


def write_split_summary_json(output_path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"shards": rows}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
