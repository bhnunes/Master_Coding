from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt

from helpers.provenance import hash_file_sha256

_REQUIRED_HDF5_DATASETS = (
    "images",
    "masks",
    "labels",
    "patient_ids",
    "filenames",
)
_OPTIONAL_HDF5_DATASETS = ("slide_ids", "source_image_paths", "source_mask_paths")
_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_PROGRESS_LOG_INTERVAL_SECONDS = 1.5


@dataclass(frozen=True)
class _ShardMetadata:
    path: Path
    row_count: int
    source_signature: str


@dataclass(frozen=True)
class _MergeRowReference:
    shard_index: int
    source_row_index: int
    label: int
    patient_id: int
    filename: str
    slide_id: str | None


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_bytes(byte_count: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(byte_count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{byte_count} B"


@dataclass
class _ProgressReporter:
    phase_name: str
    total_units: int
    unit_label: str
    context: str = ""
    start_time: float = 0.0
    last_logged_at: float = 0.0

    def __post_init__(self) -> None:
        self.start_time = perf_counter()
        self.last_logged_at = self.start_time

    def log_start(self, summary: str) -> None:
        logging.info("%s: %s", self.phase_name, summary)

    def log(
        self,
        *,
        completed_units: int,
        force: bool = False,
        extra_parts: list[str] | None = None,
    ) -> None:
        now = perf_counter()
        if not force and completed_units < self.total_units:
            if now - self.last_logged_at < _PROGRESS_LOG_INTERVAL_SECONDS:
                return
        elapsed = max(now - self.start_time, 1e-9)
        rate = completed_units / elapsed if completed_units > 0 else 0.0
        remaining_units = max(self.total_units - completed_units, 0)
        eta_seconds = remaining_units / rate if rate > 0 else 0.0
        percent = (completed_units / self.total_units) * 100.0 if self.total_units else 100.0
        parts = [
            f"{completed_units}/{self.total_units} {self.unit_label}",
            f"{percent:.1f}%",
            f"{rate:.1f} {self.unit_label}/s",
            f"ETA {_format_duration(eta_seconds)}",
        ]
        if self.context:
            parts.insert(0, self.context)
        if extra_parts:
            parts.extend(extra_parts)
        logging.info("%s: %s", self.phase_name, " | ".join(parts))
        self.last_logged_at = now


def _dataset_kwargs(compression: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"chunks": True}
    if compression is not None:
        kwargs["compression"] = compression
    return kwargs


def _normalize_hdf5_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _contiguous_index_spans(sorted_indices: npt.NDArray[np.int64]) -> list[tuple[int, int]]:
    if len(sorted_indices) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(sorted_indices[0])
    previous = start
    for index in sorted_indices[1:]:
        current = int(index)
        if current == previous + 1:
            previous = current
            continue
        spans.append((start, previous + 1))
        start = previous = current
    spans.append((start, previous + 1))
    return spans


def _read_sorted_rows(
    dataset: Any,
    sorted_indices: npt.NDArray[np.int64],
    *,
    dtype: Any | None = None,
) -> npt.NDArray[Any]:
    spans = _contiguous_index_spans(sorted_indices)
    batches = [
        np.asarray(dataset[start:stop], dtype=dtype)
        if dtype is not None
        else np.asarray(dataset[start:stop])
        for start, stop in spans
    ]
    if len(batches) == 1:
        return batches[0]
    return np.concatenate(batches, axis=0)


def _build_logical_hdf5_ref(source_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{source_path}::{dataset_name}[{row_index}]"


def _signature_hexdigest(payload: dict[str, Any]) -> str:
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


def _validate_source_hdf5_contract(source_path: Path) -> None:
    with h5py.File(source_path, "r") as handle:
        missing = [name for name in _REQUIRED_HDF5_DATASETS if name not in handle]
        if missing:
            raise ValueError(
                f"Source HDF5 dataset '{source_path}' is missing required datasets: {missing}."
            )
        filenames = cast(Any, handle["filenames"])
        row_count = len(filenames)

        for dataset_name in _REQUIRED_HDF5_DATASETS:
            dataset = cast(Any, handle[dataset_name])
            if len(dataset) != row_count:
                raise ValueError(
                    f"Source HDF5 dataset '{source_path}' has mismatched row counts for "
                    f"'{dataset_name}'."
                )

        images = cast(Any, handle["images"])
        masks = cast(Any, handle["masks"])
        if images.ndim != 4 or masks.ndim != 3:
            raise ValueError(
                f"Source HDF5 dataset '{source_path}' must store 4D images and 3D masks."
            )

        for dataset_name in _OPTIONAL_HDF5_DATASETS:
            dataset = handle.get(dataset_name)
            if dataset is not None and len(cast(Any, dataset)) != row_count:
                raise ValueError(
                    f"Source HDF5 dataset '{source_path}' has mismatched row counts for "
                    f"'{dataset_name}'."
                )


def _load_stage2_shard_manifest(shard_dir: Path) -> list[dict[str, Any]] | None:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = payload.get("shards") if isinstance(payload, dict) else None
    if not isinstance(shards, list):
        raise ValueError(f"Invalid Stage 2 shard manifest: {manifest_path}")
    normalized: list[dict[str, Any]] = []
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid Stage 2 shard manifest entry in {manifest_path}")
        normalized.append(entry)
    return normalized


def _resolve_shard_paths(shard_dir: Path) -> list[Path]:
    manifest_entries = _load_stage2_shard_manifest(shard_dir)
    if manifest_entries is None:
        return sorted(
            path for path in shard_dir.iterdir() if path.is_file() and path.suffix == ".h5"
        )

    shard_paths: list[Path] = []
    actual_paths = {
        str(path.relative_to(shard_dir)): path
        for path in shard_dir.iterdir()
        if path.is_file() and path.suffix == ".h5"
    }
    manifest_relative_paths = {str(entry.get("relative_path", "")) for entry in manifest_entries}
    extras = sorted(set(actual_paths) - manifest_relative_paths)
    if extras:
        raise ValueError(
            f"Stage 2 shard manifest is missing on-disk shards: {extras[:10]} in '{shard_dir}'."
        )

    for entry in manifest_entries:
        relative_path = str(entry.get("relative_path", ""))
        shard_path = shard_dir / relative_path
        if not shard_path.is_file():
            raise ValueError(f"Stage 2 shard manifest references a missing shard: {relative_path}.")
        with h5py.File(shard_path, "r") as handle:
            actual_signature = handle.attrs.get("source_signature")
            actual_row_count = len(cast(Any, handle["filenames"]))
        if actual_signature != entry.get("source_signature"):
            raise ValueError(f"Stage 2 shard manifest signature mismatch for {relative_path}.")
        if int(actual_row_count) != int(entry.get("row_count", -1)):
            raise ValueError(f"Stage 2 shard manifest row_count mismatch for {relative_path}.")
        shard_paths.append(shard_path)
    return shard_paths


def copy_source_hdf5_dataset(source_path: Path, output_path: Path, *, overwrite: bool) -> Path:
    if source_path.suffix.lower() != ".h5":
        raise ValueError(f"Source HDF5 input must be a .h5 file, got: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Stage 3 HDF5 finalization requires different input and output paths.")

    _validate_source_hdf5_contract(source_path)
    source_signature = hash_file_sha256(source_path)
    if output_path.exists() and not overwrite:
        return _validate_existing_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = source_path.stat().st_size
    reporter = _ProgressReporter("Copy progress", total_bytes, "bytes")
    reporter.log_start(
        "mode=copy | "
        f"source={source_path} | output={output_path} | total={_format_bytes(total_bytes)}"
    )
    copied_bytes = 0
    with source_path.open("rb") as source_handle, output_path.open("wb") as output_handle:
        while True:
            chunk = source_handle.read(_COPY_BUFFER_BYTES)
            if not chunk:
                break
            output_handle.write(chunk)
            copied_bytes += len(chunk)
            reporter.log(
                completed_units=copied_bytes,
                extra_parts=[
                    f"copied={_format_bytes(copied_bytes)}",
                    f"remaining={_format_bytes(max(total_bytes - copied_bytes, 0))}",
                ],
            )
    shutil.copystat(source_path, output_path)
    reporter.log(
        completed_units=total_bytes,
        force=True,
        extra_parts=[
            f"copied={_format_bytes(total_bytes)}",
            f"remaining={_format_bytes(0)}",
        ],
    )
    with h5py.File(output_path, "r+") as handle:
        upstream_signature = handle.attrs.get("source_signature")
        if upstream_signature is not None:
            handle.attrs["upstream_source_signature"] = upstream_signature
        handle.attrs["source_signature"] = source_signature
    return output_path


def _load_accepted_manifest_rows(
    source_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("decision") == "accepted"]
    if not rows:
        raise ValueError(f"Accepted manifest has no accepted rows: {manifest_path}")

    normalized_rows: list[dict[str, Any]] = []
    expected_source = str(source_path)
    seen_indices: set[int] = set()
    for row in rows:
        row_source = row.get("source_hdf5_path", "")
        if row_source and row_source != expected_source:
            raise ValueError(f"Accepted manifest row does not match source_hdf5_path: {row_source}")
        source_row_index = int(row["source_row_index"])
        if source_row_index in seen_indices:
            continue
        seen_indices.add(source_row_index)
        normalized_rows.append(
            {
                "filename": row.get("filename", ""),
                "source_row_index": source_row_index,
                "patient_id": row.get("patient_id", ""),
                "slide_id": row.get("slide_id", ""),
            }
        )
    return normalized_rows


def _sorted_indices(
    indices: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    order = np.argsort(indices, kind="stable")
    return indices[order], order


def filter_source_hdf5_by_manifest(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    compression: str | None,
    copy_batch_size: int,
) -> Path:
    _validate_source_hdf5_contract(source_path)
    selected_rows = _load_accepted_manifest_rows(source_path, manifest_path)
    source_signature = _signature_hexdigest(
        {
            "source_hdf5_sha256": hash_file_sha256(source_path),
            "accepted_manifest_sha256": hash_file_sha256(manifest_path),
            "source_row_indices": [row["source_row_index"] for row in selected_rows],
        }
    )
    if output_path.exists() and not overwrite:
        return _validate_existing_hdf5(output_path, source_signature)

    total_rows = len(selected_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    dataset_kwargs = _dataset_kwargs(compression)
    reporter = _ProgressReporter("Filter progress", total_rows, "rows")
    reporter.log_start(
        "mode=filter | "
        f"source={source_path} | output={output_path} | selected_rows={total_rows} | "
        f"compression={compression or 'none'} | batch_size={copy_batch_size}"
    )
    with h5py.File(source_path, "r") as source_handle, h5py.File(output_path, "w") as dest_handle:
        source_images = cast(Any, source_handle["images"])
        source_masks = cast(Any, source_handle["masks"])
        images = dest_handle.create_dataset(
            "images",
            shape=(total_rows,) + tuple(source_images.shape[1:]),
            dtype="uint8",
            **dataset_kwargs,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(total_rows,) + tuple(source_masks.shape[1:]),
            dtype="uint8",
            **dataset_kwargs,
        )
        labels = dest_handle.create_dataset("labels", shape=(total_rows,), dtype="uint8")
        patient_ids = dest_handle.create_dataset("patient_ids", shape=(total_rows,), dtype="int32")
        filenames = dest_handle.create_dataset("filenames", shape=(total_rows,), dtype=str_dtype)
        source_image_paths = dest_handle.create_dataset(
            "source_image_paths", shape=(total_rows,), dtype=str_dtype
        )
        source_mask_paths = dest_handle.create_dataset(
            "source_mask_paths", shape=(total_rows,), dtype=str_dtype
        )
        source_row_indices = dest_handle.create_dataset(
            "source_row_indices", shape=(total_rows,), dtype="int32"
        )
        slide_ids_source = source_handle.get("slide_ids")
        source_image_refs = source_handle.get("source_image_paths")
        source_mask_refs = source_handle.get("source_mask_paths")
        slide_ids = (
            dest_handle.create_dataset("slide_ids", shape=(total_rows,), dtype=str_dtype)
            if slide_ids_source is not None
            else None
        )

        source_labels = cast(Any, source_handle["labels"])
        source_patient_ids = cast(Any, source_handle["patient_ids"])
        source_filenames = cast(Any, source_handle["filenames"])
        source_slide_ids = cast(Any, slide_ids_source) if slide_ids_source is not None else None
        source_image_paths_dataset = (
            cast(Any, source_image_refs) if source_image_refs is not None else None
        )
        source_mask_paths_dataset = (
            cast(Any, source_mask_refs) if source_mask_refs is not None else None
        )

        for start in range(0, total_rows, copy_batch_size):
            end = min(start + copy_batch_size, total_rows)
            batch_rows = selected_rows[start:end]
            batch_indices = np.fromiter(
                (int(row["source_row_index"]) for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            sorted_indices, order = _sorted_indices(batch_indices)

            batch_images = np.empty(
                (len(batch_rows),) + tuple(source_images.shape[1:]), dtype=np.uint8
            )
            batch_masks = np.empty(
                (len(batch_rows),) + tuple(source_masks.shape[1:]), dtype=np.uint8
            )
            batch_images[order] = _read_sorted_rows(source_images, sorted_indices, dtype=np.uint8)
            batch_masks[order] = _read_sorted_rows(source_masks, sorted_indices, dtype=np.uint8)
            labels[start:end] = _read_sorted_rows(source_labels, sorted_indices, dtype=np.uint8)[
                order
            ]
            patient_ids[start:end] = _read_sorted_rows(
                source_patient_ids, sorted_indices, dtype=np.int32
            )[order]
            filenames[start:end] = [
                _normalize_hdf5_string(value)
                for value in _read_sorted_rows(source_filenames, sorted_indices, dtype=object)[
                    order
                ]
            ]
            if source_image_paths_dataset is not None:
                source_image_paths[start:end] = [
                    _normalize_hdf5_string(value)
                    for value in _read_sorted_rows(
                        source_image_paths_dataset,
                        sorted_indices,
                        dtype=object,
                    )[order]
                ]
            else:
                source_image_paths[start:end] = [
                    _build_logical_hdf5_ref(source_path, "images", int(index))
                    for index in batch_indices
                ]
            if source_mask_paths_dataset is not None:
                source_mask_paths[start:end] = [
                    _normalize_hdf5_string(value)
                    for value in _read_sorted_rows(
                        source_mask_paths_dataset,
                        sorted_indices,
                        dtype=object,
                    )[order]
                ]
            else:
                source_mask_paths[start:end] = [
                    _build_logical_hdf5_ref(source_path, "masks", int(index))
                    for index in batch_indices
                ]
            source_row_indices[start:end] = batch_indices.astype(np.int32, copy=False)
            if slide_ids is not None and source_slide_ids is not None:
                slide_ids[start:end] = [
                    _normalize_hdf5_string(value)
                    for value in _read_sorted_rows(source_slide_ids, sorted_indices, dtype=object)[
                        order
                    ]
                ]
            images[start:end] = batch_images
            masks[start:end] = batch_masks
            reporter.log(
                completed_units=end,
                extra_parts=[
                    f"read={end}/{total_rows} rows",
                    f"wrote={end}/{total_rows} rows",
                    f"remaining={total_rows - end} rows",
                ],
            )

        upstream_signature = source_handle.attrs.get("source_signature")
        if upstream_signature is not None:
            dest_handle.attrs["upstream_source_signature"] = upstream_signature
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["stage4_cleaning_manifest_path"] = str(manifest_path)
        dest_handle.attrs["stage4_cleaning_manifest_sha256"] = hash_file_sha256(manifest_path)
        dest_handle.attrs["stage4_cleaning_selected_rows"] = total_rows
    reporter.log(
        completed_units=total_rows,
        force=True,
        extra_parts=[
            f"read={total_rows}/{total_rows} rows",
            f"wrote={total_rows}/{total_rows} rows",
            "remaining=0 rows",
        ],
    )
    return output_path


def _shard_source_signature(handle: h5py.File, shard_path: Path) -> str:
    source_signature = handle.attrs.get("source_signature")
    if source_signature is not None:
        return _normalize_hdf5_string(source_signature)
    return hash_file_sha256(shard_path)


def _build_merge_row_references(
    *,
    shard_index: int,
    handle: h5py.File,
) -> list[_MergeRowReference]:
    labels = np.asarray(cast(Any, handle["labels"])[:], dtype=np.uint8)
    patient_ids = np.asarray(cast(Any, handle["patient_ids"])[:], dtype=np.int32)
    filenames = np.asarray(cast(Any, handle["filenames"])[:], dtype=object)
    slide_ids_dataset = handle.get("slide_ids")
    slide_ids = (
        np.asarray(cast(Any, slide_ids_dataset)[:], dtype=object)
        if slide_ids_dataset is not None
        else None
    )

    return [
        _MergeRowReference(
            shard_index=shard_index,
            source_row_index=index,
            label=int(labels[index]),
            patient_id=int(patient_ids[index]),
            filename=_normalize_hdf5_string(filenames[index]),
            slide_id=_normalize_hdf5_string(slide_ids[index]) if slide_ids is not None else None,
        )
        for index in range(len(filenames))
    ]


def _merge_signature(
    *,
    shard_metadata: list[_ShardMetadata],
    row_references: list[_MergeRowReference],
) -> str:
    payload: dict[str, Any] = {
        "shards": [
            {
                "path": str(metadata.path),
                "row_count": metadata.row_count,
                "source_signature": metadata.source_signature,
            }
            for metadata in shard_metadata
        ],
        "rows": [
            {
                "filename": row.filename,
                "label": row.label,
                "patient_id": row.patient_id,
                "slide_id": row.slide_id,
                "source_hdf5_path": str(shard_metadata[row.shard_index].path),
                "source_row_index": row.source_row_index,
            }
            for row in row_references
        ],
    }
    return _signature_hexdigest(payload)


def merge_source_hdf5_shards(
    shard_dir: Path,
    output_path: Path,
    *,
    overwrite: bool,
    compression: str | None,
    copy_batch_size: int,
) -> Path:
    shard_paths = _resolve_shard_paths(shard_dir)
    if not shard_paths:
        raise ValueError(f"No HDF5 shards found in '{shard_dir}'.")

    logging.info("Merging %s HDF5 shard(s) from %s", len(shard_paths), shard_dir)
    shard_scan_reporter = _ProgressReporter("Merge shard scan", len(shard_paths), "shards")
    shard_scan_reporter.log_start(
        "mode=merge | "
        f"source={shard_dir} | output={output_path} | shard_count={len(shard_paths)} | "
        f"compression={compression or 'none'} | batch_size={copy_batch_size}"
    )
    with ExitStack() as stack:
        shard_handles = [
            stack.enter_context(h5py.File(shard_path, "r")) for shard_path in shard_paths
        ]
        shard_metadata: list[_ShardMetadata] = []
        row_references: list[_MergeRowReference] = []

        for shard_index, (shard_path, handle) in enumerate(
            zip(shard_paths, shard_handles, strict=True), start=1
        ):
            _validate_source_hdf5_contract(shard_path)
            row_count = len(cast(Any, handle["filenames"]))
            shard_metadata.append(
                _ShardMetadata(
                    path=shard_path,
                    row_count=row_count,
                    source_signature=_shard_source_signature(handle, shard_path),
                )
            )
            row_references.extend(
                _build_merge_row_references(shard_index=shard_index - 1, handle=handle)
            )
            shard_scan_reporter.log(
                completed_units=shard_index,
                extra_parts=[
                    f"last_shard={shard_path.name}",
                    f"rows_discovered={len(row_references)}",
                    f"remaining_shards={len(shard_paths) - shard_index}",
                ],
            )

        row_references.sort(key=lambda row: (row.patient_id, row.filename))
        source_signature = _merge_signature(
            shard_metadata=shard_metadata,
            row_references=row_references,
        )
        if output_path.exists() and not overwrite:
            return _validate_existing_hdf5(output_path, source_signature)

        total_rows = len(row_references)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        first_images = cast(Any, shard_handles[0]["images"])
        first_masks = cast(Any, shard_handles[0]["masks"])
        image_shape = tuple(first_images.shape[1:])
        mask_shape = tuple(first_masks.shape[1:])
        has_slide_ids = any(row.slide_id is not None for row in row_references)
        dataset_kwargs = _dataset_kwargs(compression)
        merge_reporter = _ProgressReporter("Merge progress", total_rows, "rows")
        merge_reporter.log_start(
            f"total_rows={total_rows} | read=0/{total_rows} rows | wrote=0/{total_rows} rows | "
            f"remaining={total_rows} rows"
        )

        with h5py.File(output_path, "w") as output_handle:
            images = output_handle.create_dataset(
                "images",
                shape=(total_rows,) + image_shape,
                dtype="uint8",
                **dataset_kwargs,
            )
            masks = output_handle.create_dataset(
                "masks",
                shape=(total_rows,) + mask_shape,
                dtype="uint8",
                **dataset_kwargs,
            )
            labels = output_handle.create_dataset("labels", shape=(total_rows,), dtype="uint8")
            patient_ids = output_handle.create_dataset(
                "patient_ids", shape=(total_rows,), dtype="int32"
            )
            filenames = output_handle.create_dataset(
                "filenames", shape=(total_rows,), dtype=str_dtype
            )
            source_image_paths = output_handle.create_dataset(
                "source_image_paths", shape=(total_rows,), dtype=str_dtype
            )
            source_mask_paths = output_handle.create_dataset(
                "source_mask_paths", shape=(total_rows,), dtype=str_dtype
            )
            slide_ids = (
                output_handle.create_dataset("slide_ids", shape=(total_rows,), dtype=str_dtype)
                if has_slide_ids
                else None
            )

            for start in range(0, total_rows, copy_batch_size):
                end = min(start + copy_batch_size, total_rows)
                batch_rows = row_references[start:end]
                batch_read_rows = 0
                batch_images = np.empty((len(batch_rows),) + image_shape, dtype=np.uint8)
                batch_masks = np.empty((len(batch_rows),) + mask_shape, dtype=np.uint8)
                batch_labels = np.empty(len(batch_rows), dtype=np.uint8)
                batch_patient_ids = np.empty(len(batch_rows), dtype=np.int32)
                batch_filenames = np.empty(len(batch_rows), dtype=object)
                batch_source_image_paths = np.empty(len(batch_rows), dtype=object)
                batch_source_mask_paths = np.empty(len(batch_rows), dtype=object)
                batch_slide_ids = np.empty(len(batch_rows), dtype=object) if has_slide_ids else None

                positions_by_shard: dict[int, list[int]] = {}
                for position, row in enumerate(batch_rows):
                    positions_by_shard.setdefault(row.shard_index, []).append(position)

                for shard_index, positions in positions_by_shard.items():
                    shard_handle = shard_handles[shard_index]
                    source_indices = np.fromiter(
                        (batch_rows[position].source_row_index for position in positions),
                        dtype=np.int64,
                        count=len(positions),
                    )
                    batch_read_rows += len(source_indices)
                    sorted_indices, order = _sorted_indices(source_indices)
                    ordered_positions = np.asarray(positions, dtype=np.int64)[order]

                    shard_images = cast(Any, shard_handle["images"])
                    shard_masks = cast(Any, shard_handle["masks"])
                    shard_labels = cast(Any, shard_handle["labels"])
                    shard_patient_ids = cast(Any, shard_handle["patient_ids"])
                    shard_filenames = cast(Any, shard_handle["filenames"])
                    batch_images[ordered_positions] = _read_sorted_rows(
                        shard_images,
                        sorted_indices,
                        dtype=np.uint8,
                    )
                    batch_masks[ordered_positions] = _read_sorted_rows(
                        shard_masks,
                        sorted_indices,
                        dtype=np.uint8,
                    )
                    batch_labels[ordered_positions] = _read_sorted_rows(
                        shard_labels,
                        sorted_indices,
                        dtype=np.uint8,
                    )
                    batch_patient_ids[ordered_positions] = _read_sorted_rows(
                        shard_patient_ids,
                        sorted_indices,
                        dtype=np.int32,
                    )
                    batch_filenames[ordered_positions] = [
                        _normalize_hdf5_string(value)
                        for value in _read_sorted_rows(
                            shard_filenames, sorted_indices, dtype=object
                        )
                    ]

                    shard_source_image_paths = shard_handle.get("source_image_paths")
                    if shard_source_image_paths is not None:
                        batch_source_image_paths[ordered_positions] = [
                            _normalize_hdf5_string(value)
                            for value in _read_sorted_rows(
                                cast(Any, shard_source_image_paths),
                                sorted_indices,
                                dtype=object,
                            )
                        ]
                    else:
                        batch_source_image_paths[ordered_positions] = [
                            _build_logical_hdf5_ref(
                                shard_metadata[shard_index].path,
                                "images",
                                int(source_index),
                            )
                            for source_index in sorted_indices
                        ]

                    shard_source_mask_paths = shard_handle.get("source_mask_paths")
                    if shard_source_mask_paths is not None:
                        batch_source_mask_paths[ordered_positions] = [
                            _normalize_hdf5_string(value)
                            for value in _read_sorted_rows(
                                cast(Any, shard_source_mask_paths),
                                sorted_indices,
                                dtype=object,
                            )
                        ]
                    else:
                        batch_source_mask_paths[ordered_positions] = [
                            _build_logical_hdf5_ref(
                                shard_metadata[shard_index].path,
                                "masks",
                                int(source_index),
                            )
                            for source_index in sorted_indices
                        ]

                    if batch_slide_ids is not None:
                        shard_slide_ids = shard_handle.get("slide_ids")
                        if shard_slide_ids is not None:
                            batch_slide_ids[ordered_positions] = [
                                _normalize_hdf5_string(value)
                                for value in _read_sorted_rows(
                                    cast(Any, shard_slide_ids),
                                    sorted_indices,
                                    dtype=object,
                                )
                            ]
                        else:
                            batch_slide_ids[ordered_positions] = ""

                images[start:end] = batch_images
                masks[start:end] = batch_masks
                labels[start:end] = batch_labels
                patient_ids[start:end] = batch_patient_ids
                filenames[start:end] = batch_filenames.tolist()
                source_image_paths[start:end] = batch_source_image_paths.tolist()
                source_mask_paths[start:end] = batch_source_mask_paths.tolist()
                if slide_ids is not None and batch_slide_ids is not None:
                    slide_ids[start:end] = batch_slide_ids.tolist()
                merge_reporter.log(
                    completed_units=end,
                    extra_parts=[
                        f"read={end}/{total_rows} rows",
                        f"wrote={end}/{total_rows} rows",
                        f"batch_read={batch_read_rows} rows",
                        f"remaining={total_rows - end} rows",
                    ],
                )

            output_handle.attrs["source_signature"] = source_signature
        merge_reporter.log(
            completed_units=total_rows,
            force=True,
            extra_parts=[
                f"read={total_rows}/{total_rows} rows",
                f"wrote={total_rows}/{total_rows} rows",
                "remaining=0 rows",
            ],
        )
    return output_path
