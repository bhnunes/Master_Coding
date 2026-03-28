from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
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
    "source_image_paths",
    "source_mask_paths",
)
_OPTIONAL_HDF5_DATASETS = ("slide_ids",)


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

        row_count = len(handle["filenames"])
        for dataset_name in _REQUIRED_HDF5_DATASETS:
            if len(handle[dataset_name]) != row_count:
                raise ValueError(
                    f"Source HDF5 dataset '{source_path}' has mismatched row counts for "
                    f"'{dataset_name}'."
                )

        images = handle["images"]
        masks = handle["masks"]
        if images.ndim != 4 or masks.ndim != 3:
            raise ValueError(
                f"Source HDF5 dataset '{source_path}' must store 4D images and 3D masks."
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
            actual_row_count = len(handle["filenames"])
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
        raise ValueError("Stage 5 HDF5 finalization requires different input and output paths.")

    _validate_source_hdf5_contract(source_path)
    source_signature = hash_file_sha256(source_path)
    if output_path.exists() and not overwrite:
        return _validate_existing_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
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


def filter_source_hdf5_by_manifest(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(source_path, "r") as source_handle, h5py.File(output_path, "w") as dest_handle:
        first_row_index = int(selected_rows[0]["source_row_index"])
        first_image = np.asarray(source_handle["images"][first_row_index], dtype=np.uint8)
        first_mask = np.asarray(source_handle["masks"][first_row_index], dtype=np.uint8)
        images = dest_handle.create_dataset(
            "images",
            shape=(len(selected_rows),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(selected_rows),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = dest_handle.create_dataset("labels", shape=(len(selected_rows),), dtype="uint8")
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(selected_rows),), dtype="int32"
        )
        filenames = dest_handle.create_dataset(
            "filenames", shape=(len(selected_rows),), dtype=str_dtype
        )
        source_image_paths = dest_handle.create_dataset(
            "source_image_paths", shape=(len(selected_rows),), dtype=str_dtype
        )
        source_mask_paths = dest_handle.create_dataset(
            "source_mask_paths", shape=(len(selected_rows),), dtype=str_dtype
        )
        source_row_indices = dest_handle.create_dataset(
            "source_row_indices", shape=(len(selected_rows),), dtype="int32"
        )
        slide_ids_source = source_handle.get("slide_ids")
        slide_ids = (
            dest_handle.create_dataset("slide_ids", shape=(len(selected_rows),), dtype=str_dtype)
            if slide_ids_source is not None
            else None
        )

        for output_index, row in enumerate(selected_rows):
            source_row_index = int(row["source_row_index"])
            images[output_index] = np.asarray(
                source_handle["images"][source_row_index], dtype=np.uint8
            )
            masks[output_index] = np.asarray(
                source_handle["masks"][source_row_index], dtype=np.uint8
            )
            labels[output_index] = int(source_handle["labels"][source_row_index])
            patient_ids[output_index] = int(source_handle["patient_ids"][source_row_index])
            filenames[output_index] = source_handle["filenames"][source_row_index]
            source_image_paths[output_index] = source_handle["source_image_paths"][source_row_index]
            source_mask_paths[output_index] = source_handle["source_mask_paths"][source_row_index]
            source_row_indices[output_index] = source_row_index
            if slide_ids is not None and slide_ids_source is not None:
                slide_ids[output_index] = slide_ids_source[source_row_index]

        upstream_signature = source_handle.attrs.get("source_signature")
        if upstream_signature is not None:
            dest_handle.attrs["upstream_source_signature"] = upstream_signature
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["stage4_cleaning_manifest_path"] = str(manifest_path)
        dest_handle.attrs["stage4_cleaning_manifest_sha256"] = hash_file_sha256(manifest_path)
        dest_handle.attrs["stage4_cleaning_selected_rows"] = len(selected_rows)
    return output_path


def merge_source_hdf5_shards(shard_dir: Path, output_path: Path, *, overwrite: bool) -> Path:
    shard_paths = _resolve_shard_paths(shard_dir)
    if not shard_paths:
        raise ValueError(f"No HDF5 shards found in '{shard_dir}'.")
    if output_path.exists() and not overwrite:
        source_signature = _signature_hexdigest(
            {
                "shards": [
                    {"path": str(path), "sha256": hash_file_sha256(path)} for path in shard_paths
                ]
            }
        )
        return _validate_existing_hdf5(output_path, source_signature)

    for shard_path in shard_paths:
        _validate_source_hdf5_contract(shard_path)

    rows: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as handle:
            slide_ids = handle.get("slide_ids")
            for index in range(len(handle["filenames"])):
                row: dict[str, Any] = {
                    "image": np.asarray(handle["images"][index], dtype=np.uint8),
                    "mask": np.asarray(handle["masks"][index], dtype=np.uint8),
                    "label": int(handle["labels"][index]),
                    "patient_id": int(handle["patient_ids"][index]),
                    "filename": handle["filenames"][index],
                    "source_image_path": handle["source_image_paths"][index],
                    "source_mask_path": handle["source_mask_paths"][index],
                }
                if slide_ids is not None:
                    row["slide_id"] = slide_ids[index]
                rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["patient_id"]),
            row["filename"].decode("utf-8")
            if isinstance(row["filename"], bytes)
            else str(row["filename"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    first_image = cast(npt.NDArray[np.uint8], rows[0]["image"])
    first_mask = cast(npt.NDArray[np.uint8], rows[0]["mask"])
    has_slide_ids = any("slide_id" in row for row in rows)

    with h5py.File(output_path, "w") as handle:
        images = handle.create_dataset(
            "images",
            shape=(len(rows),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = handle.create_dataset(
            "masks",
            shape=(len(rows),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = handle.create_dataset("labels", shape=(len(rows),), dtype="uint8")
        patient_ids = handle.create_dataset("patient_ids", shape=(len(rows),), dtype="int32")
        filenames = handle.create_dataset("filenames", shape=(len(rows),), dtype=str_dtype)
        source_image_paths = handle.create_dataset(
            "source_image_paths", shape=(len(rows),), dtype=str_dtype
        )
        source_mask_paths = handle.create_dataset(
            "source_mask_paths", shape=(len(rows),), dtype=str_dtype
        )
        slide_ids = (
            handle.create_dataset("slide_ids", shape=(len(rows),), dtype=str_dtype)
            if has_slide_ids
            else None
        )

        signature_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            image = cast(npt.NDArray[np.uint8], row["image"])
            mask = cast(npt.NDArray[np.uint8], row["mask"])
            images[index] = image
            masks[index] = mask
            labels[index] = int(row["label"])
            patient_ids[index] = int(row["patient_id"])
            filenames[index] = row["filename"]
            source_image_paths[index] = row["source_image_path"]
            source_mask_paths[index] = row["source_mask_path"]
            if slide_ids is not None:
                slide_ids[index] = row.get("slide_id", "")
            signature_row = {
                "filename": row["filename"].decode("utf-8")
                if isinstance(row["filename"], bytes)
                else str(row["filename"]),
                "label": int(row["label"]),
                "patient_id": int(row["patient_id"]),
                "source_image_path": row["source_image_path"].decode("utf-8")
                if isinstance(row["source_image_path"], bytes)
                else str(row["source_image_path"]),
                "source_mask_path": row["source_mask_path"].decode("utf-8")
                if isinstance(row["source_mask_path"], bytes)
                else str(row["source_mask_path"]),
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            }
            if "slide_id" in row:
                signature_row["slide_id"] = (
                    row["slide_id"].decode("utf-8")
                    if isinstance(row["slide_id"], bytes)
                    else str(row["slide_id"])
                )
            signature_rows.append(signature_row)

        handle.attrs["source_signature"] = _signature_hexdigest(
            {
                "shards": [
                    {"path": str(path), "sha256": hash_file_sha256(path)} for path in shard_paths
                ],
                "rows": signature_rows,
            }
        )
    return output_path
