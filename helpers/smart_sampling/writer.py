from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from helpers.provenance import collect_hdf5_provenance, hash_file_sha256
from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.index import guardrail, resolve_filename_key

_FILTERED_SPLIT_MANIFEST_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("patient_id", pa.int32()),
        ("relative_hdf5_path", pa.string()),
        ("rows", pa.int64()),
        ("label_0_count", pa.int64()),
        ("label_1_count", pa.int64()),
    ]
)
_FILTERED_SAMPLE_MANIFEST_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("patient_id", pa.int32()),
        ("relative_hdf5_path", pa.string()),
        ("row_in_shard", pa.int64()),
        ("label", pa.int8()),
        ("filename", pa.string()),
    ]
)
_UPSTREAM_LINEAGE_ATTRS = (
    "source_signature",
    "source_hdf5_sha256",
    "upstream_source_signature",
    "stage4_cleaning_manifest_path",
    "stage4_cleaning_manifest_sha256",
    "stage4_cleaning_selected_rows",
    "source_split_hdf5_path",
    "source_split_hdf5_sha256",
)


def _hash_json_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _build_sampling_signature(
    selected_indices: npt.NDArray[np.int64],
    *,
    config: SmartSamplerConfig,
    source_path: Path,
    output_filename: str,
    signature_source_path: Path | None = None,
) -> str:
    source_provenance = collect_hdf5_provenance(source_path)
    if signature_source_path is not None:
        source_provenance["path"] = str(signature_source_path)
    payload = {
        "selected_indices": np.asarray(selected_indices, dtype=np.int64).tolist(),
        "source_path": str(signature_source_path or source_path),
        "source_provenance": source_provenance,
        "output_filename": output_filename,
        "stage7_metadata": _stage7_metadata_payload(config),
    }
    return _hash_json_payload(payload)


def _stage7_metadata_payload(config: SmartSamplerConfig) -> dict[str, Any]:
    return {
        "stage7_label_aware": True,
        "stage7_selector": "gist_facility_location"
        if config.use_gist
        else "legacy_adaptive_coverage",
        "stage7_model_name": config.model_name,
        "stage7_seed": config.seed,
        "stage7_stability_threshold": config.stability_threshold,
        "stage7_stability_repeats": config.stability_repeats,
        "stage7_keep_improvement_threshold": config.keep_improvement_threshold,
        "stage7_keep_patience": config.keep_patience,
        "stage7_keep_min": config.keep_min,
        "stage7_keep_step": config.keep_step,
        "stage7_m_max": config.m_max,
        "stage7_holdout_mode": "within_patient_patch_holdout",
        "stage7_protect_positive_labels": config.protect_positive_labels,
        "stage7_protect_mask_positive": config.protect_mask_positive,
        "stage7_positive_mask_fraction_threshold": config.positive_mask_fraction_threshold,
    }


def _build_shard_sampling_signature(
    selected_rows: list[dict[str, Any]],
    *,
    config: SmartSamplerConfig,
    source_shard_dir: Path,
    source_manifest_path: Path,
    signature_source_manifest_path: Path | None = None,
    signature_source_dir: Path | None = None,
) -> str:
    manifest_provenance = {
        "path": str(signature_source_manifest_path or source_manifest_path),
        "source_shard_dir": str(signature_source_dir or source_shard_dir),
        "sha256": hash_file_sha256(source_manifest_path),
    }
    payload = {
        "selected_rows": selected_rows,
        "source_shard_dir": str(signature_source_dir or source_shard_dir),
        "source_manifest": manifest_provenance,
        "output_filename": config.output_filename,
        "stage7_metadata": _stage7_metadata_payload(config),
    }
    return _hash_json_payload(payload)


def build_filtered_shard_selection_signature(
    config: SmartSamplerConfig,
    selected_rows: list[dict[str, Any]],
    *,
    source_shard_dir: Path,
    source_manifest_path: Path,
    signature_source_manifest_path: Path | None = None,
    signature_source_dir: Path | None = None,
) -> str:
    normalized_rows = [
        {
            "patient_id": int(row["patient_id"]),
            "relative_hdf5_path": str(row["relative_hdf5_path"]),
            "row_in_shard": int(row["row_in_shard"]),
        }
        for row in selected_rows
    ]
    return _build_shard_sampling_signature(
        normalized_rows,
        config=config,
        source_shard_dir=source_shard_dir,
        source_manifest_path=source_manifest_path,
        signature_source_manifest_path=signature_source_manifest_path,
        signature_source_dir=signature_source_dir,
    )


def _normalize_hdf5_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _build_filtered_patient_shard_signature(
    *,
    patient_id: int,
    selected_row_indices: list[int],
    source_shard_path: Path,
    source_relative_path: str,
    output_relative_path: str,
    config: SmartSamplerConfig,
    signature_source_dir: Path | None = None,
) -> str:
    source_provenance = collect_hdf5_provenance(source_shard_path)
    source_provenance["path"] = str(
        (signature_source_dir.parent / source_relative_path)
        if signature_source_dir is not None
        else source_shard_path
    )
    payload = {
        "patient_id": patient_id,
        "selected_row_indices": selected_row_indices,
        "source_shard_path": source_provenance["path"],
        "source_shard_provenance": source_provenance,
        "source_shard_dir": str(signature_source_dir) if signature_source_dir is not None else None,
        "output_relative_path": output_relative_path,
        "stage7_metadata": _stage7_metadata_payload(config),
    }
    return _hash_json_payload(payload)


def _validate_existing_filtered_hdf5(output_path: Path, expected_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        existing_signature = handle.attrs.get("selection_signature")
        if existing_signature != expected_signature:
            raise ValueError(
                f"Existing filtered HDF5 '{output_path}' does not match the current selection. "
                "Enable overwrite or remove the stale file."
            )
    return output_path


def _validate_existing_filtered_shard_dir(output_dir: Path, expected_signature: str) -> Path:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"Existing filtered shard directory '{output_dir}' is missing summary.json. "
            "Enable overwrite or remove the stale directory."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("selection_signature") != expected_signature:
        raise ValueError(
            f"Existing filtered shard directory '{output_dir}' does not match "
            "the current selection. "
            "Enable overwrite or remove the stale directory."
        )
    return output_dir


def write_filter_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "filter_summary.json"
    logging.info("Writing filter summary to %s", summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    return summary_path


def write_filtered_hdf5(
    config: SmartSamplerConfig,
    selected_indices: npt.NDArray[np.int64],
    *,
    source_h5_path: Path | None = None,
    output_dir: Path | None = None,
    signature_source_path: Path | None = None,
) -> Path:
    source_path = source_h5_path or config.source_h5_path
    if source_path is None:
        raise ValueError("write_filtered_hdf5 requires a source_h5_path for singleton input.")
    output_path = (output_dir or config.output_dir) / config.output_filename
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selection_signature = _build_sampling_signature(
        selected_indices,
        config=config,
        source_path=source_path,
        signature_source_path=signature_source_path,
        output_filename=config.output_filename,
    )
    if output_path.exists() and not config.overwrite_output:
        logging.info("Reusing existing filtered HDF5 at %s", output_path)
        return _validate_existing_filtered_hdf5(output_path, selection_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(selected_indices)
    logging.info("Writing filtered HDF5 with %d selected rows to %s", total, output_path)

    with h5py.File(source_path, "r") as source_handle, h5py.File(output_path, "w") as dest_handle:
        dest_handle.attrs["selection_signature"] = selection_signature
        dest_handle.attrs["source_hdf5_sha256"] = collect_hdf5_provenance(source_path)["sha256"]
        for attr_name, attr_value in _stage7_metadata_payload(config).items():
            dest_handle.attrs[attr_name] = attr_value
        for attr_name in (
            "source_signature",
            "upstream_source_signature",
            "stage4_cleaning_manifest_path",
            "stage4_cleaning_manifest_sha256",
            "stage4_cleaning_selected_rows",
        ):
            attr_value = source_handle.attrs.get(attr_name)
            if attr_value is not None:
                dest_handle.attrs[attr_name] = attr_value
        guardrail(source_handle)
        filename_key = resolve_filename_key(source_handle)
        for key in ["images", "masks", "patient_ids", "labels", filename_key]:
            target_key = "filenames" if key == filename_key else key
            source_dataset = source_handle[key]
            shape = list(cast(Any, source_dataset).shape)
            shape[0] = total
            kwargs: dict[str, Any] = {}
            if key in {"images", "masks"}:
                kwargs = {"compression": "gzip", "chunks": True}
            dest_handle.create_dataset(
                target_key,
                shape=tuple(shape),
                dtype=cast(Any, source_dataset).dtype,
                **kwargs,
            )

        batch_size = 1000
        sorted_indices = np.sort(selected_indices)
        for start in tqdm(range(0, total, batch_size), desc="Writing HDF5", disable=total == 0):
            batch_indices = sorted_indices[start : start + batch_size]
            for key in ["images", "masks", "patient_ids", "labels", filename_key]:
                target_key = "filenames" if key == filename_key else key
                dest_dataset = dest_handle[target_key]
                source_dataset = source_handle[key]
                cast(Any, dest_dataset)[start : start + len(batch_indices)] = cast(
                    Any, source_dataset
                )[batch_indices]

    logging.info("Finished writing filtered HDF5 to %s", output_path)
    return output_path


def write_filtered_hdf5_from_shards(
    config: SmartSamplerConfig,
    selected_rows: list[dict[str, Any]],
    *,
    source_shard_dir: Path,
    source_manifest_path: Path,
    output_dir: Path | None = None,
    signature_source_dir: Path | None = None,
) -> Path:
    output_path = (output_dir or config.output_dir) / config.output_filename
    normalized_rows = [
        {
            "patient_id": int(row["patient_id"]),
            "relative_hdf5_path": str(row["relative_hdf5_path"]),
            "row_in_shard": int(row["row_in_shard"]),
        }
        for row in selected_rows
    ]
    selection_signature = _build_shard_sampling_signature(
        normalized_rows,
        config=config,
        source_shard_dir=source_shard_dir,
        source_manifest_path=source_manifest_path,
        signature_source_dir=signature_source_dir,
    )
    if output_path.exists() and not config.overwrite_output:
        logging.info("Reusing existing filtered HDF5 at %s", output_path)
        return _validate_existing_filtered_hdf5(output_path, selection_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(normalized_rows)
    logging.info("Writing filtered HDF5 with %d selected rows to %s", total, output_path)
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_rows:
        grouped_rows.setdefault(str(row["relative_hdf5_path"]), []).append(row)

    first_row = normalized_rows[0] if normalized_rows else None
    first_image_shape: tuple[int, ...] = (0,)
    first_mask_shape: tuple[int, ...] = (0,)
    if first_row is not None:
        first_shard_path = source_shard_dir.parent / str(first_row["relative_hdf5_path"])
        with h5py.File(first_shard_path, "r") as handle:
            first_image_shape = tuple(np.asarray(handle["images"][0], dtype=np.uint8).shape)
            first_mask_shape = tuple(np.asarray(handle["masks"][0], dtype=np.uint8).shape)

    with h5py.File(output_path, "w") as dest_handle:
        dest_handle.attrs["selection_signature"] = selection_signature
        dest_handle.attrs["source_shard_dir"] = str(signature_source_dir or source_shard_dir)
        dest_handle.attrs["source_shard_manifest_path"] = str(source_manifest_path)
        dest_handle.attrs["source_shard_manifest_sha256"] = hash_file_sha256(source_manifest_path)
        for attr_name, attr_value in _stage7_metadata_payload(config).items():
            dest_handle.attrs[attr_name] = attr_value

        str_dtype = h5py.string_dtype(encoding="utf-8")
        images = dest_handle.create_dataset(
            "images",
            shape=(total,) + first_image_shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(total,) + first_mask_shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        patient_ids = dest_handle.create_dataset("patient_ids", shape=(total,), dtype="int32")
        labels = dest_handle.create_dataset("labels", shape=(total,), dtype="uint8")
        filenames = dest_handle.create_dataset("filenames", shape=(total,), dtype=str_dtype)

        output_offset = 0
        for relative_hdf5_path, shard_rows in grouped_rows.items():
            shard_path = source_shard_dir.parent / relative_hdf5_path
            sorted_rows = sorted(shard_rows, key=lambda row: int(row["row_in_shard"]))
            shard_indices = np.asarray(
                [int(row["row_in_shard"]) for row in sorted_rows], dtype=np.int64
            )
            with h5py.File(shard_path, "r") as source_handle:
                guardrail(source_handle)
                filename_key = resolve_filename_key(source_handle)
                batch_size = len(shard_indices)
                images[output_offset : output_offset + batch_size] = np.asarray(
                    source_handle["images"][shard_indices], dtype=np.uint8
                )
                masks[output_offset : output_offset + batch_size] = np.asarray(
                    source_handle["masks"][shard_indices], dtype=np.uint8
                )
                patient_ids[output_offset : output_offset + batch_size] = np.asarray(
                    source_handle["patient_ids"][shard_indices], dtype=np.int32
                )
                labels[output_offset : output_offset + batch_size] = np.asarray(
                    source_handle["labels"][shard_indices], dtype=np.uint8
                )
                filenames[output_offset : output_offset + batch_size] = np.asarray(
                    [
                        value.decode("utf-8") if isinstance(value, bytes) else str(value)
                        for value in source_handle[filename_key][shard_indices].tolist()
                    ],
                    dtype=object,
                )
                for attr_name in (
                    "source_signature",
                    "upstream_source_signature",
                    "stage4_cleaning_manifest_path",
                    "stage4_cleaning_manifest_sha256",
                    "stage4_cleaning_selected_rows",
                ):
                    attr_value = source_handle.attrs.get(attr_name)
                    if attr_value is not None and attr_name not in dest_handle.attrs:
                        dest_handle.attrs[attr_name] = attr_value
            output_offset += batch_size

    logging.info("Finished writing filtered HDF5 to %s", output_path)
    return output_path


def write_filtered_shards(
    config: SmartSamplerConfig,
    selected_rows: list[dict[str, Any]],
    *,
    source_shard_dir: Path,
    source_manifest_path: Path,
    output_dir: Path | None = None,
    signature_source_dir: Path | None = None,
) -> Path:
    filtered_shard_dir = (output_dir or config.output_dir) / config.output_filename
    normalized_rows = [
        {
            "patient_id": int(row["patient_id"]),
            "relative_hdf5_path": str(row["relative_hdf5_path"]),
            "row_in_shard": int(row["row_in_shard"]),
        }
        for row in selected_rows
    ]
    selection_signature = build_filtered_shard_selection_signature(
        config,
        normalized_rows,
        source_shard_dir=source_shard_dir,
        source_manifest_path=source_manifest_path,
        signature_source_manifest_path=config.source_manifest_path,
        signature_source_dir=signature_source_dir,
    )
    if filtered_shard_dir.exists() and not config.overwrite_output:
        logging.info("Reusing existing filtered shard directory at %s", filtered_shard_dir)
        return _validate_existing_filtered_shard_dir(filtered_shard_dir, selection_signature)

    if filtered_shard_dir.exists():
        shutil.rmtree(filtered_shard_dir)

    filtered_shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    sample_manifest_rows: list[dict[str, Any]] = []
    grouped_rows: dict[tuple[int, str], list[int]] = {}
    for row in normalized_rows:
        patient_id = cast(int, row["patient_id"])
        relative_hdf5_path = cast(str, row["relative_hdf5_path"])
        row_in_shard = cast(int, row["row_in_shard"])
        grouped_rows.setdefault((patient_id, relative_hdf5_path), []).append(row_in_shard)

    for (patient_id, relative_hdf5_path), row_indices in sorted(grouped_rows.items()):
        output_relative_path = f"{config.output_filename}/{patient_id}.h5"
        output_path = filtered_shard_dir / f"{patient_id}.h5"
        source_shard_path = source_shard_dir.parent / relative_hdf5_path
        patient_manifest_row, patient_sample_rows = write_filtered_patient_shard(
            config,
            patient_id=patient_id,
            source_shard_path=source_shard_path,
            source_relative_path=relative_hdf5_path,
            output_path=output_path,
            output_relative_path=output_relative_path,
            selected_row_indices=sorted(row_indices),
            signature_source_dir=signature_source_dir,
        )
        manifest_rows.append(patient_manifest_row)
        sample_manifest_rows.extend(patient_sample_rows)

    write_filtered_shard_metadata(
        filtered_shard_dir,
        manifest_rows=manifest_rows,
        sample_manifest_rows=sample_manifest_rows,
        selection_signature=selection_signature,
        source_shard_dir=signature_source_dir or source_shard_dir,
        source_manifest_path=source_manifest_path,
    )
    logging.info("Finished writing filtered shard directory to %s", filtered_shard_dir)
    return filtered_shard_dir


def write_filtered_patient_shard(
    config: SmartSamplerConfig,
    *,
    patient_id: int,
    source_shard_path: Path,
    source_relative_path: str,
    output_path: Path,
    output_relative_path: str,
    selected_row_indices: list[int],
    signature_source_dir: Path | None,
    stage7_summary_attrs: dict[str, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_signature = _build_filtered_patient_shard_signature(
        patient_id=patient_id,
        selected_row_indices=selected_row_indices,
        source_shard_path=source_shard_path,
        source_relative_path=source_relative_path,
        output_relative_path=output_relative_path,
        config=config,
        signature_source_dir=signature_source_dir,
    )
    with h5py.File(source_shard_path, "r") as source_handle:
        guardrail(source_handle)
        filename_key = resolve_filename_key(source_handle)
        selected_indices = np.asarray(selected_row_indices, dtype=np.int64)
        images_data = np.asarray(source_handle["images"][selected_indices], dtype=np.uint8)
        masks_data = np.asarray(source_handle["masks"][selected_indices], dtype=np.uint8)
        labels_data = np.asarray(source_handle["labels"][selected_indices], dtype=np.uint8)
        patient_ids_data = np.asarray(
            source_handle["patient_ids"][selected_indices], dtype=np.int32
        )
        filenames_data = [
            _normalize_hdf5_string(value)
            for value in source_handle[filename_key][selected_indices].tolist()
        ]

        unique_patient_ids = {int(value) for value in patient_ids_data.tolist()}
        if unique_patient_ids != {patient_id}:
            raise ValueError(
                f"Filtered Stage 7 shard for patient {patient_id} contains unexpected patient ids: "
                f"{sorted(unique_patient_ids)}."
            )

        with h5py.File(output_path, "w") as dest_handle:
            for attr_name in _UPSTREAM_LINEAGE_ATTRS:
                attr_value = source_handle.attrs.get(attr_name)
                if attr_value is not None:
                    dest_handle.attrs[attr_name] = attr_value
            dest_handle.attrs["selection_signature"] = selection_signature
            dest_handle.attrs["patient_id"] = patient_id
            dest_handle.attrs["split_name"] = "TRAIN_FILTERED"
            dest_handle.attrs["source_patient_relative_hdf5_path"] = source_relative_path
            dest_handle.attrs["source_patient_shard_path"] = str(source_shard_path)
            dest_handle.attrs["source_patient_shard_sha256"] = hash_file_sha256(source_shard_path)
            dest_handle.attrs["source_patient_shard_row_indices_json"] = json.dumps(
                selected_row_indices
            )
            source_shard_selection_signature = source_handle.attrs.get("selection_signature")
            if source_shard_selection_signature is not None:
                dest_handle.attrs["source_patient_shard_selection_signature"] = (
                    source_shard_selection_signature
                )
            for attr_name, attr_value in _stage7_metadata_payload(config).items():
                dest_handle.attrs[attr_name] = attr_value
            if stage7_summary_attrs is not None:
                for attr_name, attr_value in stage7_summary_attrs.items():
                    dest_handle.attrs[attr_name] = attr_value

            str_dtype = h5py.string_dtype(encoding="utf-8")
            dest_handle.create_dataset(
                "images",
                data=images_data,
                dtype="uint8",
                compression="gzip",
                chunks=True,
            )
            dest_handle.create_dataset(
                "masks",
                data=masks_data,
                dtype="uint8",
                compression="gzip",
                chunks=True,
            )
            dest_handle.create_dataset("labels", data=labels_data, dtype="uint8")
            dest_handle.create_dataset("patient_ids", data=patient_ids_data, dtype="int32")
            dest_handle.create_dataset(
                "filenames", data=np.asarray(filenames_data, dtype=object), dtype=str_dtype
            )

    manifest_row = {
        "split": "TRAIN_FILTERED",
        "patient_id": patient_id,
        "relative_hdf5_path": output_relative_path,
        "rows": len(selected_row_indices),
        "label_0_count": int((labels_data == 0).sum()),
        "label_1_count": int((labels_data == 1).sum()),
    }
    sample_manifest_rows = [
        {
            "split": "TRAIN_FILTERED",
            "patient_id": patient_id,
            "relative_hdf5_path": output_relative_path,
            "row_in_shard": row_in_shard,
            "label": int(label),
            "filename": filename,
        }
        for row_in_shard, (label, filename) in enumerate(
            zip(labels_data.tolist(), filenames_data, strict=True)
        )
    ]
    return manifest_row, sample_manifest_rows


def read_existing_filtered_patient_selection(
    output_path: Path,
    *,
    patient_id: int,
    source_relative_hdf5_path: str,
) -> list[dict[str, Any]]:
    with h5py.File(output_path, "r") as handle:
        observed_patient_id = int(handle.attrs["patient_id"])
        if observed_patient_id != patient_id:
            raise ValueError(
                f"Existing filtered shard '{output_path}' has patient_id={observed_patient_id}, "
                f"expected {patient_id}."
            )
        selected_row_indices = json.loads(
            _normalize_hdf5_string(handle.attrs["source_patient_shard_row_indices_json"])
        )
    return [
        {
            "patient_id": patient_id,
            "relative_hdf5_path": source_relative_hdf5_path,
            "row_in_shard": int(row_index),
        }
        for row_index in selected_row_indices
    ]


def collect_filtered_shard_metadata(
    filtered_shard_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    sample_manifest_rows: list[dict[str, Any]] = []
    for output_path in sorted(filtered_shard_dir.glob("*.h5"), key=lambda path: int(path.stem)):
        output_relative_path = f"{filtered_shard_dir.name}/{output_path.name}"
        with h5py.File(output_path, "r") as handle:
            labels = np.asarray(handle["labels"][:], dtype=np.uint8)
            filenames = [_normalize_hdf5_string(value) for value in handle["filenames"][:].tolist()]
            patient_ids = np.asarray(handle["patient_ids"][:], dtype=np.int32)
            patient_id = int(handle.attrs["patient_id"])
        unique_patient_ids = {int(value) for value in patient_ids.tolist()}
        if unique_patient_ids != {patient_id}:
            raise ValueError(
                f"Filtered shard '{output_path}' contains unexpected patient ids: "
                f"{sorted(unique_patient_ids)}."
            )
        manifest_rows.append(
            {
                "split": "TRAIN_FILTERED",
                "patient_id": patient_id,
                "relative_hdf5_path": output_relative_path,
                "rows": len(labels),
                "label_0_count": int((labels == 0).sum()),
                "label_1_count": int((labels == 1).sum()),
            }
        )
        sample_manifest_rows.extend(
            {
                "split": "TRAIN_FILTERED",
                "patient_id": patient_id,
                "relative_hdf5_path": output_relative_path,
                "row_in_shard": row_in_shard,
                "label": int(label),
                "filename": filename,
            }
            for row_in_shard, (label, filename) in enumerate(
                zip(labels.tolist(), filenames, strict=True)
            )
        )
    return manifest_rows, sample_manifest_rows


def collect_filtered_shard_summary_attrs(filtered_shard_dir: Path) -> dict[str, int]:
    totals = {
        "protected_kept_samples": 0,
        "protected_positive_label_kept_samples": 0,
        "protected_mask_positive_kept_samples": 0,
        "sampled_reducible_samples": 0,
        "rejected_reducible_samples": 0,
        "total_positive_label_count": 0,
        "total_negative_label_count": 0,
        "selected_positive_label_count": 0,
        "selected_negative_label_count": 0,
        "patients_reduced_count": 0,
    }
    attr_names = tuple(totals.keys())
    for output_path in filtered_shard_dir.glob("*.h5"):
        with h5py.File(output_path, "r") as handle:
            for attr_name in attr_names:
                attr_value = handle.attrs.get(attr_name)
                if attr_value is None:
                    raise ValueError(
                        f"Filtered shard '{output_path}' is missing required summary attr "
                        f"'{attr_name}'. Regenerate the shard before rebuilding metadata."
                    )
                totals[attr_name] += int(attr_value)
    return totals


def write_filtered_shard_metadata(
    filtered_shard_dir: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    sample_manifest_rows: list[dict[str, Any]],
    selection_signature: str,
    source_shard_dir: Path,
    source_manifest_path: Path,
) -> None:
    filtered_shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(manifest_rows, schema=_FILTERED_SPLIT_MANIFEST_SCHEMA),
        filtered_shard_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(sample_manifest_rows, schema=_FILTERED_SAMPLE_MANIFEST_SCHEMA),
        filtered_shard_dir / "sample_manifest.parquet",
    )
    summary_payload = {
        "split": "TRAIN_FILTERED",
        "selection_signature": selection_signature,
        "source_shard_dir": str(source_shard_dir),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": hash_file_sha256(source_manifest_path),
        "output_shard_dir": str(filtered_shard_dir),
        "patient_count": len(manifest_rows),
        "rows": int(sum(int(row["rows"]) for row in manifest_rows)),
        "label_0_count": int(sum(int(row["label_0_count"]) for row in manifest_rows)),
        "label_1_count": int(sum(int(row["label_1_count"]) for row in manifest_rows)),
    }
    with (filtered_shard_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)


def write_sidecar_artifacts(
    config: SmartSamplerConfig,
    *,
    selection_manifest: list[dict[str, Any]],
    stats_log: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    if not config.write_sidecars:
        return None, None, None

    resolved_output_dir = output_dir or config.output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    selection_csv_path = resolved_output_dir / "train_filtered_selection.csv"
    stats_csv_path = resolved_output_dir / "patient_filter_stats.csv"
    run_config_path = resolved_output_dir / "filter_run_config.json"
    logging.info("Writing smart-sampling sidecars to %s", resolved_output_dir)

    pd.DataFrame(selection_manifest).to_csv(selection_csv_path, index=False)
    pd.DataFrame(stats_log).to_csv(stats_csv_path, index=False)
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, default=str)

    return selection_csv_path, stats_csv_path, run_config_path
