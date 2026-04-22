from __future__ import annotations

import hashlib
import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from helpers.crossfold.logging import ProgressReporter
from helpers.provenance import collect_hdf5_provenance


class NormalizerProtocol(Protocol):
    def transform(self, image_rgb: Any) -> Any: ...


@dataclass(frozen=True)
class SplitHDF5WriteConfig:
    split_df: pd.DataFrame
    source_hdf5_path: Path
    output_path: Path
    normalizer: NormalizerProtocol | None
    normalization_method: str
    source_hdf5_provenance: dict[str, Any] | None = None
    hdf5_compression: str = "NONE"
    copy_batch_size: int = 256
    overwrite: bool = False


@dataclass(frozen=True)
class _OrderedSplitInputs:
    source_paths: npt.NDArray[np.str_]
    source_row_indices: npt.NDArray[np.int64]
    labels: npt.NDArray[np.uint8]
    patient_ids: npt.NDArray[np.int32]
    filenames: npt.NDArray[np.object_]


def _build_hdf5_split_signature(
    split_df: pd.DataFrame,
    *,
    source_hdf5_provenance: dict[str, Any],
    normalization_method: str,
) -> str:
    payload = {
        "source_hdf5": source_hdf5_provenance,
        "normalization_method": normalization_method,
        "rows": split_df[
            [
                column
                for column in (
                    "filename",
                    "patient_id",
                    "label",
                    "source_hdf5_path",
                    "source_row_index",
                )
                if column in split_df.columns
            ]
        ].to_dict("records"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_existing_split_hdf5(output_path: Path, expected_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        if handle.attrs.get("source_signature") != expected_signature:
            raise ValueError(
                f"Existing split HDF5 '{output_path}' does not match the current split inputs. "
                "Enable overwrite or remove the stale file."
            )
    return output_path


def _normalize_image_array(
    image: npt.NDArray[np.uint8],
    normalizer: NormalizerProtocol | None,
) -> npt.NDArray[np.uint8]:
    if normalizer is None:
        return image.astype(np.uint8, copy=False)
    transformed = normalizer.transform(image)
    return np.asarray(transformed, dtype=np.uint8)


def _resolve_hdf5_compression(compression: str) -> str | None:
    normalized = compression.strip().lower()
    if normalized == "none":
        return None
    if normalized in {"gzip", "lzf"}:
        return normalized
    raise ValueError(f"Unsupported Stage 5 HDF5 compression: {compression}")


def _load_rows_by_source_index(
    dataset: Any,
    source_indices: list[int],
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


def _resolve_source_paths(batch_df: pd.DataFrame, fallback_source_hdf5_path: Path) -> pd.Series:
    if "source_hdf5_path" in batch_df.columns:
        return batch_df["source_hdf5_path"].astype(str)
    return pd.Series([str(fallback_source_hdf5_path)] * len(batch_df), index=batch_df.index)


def _prepare_ordered_split_inputs(
    split_df: pd.DataFrame,
    fallback_source_hdf5_path: Path,
) -> _OrderedSplitInputs:
    ordered_split_df = split_df.reset_index(drop=True)
    return _OrderedSplitInputs(
        source_paths=_resolve_source_paths(ordered_split_df, fallback_source_hdf5_path).to_numpy(
            dtype=str
        ),
        source_row_indices=ordered_split_df["source_row_index"].to_numpy(dtype=np.int64),
        labels=ordered_split_df["label"].to_numpy(dtype=np.uint8),
        patient_ids=ordered_split_df["patient_id"].to_numpy(dtype=np.int32),
        filenames=ordered_split_df["filename"].astype(str).to_numpy(dtype=object),
    )


def _copy_split_batch_from_handles(
    *,
    source_handles: dict[str, h5py.File],
    ordered_inputs: _OrderedSplitInputs,
    batch_start: int,
    batch_stop: int,
    first_image_shape: tuple[int, ...],
    first_mask_shape: tuple[int, ...],
    normalizer: NormalizerProtocol | None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    batch_source_paths = ordered_inputs.source_paths[batch_start:batch_stop]
    batch_source_indices = ordered_inputs.source_row_indices[batch_start:batch_stop]
    image_batch = np.empty((len(batch_source_indices),) + first_image_shape, dtype=np.uint8)
    mask_batch = np.empty((len(batch_source_indices),) + first_mask_shape, dtype=np.uint8)
    positions_by_source: dict[str, list[int]] = {}
    for output_offset, source_path in enumerate(batch_source_paths.tolist()):
        positions_by_source.setdefault(str(source_path), []).append(output_offset)
    for source_path, group_positions in positions_by_source.items():
        source_indices = sorted(
            int(batch_source_indices[position]) for position in group_positions
        )
        source_handle = source_handles[source_path]
        images_by_index = _load_rows_by_source_index(source_handle["images"], source_indices)
        masks_by_index = _load_rows_by_source_index(source_handle["masks"], source_indices)

        for output_offset in group_positions:
            source_index = int(batch_source_indices[output_offset])
            image_batch[output_offset] = _normalize_image_array(
                images_by_index[source_index],
                normalizer,
            )
            mask_batch[output_offset] = masks_by_index[source_index]
    return image_batch, mask_batch


def write_split_hdf5(config: SplitHDF5WriteConfig) -> Path:
    if config.split_df.empty:
        raise ValueError(
            f"Cannot create '{config.output_path.name}' from an empty split dataframe."
        )

    resolved_source_hdf5_provenance = config.source_hdf5_provenance or collect_hdf5_provenance(
        config.source_hdf5_path
    )
    source_signature = _build_hdf5_split_signature(
        config.split_df,
        source_hdf5_provenance=resolved_source_hdf5_provenance,
        normalization_method=config.normalization_method,
    )
    if config.output_path.exists() and not config.overwrite:
        return _validate_existing_split_hdf5(config.output_path, source_signature)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    ordered_split_df = config.split_df.reset_index(drop=True)
    ordered_inputs = _prepare_ordered_split_inputs(ordered_split_df, config.source_hdf5_path)
    resolved_compression = _resolve_hdf5_compression(config.hdf5_compression)
    reporter = ProgressReporter(
        "Stage 5 split write",
        len(ordered_split_df),
        "rows",
        context=config.output_path.name,
    )
    reporter.log_start(
        f"rows={len(ordered_split_df)} | compression={resolved_compression or 'none'} "
        f"| batch_size={config.copy_batch_size}"
    )

    unique_source_paths = list(dict.fromkeys(ordered_inputs.source_paths.tolist()))
    with ExitStack() as stack, h5py.File(config.output_path, "w") as dest_handle:
        source_handles = {
            source_path: stack.enter_context(h5py.File(Path(source_path), "r"))
            for source_path in unique_source_paths
        }
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["source_hdf5_sha256"] = resolved_source_hdf5_provenance["sha256"]
        for attr_name in (
            "upstream_source_signature",
            "stage4_cleaning_manifest_path",
            "stage4_cleaning_manifest_sha256",
            "stage4_cleaning_selected_rows",
        ):
            attr_value = resolved_source_hdf5_provenance.get("attrs", {}).get(attr_name)
            if attr_value is not None:
                dest_handle.attrs[attr_name] = attr_value

        first_source_path = ordered_inputs.source_paths[0]
        first_index = int(ordered_inputs.source_row_indices[0])
        first_source_handle = source_handles[str(first_source_path)]
        first_image = np.asarray(first_source_handle["images"][first_index])
        first_mask = np.asarray(first_source_handle["masks"][first_index])
        images = dest_handle.create_dataset(
            "images",
            shape=(len(ordered_split_df),) + first_image.shape,
            dtype="uint8",
            compression=resolved_compression,
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(ordered_split_df),) + first_mask.shape,
            dtype="uint8",
            compression=resolved_compression,
            chunks=True,
        )
        labels = dest_handle.create_dataset("labels", shape=(len(ordered_split_df),), dtype="uint8")
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(ordered_split_df),), dtype="int32"
        )
        filenames = dest_handle.create_dataset(
            "filenames", shape=(len(ordered_split_df),), dtype=str_dtype
        )

        for batch_start in range(0, len(ordered_split_df), config.copy_batch_size):
            batch_stop = min(batch_start + config.copy_batch_size, len(ordered_split_df))
            image_batch, mask_batch = _copy_split_batch_from_handles(
                source_handles=source_handles,
                ordered_inputs=ordered_inputs,
                batch_start=batch_start,
                batch_stop=batch_stop,
                first_image_shape=first_image.shape,
                first_mask_shape=first_mask.shape,
                normalizer=config.normalizer,
            )

            images[batch_start:batch_stop] = image_batch
            masks[batch_start:batch_stop] = mask_batch
            labels[batch_start:batch_stop] = ordered_inputs.labels[batch_start:batch_stop]
            patient_ids[batch_start:batch_stop] = ordered_inputs.patient_ids[batch_start:batch_stop]
            filenames[batch_start:batch_stop] = ordered_inputs.filenames[batch_start:batch_stop]
            reporter.log(
                completed_units=batch_stop,
                extra_parts=[f"remaining={len(ordered_split_df) - batch_stop} rows"],
            )

    logging.info(
        "Stage 5 split verify: %s | rows=%s",
        config.output_path.name,
        len(ordered_split_df),
    )

    return config.output_path


def verify_split_hdf5_integrity(output_path: Path, split_df: pd.DataFrame) -> None:
    ordered_split_df = split_df.reset_index(drop=True)
    with h5py.File(output_path, "r") as handle:
        labels_dataset = handle["labels"]
        patient_ids_dataset = handle["patient_ids"]
        filenames_dataset = handle["filenames"]

        if len(labels_dataset) != len(ordered_split_df):
            raise ValueError(
                "Integrity FAILED for "
                f"{output_path.name}: row count does not match split dataframe."
            )

        observed_labels = np.asarray(labels_dataset[:], dtype=np.uint8)
        observed_patient_ids = np.asarray(patient_ids_dataset[:], dtype=np.int32)
        observed_filenames_raw = filenames_dataset[:]

    expected_labels = ordered_split_df["label"].to_numpy(dtype=np.uint8)
    expected_patient_ids = ordered_split_df["patient_id"].to_numpy(dtype=np.int32)
    expected_filenames = ordered_split_df["filename"].astype(str).to_numpy(dtype=object)
    observed_filenames = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in observed_filenames_raw
        ],
        dtype=object,
    )

    if not (
        np.array_equal(observed_labels, expected_labels)
        and np.array_equal(observed_patient_ids, expected_patient_ids)
        and np.array_equal(observed_filenames, expected_filenames)
    ):
        raise ValueError(
            "Integrity FAILED for "
            f"{output_path.name}: HDF5 row metadata does not match split dataframe."
        )
