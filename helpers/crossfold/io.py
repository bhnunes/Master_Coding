from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from helpers.provenance import collect_hdf5_provenance


class NormalizerProtocol(Protocol):
    def transform(self, image_rgb: Any) -> Any: ...


def _build_hdf5_split_signature(
    split_df: pd.DataFrame,
    *,
    source_hdf5_path: Path,
    normalization_method: str,
) -> str:
    payload = {
        "source_hdf5": collect_hdf5_provenance(source_hdf5_path),
        "normalization_method": normalization_method,
        "rows": split_df[["filename", "patient_id", "label", "source_row_index"]].to_dict(
            "records"
        ),
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


def write_split_hdf5(
    *,
    split_df: pd.DataFrame,
    source_hdf5_path: Path,
    output_path: Path,
    normalizer: NormalizerProtocol | None,
    normalization_method: str,
    overwrite: bool,
) -> Path:
    if split_df.empty:
        raise ValueError(f"Cannot create '{output_path.name}' from an empty split dataframe.")

    source_signature = _build_hdf5_split_signature(
        split_df,
        source_hdf5_path=source_hdf5_path,
        normalization_method=normalization_method,
    )
    if output_path.exists() and not overwrite:
        return _validate_existing_split_hdf5(output_path, source_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    ordered_split_df = split_df.reset_index(drop=True)

    with (
        h5py.File(source_hdf5_path, "r") as source_handle,
        h5py.File(output_path, "w") as dest_handle,
    ):
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["source_hdf5_sha256"] = collect_hdf5_provenance(source_hdf5_path)[
            "sha256"
        ]
        for attr_name in (
            "upstream_source_signature",
            "stage4_cleaning_manifest_path",
            "stage4_cleaning_manifest_sha256",
            "stage4_cleaning_selected_rows",
        ):
            attr_value = source_handle.attrs.get(attr_name)
            if attr_value is not None:
                dest_handle.attrs[attr_name] = attr_value

        first_index = int(ordered_split_df.iloc[0]["source_row_index"])
        first_image = np.asarray(source_handle["images"][first_index])
        first_mask = np.asarray(source_handle["masks"][first_index])
        images = dest_handle.create_dataset(
            "images",
            shape=(len(ordered_split_df),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(ordered_split_df),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = dest_handle.create_dataset("labels", shape=(len(ordered_split_df),), dtype="uint8")
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(ordered_split_df),), dtype="int32"
        )
        filenames = dest_handle.create_dataset(
            "filenames", shape=(len(ordered_split_df),), dtype=str_dtype
        )

        for output_index, row in enumerate(ordered_split_df.itertuples(index=False)):
            source_index = int(row.source_row_index)
            image = np.asarray(source_handle["images"][source_index], dtype=np.uint8)
            mask = np.asarray(source_handle["masks"][source_index], dtype=np.uint8)
            images[output_index] = _normalize_image_array(image, normalizer)
            masks[output_index] = mask
            labels[output_index] = int(row.label)
            patient_ids[output_index] = int(row.patient_id)
            filenames[output_index] = str(row.filename)

    return output_path


def verify_split_hdf5_integrity(output_path: Path, split_df: pd.DataFrame) -> None:
    ordered_split_df = split_df.reset_index(drop=True)
    with h5py.File(output_path, "r") as handle:
        if len(handle["labels"]) != len(ordered_split_df):
            raise ValueError(
                "Integrity FAILED for "
                f"{output_path.name}: row count does not match split dataframe."
            )
        observed = [
            {
                "label": int(handle["labels"][index]),
                "patient_id": int(handle["patient_ids"][index]),
                "filename": (
                    handle["filenames"][index].decode("utf-8")
                    if isinstance(handle["filenames"][index], bytes)
                    else str(handle["filenames"][index])
                ),
            }
            for index in range(len(ordered_split_df))
        ]
    expected = ordered_split_df[["label", "patient_id", "filename"]].to_dict("records")
    if observed != expected:
        raise ValueError(
            "Integrity FAILED for "
            f"{output_path.name}: HDF5 row metadata does not match split dataframe."
        )
