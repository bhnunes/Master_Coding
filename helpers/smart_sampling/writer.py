from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

from helpers.provenance import collect_hdf5_provenance
from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.index import guardrail, resolve_filename_key


def _hash_json_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _build_sampling_signature(
    selected_indices: npt.NDArray[np.int64],
    *,
    source_path: Path,
    output_filename: str,
) -> str:
    source_provenance = collect_hdf5_provenance(source_path)
    payload = {
        "selected_indices": np.asarray(selected_indices, dtype=np.int64).tolist(),
        "source_path": str(source_path),
        "source_provenance": source_provenance,
        "output_filename": output_filename,
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


def write_filtered_hdf5(
    config: SmartSamplerConfig,
    selected_indices: npt.NDArray[np.int64],
    *,
    source_h5_path: Path | None = None,
) -> Path:
    source_path = source_h5_path or config.source_h5_path
    output_path = config.output_dir / config.output_filename
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selection_signature = _build_sampling_signature(
        selected_indices,
        source_path=source_path,
        output_filename=config.output_filename,
    )
    if output_path.exists() and not config.overwrite_output:
        return _validate_existing_filtered_hdf5(output_path, selection_signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(selected_indices)

    with h5py.File(source_path, "r") as source_handle, h5py.File(output_path, "w") as dest_handle:
        dest_handle.attrs["selection_signature"] = selection_signature
        dest_handle.attrs["source_hdf5_sha256"] = collect_hdf5_provenance(source_path)["sha256"]
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

    return output_path


def write_sidecar_artifacts(
    config: SmartSamplerConfig,
    *,
    selection_manifest: list[dict[str, Any]],
    stats_log: list[dict[str, Any]],
) -> tuple[Path | None, Path | None, Path | None]:
    if not config.write_sidecars:
        return None, None, None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    selection_csv_path = config.output_dir / "train_filtered_selection.csv"
    stats_csv_path = config.output_dir / "patient_filter_stats.csv"
    run_config_path = config.output_dir / "filter_run_config.json"

    pd.DataFrame(selection_manifest).to_csv(selection_csv_path, index=False)
    pd.DataFrame(stats_log).to_csv(stats_csv_path, index=False)
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, default=str)

    return selection_csv_path, stats_csv_path, run_config_path
