from __future__ import annotations

import hashlib
import json
import logging
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


def _validate_existing_filtered_hdf5(output_path: Path, expected_signature: str) -> Path:
    with h5py.File(output_path, "r") as handle:
        existing_signature = handle.attrs.get("selection_signature")
        if existing_signature != expected_signature:
            raise ValueError(
                f"Existing filtered HDF5 '{output_path}' does not match the current selection. "
                "Enable overwrite or remove the stale file."
            )
    return output_path


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
