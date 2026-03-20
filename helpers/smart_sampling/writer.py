from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.index import guardrail, resolve_filename_key


def write_filtered_hdf5(
    config: SmartSamplerConfig,
    selected_indices: npt.NDArray[np.int64],
    *,
    source_h5_path: Path | None = None,
) -> Path:
    source_path = source_h5_path or config.source_h5_path
    output_path = config.output_dir / config.output_filename
    if output_path.exists() and not config.overwrite_output:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    total = len(selected_indices)

    with h5py.File(source_path, "r") as source_handle, h5py.File(output_path, "w") as dest_handle:
        guardrail(source_handle)
        filename_key = resolve_filename_key(source_handle)
        for key in ["images", "masks", "patient_ids", "labels", filename_key]:
            target_key = "filenames" if key == filename_key else key
            shape = list(source_handle[key].shape)
            shape[0] = total
            kwargs: dict[str, Any] = {}
            if key in {"images", "masks"}:
                kwargs = {"compression": "gzip", "chunks": True}
            dest_handle.create_dataset(
                target_key, shape=tuple(shape), dtype=source_handle[key].dtype, **kwargs
            )

        batch_size = 1000
        sorted_indices = np.sort(selected_indices)
        for start in tqdm(range(0, total, batch_size), desc="Writing HDF5", disable=total == 0):
            batch_indices = sorted_indices[start : start + batch_size]
            for key in ["images", "masks", "patient_ids", "labels", filename_key]:
                target_key = "filenames" if key == filename_key else key
                dest_handle[target_key][start : start + len(batch_indices)] = source_handle[key][
                    batch_indices
                ]

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
