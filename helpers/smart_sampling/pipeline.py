from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.embeddings import EmbeddingExtractor
from helpers.smart_sampling.index import H5MetadataIndex
from helpers.smart_sampling.selection import select_patient_samples
from helpers.smart_sampling.storage import prepare_source_h5
from helpers.smart_sampling.writer import write_filtered_hdf5, write_sidecar_artifacts


@dataclass(frozen=True)
class SmartSamplingOutputs:
    filtered_h5_path: Path
    selection_csv_path: Path | None
    stats_csv_path: Path | None
    run_config_path: Path | None
    selected_sample_count: int


def run_smart_sampling_pipeline(
    config: SmartSamplerConfig,
    *,
    extractor_factory: type[EmbeddingExtractor] | Any = EmbeddingExtractor,
) -> SmartSamplingOutputs:
    logging.info("Starting Stage 8 smart sampling from %s", config.source_h5_path)
    source_h5_path = prepare_source_h5(config)
    h5_index = H5MetadataIndex.build(source_h5_path)
    extractor = extractor_factory(config)

    selection_manifest: list[dict[str, Any]] = []
    stats_log: list[dict[str, Any]] = []
    all_selected_indices: list[int] = []

    for patient_id in sorted(h5_index.patient_map.keys()):
        start_time = time.time()
        patient_indices = h5_index.patient_map[patient_id]
        result = select_patient_samples(
            str(source_h5_path),
            patient_id,
            patient_indices,
            extractor,
            config,
        )
        selected_indices = result.selected_indices.astype(int).tolist()
        all_selected_indices.extend(selected_indices)
        stats_log.append(
            {
                "patient_id": patient_id,
                "total_patches": len(patient_indices),
                "chosen_n_embed": result.chosen_n_embed,
                "k_clusters": result.k_clusters,
                "selected_m": len(selected_indices),
                "runtime_sec": round(time.time() - start_time, 2),
                "stability_trace": str(result.stability_history),
            }
        )
        for selected_index in selected_indices:
            selection_manifest.append(
                {
                    "patient_id": patient_id,
                    "global_index": selected_index,
                    "embedding_source_n_embed": result.chosen_n_embed,
                    "k_clusters": result.k_clusters,
                    "selection_method": result.selection_method,
                }
            )

    unique_selected_indices = np.asarray(sorted(set(all_selected_indices)), dtype=np.int64)
    filtered_h5_path = write_filtered_hdf5(
        config,
        unique_selected_indices,
        source_h5_path=source_h5_path,
    )
    selection_csv_path, stats_csv_path, run_config_path = write_sidecar_artifacts(
        config,
        selection_manifest=selection_manifest,
        stats_log=stats_log,
    )
    return SmartSamplingOutputs(
        filtered_h5_path=filtered_h5_path,
        selection_csv_path=selection_csv_path,
        stats_csv_path=stats_csv_path,
        run_config_path=run_config_path,
        selected_sample_count=len(unique_selected_indices),
    )
