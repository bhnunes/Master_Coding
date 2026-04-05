from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.embeddings import EmbeddingExtractor
from helpers.smart_sampling.index import H5MetadataIndex
from helpers.smart_sampling.selection import select_patient_samples
from helpers.smart_sampling.storage import cleanup_local_work_dir, prepare_storage, publish_outputs
from helpers.smart_sampling.writer import (
    write_filter_summary,
    write_filtered_hdf5,
    write_sidecar_artifacts,
)


@dataclass(frozen=True)
class SmartSamplingOutputs:
    filtered_h5_path: Path
    selection_csv_path: Path | None
    stats_csv_path: Path | None
    run_config_path: Path | None
    summary_json_path: Path | None
    total_input_samples: int
    selected_sample_count: int
    rejected_sample_count: int
    kept_fraction: float
    patient_count: int
    patients_reduced_count: int


def run_smart_sampling_pipeline(
    config: SmartSamplerConfig,
    *,
    extractor_factory: type[EmbeddingExtractor] | Any = EmbeddingExtractor,
) -> SmartSamplingOutputs:
    logging.info("Starting Stage 8 smart sampling from %s", config.source_h5_path)
    logging.info("Preparing Stage 7 storage")
    storage = prepare_storage(config)
    source_h5_path = storage.source_h5_path
    logging.info("Building HDF5 patient index from %s", source_h5_path)
    h5_index = H5MetadataIndex.build(source_h5_path)
    logging.info(
        "Loaded %d samples across %d patients",
        h5_index.total_samples,
        len(h5_index.patient_map),
    )
    logging.info("Initializing embedding extractor on %s", config.device)
    extractor = extractor_factory(config)

    selection_manifest: list[dict[str, Any]] = []
    stats_log: list[dict[str, Any]] = []
    all_selected_indices: list[int] = []
    selected_total = 0
    rejected_total = 0
    reduced_patients = 0

    patient_ids = sorted(h5_index.patient_map.keys())
    patient_progress = tqdm(patient_ids, desc="Selecting patients", unit="patient")
    for patient_id in patient_progress:
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
        patient_rejected = len(patient_indices) - len(selected_indices)
        selected_total += len(selected_indices)
        rejected_total += patient_rejected
        if patient_rejected > 0:
            reduced_patients += 1
        patient_progress.set_postfix(
            patient=patient_id,
            patches=len(patient_indices),
            kept=selected_total,
            rejected=rejected_total,
        )
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

    patient_progress.close()

    unique_selected_indices = np.asarray(sorted(set(all_selected_indices)), dtype=np.int64)
    kept_samples = len(unique_selected_indices)
    total_input_samples = h5_index.total_samples
    rejected_samples = total_input_samples - kept_samples
    kept_fraction = 0.0 if total_input_samples == 0 else kept_samples / total_input_samples
    summary_payload = {
        "total_input_samples": total_input_samples,
        "kept_samples": kept_samples,
        "rejected_samples": rejected_samples,
        "kept_fraction": kept_fraction,
        "patient_count": len(patient_ids),
        "patients_reduced_count": reduced_patients,
        "source_h5_path": str(config.source_h5_path),
        "filtered_h5_filename": config.output_filename,
    }

    filtered_h5_path = write_filtered_hdf5(
        config,
        unique_selected_indices,
        source_h5_path=source_h5_path,
        output_dir=storage.output_dir,
        signature_source_path=config.source_h5_path,
    )
    selection_csv_path, stats_csv_path, run_config_path = write_sidecar_artifacts(
        config,
        selection_manifest=selection_manifest,
        stats_log=stats_log,
        output_dir=storage.output_dir,
    )
    summary_json_path = write_filter_summary(summary_payload, output_dir=storage.output_dir)
    if storage.should_publish_outputs:
        filtered_h5_path, selection_csv_path, stats_csv_path, run_config_path = publish_outputs(
            config,
            filtered_h5_path=filtered_h5_path,
            selection_csv_path=selection_csv_path,
            stats_csv_path=stats_csv_path,
            run_config_path=run_config_path,
        )
        summary_json_path = write_filter_summary(summary_payload, output_dir=config.output_dir)
    logging.info(
        (
            "Stage 7 summary: kept %d/%d patches "
            "(rejected %d, %.2f%% kept) across %d patients; %d patients reduced"
        ),
        kept_samples,
        total_input_samples,
        rejected_samples,
        kept_fraction * 100.0,
        len(patient_ids),
        reduced_patients,
    )
    if rejected_samples == 0:
        logging.warning(
            "Smart sampling kept every patch in this run; "
            "file size differences may come only from HDF5 compression."
        )
    cleanup_local_work_dir(config)
    return SmartSamplingOutputs(
        filtered_h5_path=filtered_h5_path,
        selection_csv_path=selection_csv_path,
        stats_csv_path=stats_csv_path,
        run_config_path=run_config_path,
        summary_json_path=summary_json_path,
        total_input_samples=total_input_samples,
        selected_sample_count=kept_samples,
        rejected_sample_count=rejected_samples,
        kept_fraction=kept_fraction,
        patient_count=len(patient_ids),
        patients_reduced_count=reduced_patients,
    )
