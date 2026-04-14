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
from helpers.smart_sampling.index import ShardManifestIndex
from helpers.smart_sampling.selection import select_patient_samples
from helpers.smart_sampling.storage import (
    cleanup_local_work_dir,
    cleanup_patient_workspace,
    prepare_patient_output_path,
    prepare_storage,
    publish_outputs,
    publish_patient_output,
    published_patient_output_path,
    stage_patient_source_shard,
)
from helpers.smart_sampling.writer import (
    build_filtered_shard_selection_signature,
    collect_filtered_shard_metadata,
    collect_filtered_shard_summary_attrs,
    read_existing_filtered_patient_selection,
    write_filter_summary,
    write_filtered_patient_shard,
    write_filtered_shard_metadata,
    write_sidecar_artifacts,
)


@dataclass(frozen=True)
class SmartSamplingOutputs:
    filtered_shard_dir: Path
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
    if config.source_shard_dir is None or config.source_manifest_path is None:
        raise ValueError("Stage 7 now requires source_shard_dir and source_manifest_path.")
    logging.info("Starting Stage 7 label-aware smart sampling from %s", config.source_shard_dir)
    logging.info(
        "Using Stage 7 selector: %s",
        "GIST facility-location" if config.use_gist else "legacy adaptive coverage",
    )
    logging.info("Using Stage 7 embedding model: %s", config.model_name)
    logging.info(
        "Stage 7 preserves positive labels=%s and mask-positive rows=%s before reducible sampling",
        config.protect_positive_labels,
        config.protect_mask_positive,
    )
    logging.info("Preparing Stage 7 storage")
    storage = prepare_storage(config)
    source_shard_dir = storage.source_shard_dir
    source_manifest_path = storage.source_manifest_path
    logging.info("Building shard manifest index from %s", source_manifest_path)
    shard_index = ShardManifestIndex.build(source_shard_dir, source_manifest_path)
    logging.info(
        "Loaded %d samples across %d patients",
        shard_index.total_samples,
        len(shard_index.patient_map),
    )
    logging.info("Initializing embedding extractor on %s", config.device)
    extractor = extractor_factory(config)

    selection_manifest: list[dict[str, Any]] = []
    stats_log: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    kept_total = 0
    rejected_total = 0

    patient_ids = sorted(shard_index.patient_map.keys())
    patient_progress = tqdm(patient_ids, desc="Selecting patients", unit="patient")
    for patient_id in patient_progress:
        start_time = time.time()
        patient_reference = shard_index.patient_map[patient_id]
        patient_indices = np.arange(patient_reference.row_count, dtype=np.int64)
        existing_output_path = published_patient_output_path(config, patient_id=patient_id)
        if existing_output_path.exists() and not config.overwrite_output:
            logging.info(
                "Skipping patient %s because filtered shard already exists at %s",
                patient_id,
                existing_output_path,
            )
            resumed_selected_rows = read_existing_filtered_patient_selection(
                existing_output_path,
                patient_id=patient_id,
                source_relative_hdf5_path=patient_reference.relative_hdf5_path,
            )
            all_selected_rows.extend(resumed_selected_rows)
            resumed_selected_count = len(resumed_selected_rows)
            kept_total += resumed_selected_count
            rejected_total += len(patient_indices) - resumed_selected_count
            patient_progress.set_postfix(
                patient=patient_id,
                patches=len(patient_indices),
                kept=kept_total,
                rejected=rejected_total,
            )
            continue
        staged_source_shard_path = stage_patient_source_shard(
            config, storage, patient_reference.shard_path
        )
        result = select_patient_samples(
            str(staged_source_shard_path),
            patient_id,
            patient_indices,
            extractor,
            config,
        )
        selected_indices = result.selected_indices.astype(int).tolist()
        selected_rows = [
            {
                "patient_id": patient_id,
                "relative_hdf5_path": patient_reference.relative_hdf5_path,
                "row_in_shard": row_index,
            }
            for row_index in selected_indices
        ]
        all_selected_rows.extend(selected_rows)
        if selected_indices:
            patient_output_path = prepare_patient_output_path(config, patient_id=patient_id)
            output_relative_path = f"{config.output_filename}/{patient_id}.h5"
            write_filtered_patient_shard(
                config,
                patient_id=patient_id,
                source_shard_path=staged_source_shard_path,
                source_relative_path=patient_reference.relative_hdf5_path,
                output_path=patient_output_path,
                output_relative_path=output_relative_path,
                selected_row_indices=selected_indices,
                signature_source_dir=config.source_shard_dir,
                stage7_summary_attrs={
                    "protected_kept_samples": result.protected_count,
                    "protected_positive_label_kept_samples": result.protected_positive_label_count,
                    "protected_mask_positive_kept_samples": result.protected_mask_positive_count,
                    "sampled_reducible_samples": result.selected_reducible_count,
                    "rejected_reducible_samples": result.rejected_reducible_count,
                    "total_positive_label_count": result.total_positive_label_count,
                    "total_negative_label_count": result.total_negative_label_count,
                    "selected_positive_label_count": result.selected_positive_label_count,
                    "selected_negative_label_count": result.selected_negative_label_count,
                    "patients_reduced_count": int(len(patient_indices) != len(selected_indices)),
                },
            )
            if config.stage_outputs_locally:
                publish_patient_output(config, patient_output_path)
        cleanup_patient_workspace(
            transient_input_path=(
                staged_source_shard_path
                if staged_source_shard_path.parent.name == "input"
                else None
            ),
            transient_output_path=(patient_output_path if config.stage_outputs_locally else None),
        )
        patient_rejected = len(patient_indices) - len(selected_indices)
        kept_total += len(selected_indices)
        rejected_total += patient_rejected
        patient_progress.set_postfix(
            patient=patient_id,
            patches=len(patient_indices),
            kept=kept_total,
            rejected=rejected_total,
        )
        stats_log.append(
            {
                "patient_id": patient_id,
                "total_patches": len(patient_indices),
                "protected_count": result.protected_count,
                "protected_positive_label_count": result.protected_positive_label_count,
                "protected_mask_positive_count": result.protected_mask_positive_count,
                "reducible_count": result.reducible_count,
                "total_positive_label_count": result.total_positive_label_count,
                "total_negative_label_count": result.total_negative_label_count,
                "selected_positive_label_count": result.selected_positive_label_count,
                "selected_negative_label_count": result.selected_negative_label_count,
                "chosen_n_embed": result.chosen_n_embed,
                "k_clusters": result.k_clusters,
                "adaptive_m_target": result.adaptive_m_target,
                "heldout_count": result.heldout_count,
                "selected_m": len(selected_indices),
                "sampled_reducible_count": result.selected_reducible_count,
                "rejected_reducible_count": result.rejected_reducible_count,
                "plateau_threshold": result.plateau_threshold,
                "plateau_trigger_improvement": result.plateau_trigger_improvement,
                "plateau_trigger_keep_count": result.plateau_trigger_keep_count,
                "plateau_trigger_step": result.plateau_trigger_step,
                "plateau_stop_reason": result.plateau_stop_reason,
                "plateau_evaluation_mode": result.plateau_evaluation_mode,
                "runtime_sec": round(time.time() - start_time, 2),
                "stability_trace": str(result.stability_history),
                "retention_trace": str(result.retention_history),
            }
        )
        for protected_index in result.protected_indices.astype(int).tolist():
            selection_manifest.append(
                {
                    "patient_id": patient_id,
                    "relative_hdf5_path": patient_reference.relative_hdf5_path,
                    "row_in_shard": protected_index,
                    "embedding_source_n_embed": 0,
                    "k_clusters": 0,
                    "adaptive_m_target": result.adaptive_m_target,
                    "heldout_count": result.heldout_count,
                    "label_aware_stage7": True,
                    "selection_method": "protected_retention",
                    "selection_bucket": "protected_kept",
                    "protected_positive_label_count": result.protected_positive_label_count,
                    "protected_mask_positive_count": result.protected_mask_positive_count,
                    "plateau_threshold": result.plateau_threshold,
                    "plateau_trigger_improvement": result.plateau_trigger_improvement,
                    "plateau_trigger_keep_count": result.plateau_trigger_keep_count,
                    "plateau_trigger_step": result.plateau_trigger_step,
                    "plateau_stop_reason": result.plateau_stop_reason,
                    "plateau_evaluation_mode": result.plateau_evaluation_mode,
                }
            )
        sampled_bucket = "gist_sampled" if "gist" in result.selection_method else "legacy_sampled"
        for sampled_index in result.sampled_indices.astype(int).tolist():
            selection_manifest.append(
                {
                    "patient_id": patient_id,
                    "relative_hdf5_path": patient_reference.relative_hdf5_path,
                    "row_in_shard": sampled_index,
                    "embedding_source_n_embed": result.chosen_n_embed,
                    "k_clusters": result.k_clusters,
                    "adaptive_m_target": result.adaptive_m_target,
                    "heldout_count": result.heldout_count,
                    "label_aware_stage7": True,
                    "selection_method": result.selection_method,
                    "selection_bucket": sampled_bucket,
                    "protected_positive_label_count": result.protected_positive_label_count,
                    "protected_mask_positive_count": result.protected_mask_positive_count,
                    "plateau_threshold": result.plateau_threshold,
                    "plateau_trigger_improvement": result.plateau_trigger_improvement,
                    "plateau_trigger_keep_count": result.plateau_trigger_keep_count,
                    "plateau_trigger_step": result.plateau_trigger_step,
                    "plateau_stop_reason": result.plateau_stop_reason,
                    "plateau_evaluation_mode": result.plateau_evaluation_mode,
                }
            )

    patient_progress.close()

    unique_selected_rows = sorted(
        {
            (int(row["patient_id"]), str(row["relative_hdf5_path"]), int(row["row_in_shard"]))
            for row in all_selected_rows
        }
    )
    normalized_selected_rows = [
        {
            "patient_id": patient_id,
            "relative_hdf5_path": relative_hdf5_path,
            "row_in_shard": row_in_shard,
        }
        for patient_id, relative_hdf5_path, row_in_shard in unique_selected_rows
    ]
    kept_samples = len(normalized_selected_rows)
    total_input_samples = shard_index.total_samples
    rejected_samples = total_input_samples - kept_samples
    kept_fraction = 0.0 if total_input_samples == 0 else kept_samples / total_input_samples

    filtered_shard_dir = storage.final_output_dir / config.output_filename
    selection_signature = build_filtered_shard_selection_signature(
        config,
        normalized_selected_rows,
        source_shard_dir=source_shard_dir,
        source_manifest_path=source_manifest_path,
        signature_source_manifest_path=config.source_manifest_path,
        signature_source_dir=config.source_shard_dir,
    )
    filtered_manifest_rows, filtered_sample_manifest_rows = collect_filtered_shard_metadata(
        filtered_shard_dir
    )
    summary_attrs = collect_filtered_shard_summary_attrs(filtered_shard_dir)
    total_positive_label_count = summary_attrs["total_positive_label_count"]
    total_negative_label_count = summary_attrs["total_negative_label_count"]
    selected_positive_label_count = summary_attrs["selected_positive_label_count"]
    selected_negative_label_count = summary_attrs["selected_negative_label_count"]
    summary_payload = {
        "total_input_samples": total_input_samples,
        "kept_samples": kept_samples,
        "rejected_samples": rejected_samples,
        "kept_fraction": kept_fraction,
        "patient_count": len(patient_ids),
        "patients_reduced_count": summary_attrs["patients_reduced_count"],
        "label_aware_stage7": True,
        "protected_kept_samples": summary_attrs["protected_kept_samples"],
        "protected_positive_label_kept_samples": summary_attrs[
            "protected_positive_label_kept_samples"
        ],
        "protected_mask_positive_kept_samples": summary_attrs[
            "protected_mask_positive_kept_samples"
        ],
        "sampled_reducible_samples": summary_attrs["sampled_reducible_samples"],
        "rejected_reducible_samples": summary_attrs["rejected_reducible_samples"],
        "total_positive_label_count": total_positive_label_count,
        "total_negative_label_count": total_negative_label_count,
        "selected_positive_label_count": selected_positive_label_count,
        "selected_negative_label_count": selected_negative_label_count,
        "selected_positive_label_fraction": (
            0.0
            if total_positive_label_count == 0
            else selected_positive_label_count / total_positive_label_count
        ),
        "selected_negative_label_fraction": (
            0.0
            if total_negative_label_count == 0
            else selected_negative_label_count / total_negative_label_count
        ),
        "holdout_evaluation_mode": "within_patient_patch_holdout",
        "source_shard_dir": str(config.source_shard_dir),
        "source_manifest_path": str(config.source_manifest_path),
        "filtered_shard_dirname": config.output_filename,
        "model_name": config.model_name,
        "use_gist": config.use_gist,
    }
    write_filtered_shard_metadata(
        filtered_shard_dir,
        manifest_rows=filtered_manifest_rows,
        sample_manifest_rows=filtered_sample_manifest_rows,
        selection_signature=selection_signature,
        source_shard_dir=config.source_shard_dir,
        source_manifest_path=config.source_manifest_path,
    )
    selection_csv_path, stats_csv_path, run_config_path = write_sidecar_artifacts(
        config,
        selection_manifest=selection_manifest,
        stats_log=stats_log,
        output_dir=storage.sidecar_output_dir,
    )
    summary_json_path = write_filter_summary(summary_payload, output_dir=storage.sidecar_output_dir)
    if storage.should_publish_outputs:
        selection_csv_path, stats_csv_path, run_config_path = publish_outputs(
            config,
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
        summary_attrs["patients_reduced_count"],
    )
    if rejected_samples == 0:
        logging.warning(
            "Smart sampling kept every patch in this run; "
            "file size differences may come only from HDF5 compression."
        )
    cleanup_local_work_dir(config)
    return SmartSamplingOutputs(
        filtered_shard_dir=filtered_shard_dir,
        selection_csv_path=selection_csv_path,
        stats_csv_path=stats_csv_path,
        run_config_path=run_config_path,
        summary_json_path=summary_json_path,
        total_input_samples=total_input_samples,
        selected_sample_count=kept_samples,
        rejected_sample_count=rejected_samples,
        kept_fraction=kept_fraction,
        patient_count=len(patient_ids),
        patients_reduced_count=summary_attrs["patients_reduced_count"],
    )
