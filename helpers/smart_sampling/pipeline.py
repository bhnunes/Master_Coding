from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from helpers.extraction.master_manifest import STAGE7_STAGE_NAME, MasterManifest
from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.embeddings import EmbeddingExtractor
from helpers.smart_sampling.index import CanonicalPatientReference, MasterManifestIndex
from helpers.smart_sampling.selection import select_patient_samples
from helpers.smart_sampling.storage import (
    cleanup_local_work_dir,
    cleanup_patient_workspace,
    prepare_storage,
    publish_outputs,
    stage_patient_source_shard,
)
from helpers.smart_sampling.writer import (
    write_filter_summary,
    write_sidecar_artifacts,
)


@dataclass(frozen=True)
class SmartSamplingOutputs:
    output_dir: Path
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
    logging.info(
        "Starting Stage 7 label-aware smart sampling from %s",
        config.master_manifest_path,
    )
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
    logging.info("Building TRAIN patient index from %s", config.master_manifest_path)
    shard_index = MasterManifestIndex.build(config.master_manifest_path)
    logging.info(
        "Loaded %d samples across %d patients",
        shard_index.total_samples,
        len(shard_index.patient_map),
    )
    logging.info("Initializing embedding extractor on %s", config.device)
    extractor = extractor_factory(config)

    selection_manifest: list[dict[str, Any]] = []
    stats_log: list[dict[str, Any]] = []
    sampling_decisions: list[dict[str, object]] = []
    kept_total = 0
    rejected_total = 0
    protected_kept_total = 0
    protected_positive_label_kept_total = 0
    protected_mask_positive_kept_total = 0
    sampled_reducible_total = 0
    rejected_reducible_total = 0
    total_positive_label_count = 0
    total_negative_label_count = 0
    selected_positive_label_count = 0
    selected_negative_label_count = 0
    patients_reduced_count = 0

    patient_ids = sorted(shard_index.patient_map.keys())
    patient_progress = tqdm(patient_ids, desc="Selecting patients", unit="patient")
    for patient_id in patient_progress:
        start_time = time.time()
        patient_reference = shard_index.patient_map[patient_id]
        staged_source_shard_path = stage_patient_source_shard(
            config,
            storage,
            patient_reference.source_hdf5_path,
        )
        result = select_patient_samples(
            str(staged_source_shard_path),
            patient_id,
            patient_reference.source_row_indices,
            extractor,
            config,
        )
        selected_index_set = {int(index) for index in result.selected_indices.tolist()}
        protected_index_set = {int(index) for index in result.protected_indices.tolist()}
        sampled_index_set = {int(index) for index in result.sampled_indices.tolist()}
        sampling_decisions.extend(
            _build_patient_sampling_decisions(
                patient_reference=patient_reference,
                selected_index_set=selected_index_set,
                protected_index_set=protected_index_set,
                sampled_index_set=sampled_index_set,
            )
        )
        selection_manifest.extend(
            _build_patient_selection_manifest_rows(
                patient_reference=patient_reference,
                result=result,
            )
        )
        cleanup_patient_workspace(
            transient_input_path=(
                staged_source_shard_path
                if staged_source_shard_path.parent.name == "input"
                else None
            ),
            transient_output_path=None,
        )
        patient_row_count = len(patient_reference.source_row_indices)
        selected_indices = result.selected_indices.astype(int).tolist()
        patient_rejected = patient_row_count - len(selected_indices)
        kept_total += len(selected_indices)
        rejected_total += patient_rejected
        protected_kept_total += result.protected_count
        protected_positive_label_kept_total += result.protected_positive_label_count
        protected_mask_positive_kept_total += result.protected_mask_positive_count
        sampled_reducible_total += result.selected_reducible_count
        rejected_reducible_total += result.rejected_reducible_count
        total_positive_label_count += result.total_positive_label_count
        total_negative_label_count += result.total_negative_label_count
        selected_positive_label_count += result.selected_positive_label_count
        selected_negative_label_count += result.selected_negative_label_count
        patients_reduced_count += int(patient_row_count != len(selected_indices))
        patient_progress.set_postfix(
            patient=patient_id,
            patches=patient_row_count,
            kept=kept_total,
            rejected=rejected_total,
        )
        stats_log.append(
            {
                "patient_id": patient_id,
                "total_patches": patient_row_count,
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

    patient_progress.close()

    kept_samples = kept_total
    total_input_samples = shard_index.total_samples
    rejected_samples = total_input_samples - kept_samples
    kept_fraction = 0.0 if total_input_samples == 0 else kept_samples / total_input_samples

    summary_payload = {
        "total_input_samples": total_input_samples,
        "kept_samples": kept_samples,
        "rejected_samples": rejected_samples,
        "kept_fraction": kept_fraction,
        "patient_count": len(patient_ids),
        "patients_reduced_count": patients_reduced_count,
        "label_aware_stage7": True,
        "protected_kept_samples": protected_kept_total,
        "protected_positive_label_kept_samples": protected_positive_label_kept_total,
        "protected_mask_positive_kept_samples": protected_mask_positive_kept_total,
        "sampled_reducible_samples": sampled_reducible_total,
        "rejected_reducible_samples": rejected_reducible_total,
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
        "master_manifest_path": str(config.master_manifest_path),
        "model_name": config.model_name,
        "use_gist": config.use_gist,
    }
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
        patients_reduced_count,
    )
    if rejected_samples == 0:
        logging.warning(
            "Smart sampling kept every patch in this run; "
            "file size differences may come only from HDF5 compression."
        )
    master_manifest = MasterManifest(config.master_manifest_path)
    master_manifest.create_run(
        stage_name=STAGE7_STAGE_NAME,
        config_path=run_config_path,
        input_summary_json_path=summary_json_path,
    )
    master_manifest.update_stage7_sampling_decisions(decisions=sampling_decisions)
    cleanup_local_work_dir(config)
    return SmartSamplingOutputs(
        output_dir=storage.final_output_dir,
        selection_csv_path=selection_csv_path,
        stats_csv_path=stats_csv_path,
        run_config_path=run_config_path,
        summary_json_path=summary_json_path,
        total_input_samples=total_input_samples,
        selected_sample_count=kept_samples,
        rejected_sample_count=rejected_samples,
        kept_fraction=kept_fraction,
        patient_count=len(patient_ids),
        patients_reduced_count=patients_reduced_count,
    )


def _build_patient_sampling_decisions(
    *,
    patient_reference: CanonicalPatientReference,
    selected_index_set: set[int],
    protected_index_set: set[int],
    sampled_index_set: set[int],
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for row in patient_reference.rows:
        sampling_decision = "rejected_reducible"
        if row.source_row_index in protected_index_set:
            sampling_decision = "protected_kept"
        elif row.source_row_index in sampled_index_set:
            sampling_decision = "sampled_kept"
        decisions.append(
            {
                "filename": row.filename,
                "patient_id": patient_reference.patient_id,
                "label": row.label,
                "source_hdf5_path": str(patient_reference.source_hdf5_path),
                "source_row_index": row.source_row_index,
                "sampling_decision": sampling_decision,
                "is_stage7_selected": row.source_row_index in selected_index_set,
            }
        )
    return decisions


def _build_patient_selection_manifest_rows(
    *,
    patient_reference: CanonicalPatientReference,
    result: Any,
) -> list[dict[str, Any]]:
    rows_by_index = {row.source_row_index: row for row in patient_reference.rows}
    manifest_rows: list[dict[str, Any]] = []
    for protected_index in result.protected_indices.astype(int).tolist():
        row = rows_by_index[protected_index]
        manifest_rows.append(
            {
                "patient_id": patient_reference.patient_id,
                "source_hdf5_path": str(patient_reference.source_hdf5_path),
                "source_row_index": protected_index,
                "label": row.label,
                "filename": row.filename,
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
    sampled_bucket = "gist_sampled" if "gist" in str(result.selection_method) else "legacy_sampled"
    for sampled_index in result.sampled_indices.astype(int).tolist():
        row = rows_by_index[sampled_index]
        manifest_rows.append(
            {
                "patient_id": patient_reference.patient_id,
                "source_hdf5_path": str(patient_reference.source_hdf5_path),
                "source_row_index": sampled_index,
                "label": row.label,
                "filename": row.filename,
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
    return manifest_rows
