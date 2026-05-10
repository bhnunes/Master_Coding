from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.data import (
    ValidationDatasetLayout,
    collect_validation_provenance,
    create_validation_dataloader,
    setup_validation_data,
    summarize_validation_data,
)
from helpers.ensemble_optimizer.metadata import (
    SelectedModelMetadata,
    load_model_candidates,
    select_unique_candidates_by_architecture,
)
from helpers.ensemble_optimizer.models import load_ensemble_models
from helpers.ensemble_optimizer.optimization import run_two_stream_optimization
from helpers.ensemble_optimizer.reporting import (
    RecipeMetadataConfig,
    build_recipe_metadata,
    write_recipe_metadata,
)
from helpers.ensemble_optimizer.splitting import HoldoutSplit, build_holdout_split
from helpers.provenance import build_split_fingerprint
from helpers.training.device import require_cuda_device
from helpers.training.runtime import seed_everything
from helpers.training.stain_normalization import resolve_dataloader_stain_normalizer_device
from helpers.training.utils import get_formatted_datetime_string

LOGGER = logging.getLogger(__name__)
_ENSEMBLE_SELECTION_MIN_MODELS = 2
_MODEL_SELECTION_STRATEGY = "strict_unique_metadata_per_requested_architecture"


@dataclass(frozen=True)
class EnsembleOptimizerOutputs:
    recipe_path: Path
    run_config_path: Path


def _prepare_output_dir(config: EnsembleOptimizerConfig) -> None:
    if config.output_dir.exists() and config.overwrite_output:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _serialize_config(config: EnsembleOptimizerConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "master_manifest_path",
        "metadata_dir",
        "output_dir",
        "local_data_dir",
        "pred_cache_dir",
        "log_folder",
    ):
        payload[key] = str(payload[key])
    payload["semantic_architectures"] = list(config.semantic_architectures)
    payload["spatial_architectures"] = list(config.spatial_architectures)
    return payload


def _select_models_from_metadata(
    config: EnsembleOptimizerConfig,
) -> list[SelectedModelMetadata]:
    requested_architectures = tuple(
        dict.fromkeys(config.semantic_architectures + config.spatial_architectures)
    )
    candidates = load_model_candidates(
        config.metadata_dir,
        config.sort_metric,
        requested_architectures=requested_architectures,
    )
    selected_models = select_unique_candidates_by_architecture(
        candidates,
        requested_architectures=requested_architectures,
        minimum_models=_ENSEMBLE_SELECTION_MIN_MODELS,
    )
    LOGGER.info(
        "Selected %s models using strict unique metadata contract.",
        len(selected_models),
    )
    return selected_models


def _build_validation_split(
    config: EnsembleOptimizerConfig, validation_layout: ValidationDatasetLayout
) -> HoldoutSplit:
    ordered_patients, positive_patients = summarize_validation_data(validation_layout)
    return build_holdout_split(
        ordered_patients,
        positive_patients,
        calibration_frac=config.val_calibration_frac,
        holdout_frac=config.val_holdout_frac,
        seed=config.seed + 123,
    )


def _execute_pipeline(config: EnsembleOptimizerConfig) -> EnsembleOptimizerOutputs:
    seed_everything(config.seed)
    selected_models = _select_models_from_metadata(config)
    device = require_cuda_device()
    LOGGER.info(
        "Stage 9 starting on %s. validation staging=%s output_dir=%s",
        device,
        "local" if config.stage_input_locally else "direct",
        config.output_dir,
    )
    LOGGER.info("Preparing validation data.")
    validation_layout = setup_validation_data(
        config.master_manifest_path,
        config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
        runtime_normalization_method=config.runtime_normalization_method,
        runtime_vahadane_backend=config.runtime_vahadane_backend,
        normalizer_device=resolve_dataloader_stain_normalizer_device(
            device,
            workers=config.workers,
        ),
    )
    LOGGER.info("Validation source ready: %s", config.master_manifest_path)
    split = _build_validation_split(config, validation_layout)
    LOGGER.info(
        "Split ready: optimization=%s calibration=%s holdout=%s patients.",
        len(split.optimization_patients),
        len(split.calibration_patients),
        len(split.holdout_patients),
    )
    split_fingerprint = build_split_fingerprint(
        optimization_patients=split.optimization_patients,
        calibration_patients=split.calibration_patients,
        holdout_patients=split.holdout_patients,
    )
    LOGGER.info("Loading %s selected models.", len(selected_models))
    models, _ = load_ensemble_models(selected_models, device)
    LOGGER.info(
        "Running two-stream optimization with %s semantic trials and %s spatial trials.",
        config.num_trials_semantic,
        config.num_trials_spatial,
    )
    dataloader = create_validation_dataloader(
        validation_layout,
        batch_size=config.batch_size,
        workers=config.workers,
    )
    optimization_result = run_two_stream_optimization(
        config,
        models,
        dataloader,
        device=device,
        predefined_split=split,
    )
    LOGGER.info(
        "Optimization summary: roi_threshold=%.4f decision_threshold=%.4f holdout_objective=%.4f.",
        optimization_result.roi_threshold,
        optimization_result.decision_threshold,
        float(
            optimization_result.holdout_metrics.get(
                "Objective_Composite",
                optimization_result.holdout_metrics.get("Macro_AUPRC_in_ROI", 0.0),
            )
        ),
    )
    timestamp = get_formatted_datetime_string()
    compatibility_signature = str(selected_models[0].raw_metadata["compatibility_signature"])
    validation_provenance = collect_validation_provenance(validation_layout)
    payload = build_recipe_metadata(
        config=RecipeMetadataConfig(
            selected_models=selected_models,
            semantic_indices=optimization_result.semantic_indices,
            spatial_indices=optimization_result.spatial_indices,
            semantic_weights=optimization_result.semantic_weights,
            spatial_weights=optimization_result.spatial_weights,
            roi_context_scale=config.roi_context_scale,
            roi_threshold=optimization_result.roi_threshold,
            decision_threshold=optimization_result.decision_threshold,
            spill_penalty_lambda=config.spill_penalty_lambda,
            spatial_patient_policy=config.spatial_patient_policy,
            calibration_metrics=optimization_result.calibration_metrics,
            holdout_metrics=optimization_result.holdout_metrics,
            generated_at=timestamp,
            compatibility_signature=compatibility_signature,
            validation_provenance=validation_provenance,
            split_fingerprint=split_fingerprint,
        )
    )
    recipe_path = write_recipe_metadata(payload, config.output_dir, timestamp)
    run_config_path = config.output_dir / "ensemble_optimizer_run_config.json"
    LOGGER.info("Writing recipe and run configuration.")
    run_payload = _serialize_config(config)
    run_payload.update(
        {
            "validation_master_manifest_path": str(config.master_manifest_path),
            "optimization_patients": sorted(split.optimization_patients),
            "calibration_patients": sorted(split.calibration_patients),
            "holdout_patients": sorted(split.holdout_patients),
            "recipe_path": str(recipe_path),
            "generated_at": timestamp,
            "compatibility_signature": compatibility_signature,
            "roi_threshold": optimization_result.roi_threshold,
            "decision_threshold": optimization_result.decision_threshold,
            "calibration_metrics": optimization_result.calibration_metrics,
            "validation_provenance": validation_provenance,
            "split_fingerprint": split_fingerprint,
            "model_selection_strategy": _MODEL_SELECTION_STRATEGY,
            "selected_semantic_architectures": [
                model.architecture
                for model in selected_models
                if model.architecture in config.semantic_architectures
            ],
            "selected_spatial_architectures": [
                model.architecture
                for model in selected_models
                if model.architecture in config.spatial_architectures
            ],
            "selected_models": [
                {
                    "architecture": model.architecture,
                    "encoder": model.encoder,
                    "checkpoint_path": model.checkpoint_path,
                    "metadata_sort_metric_value": model.sort_metric_value,
                    "metadata_filename": model.metadata_filename,
                    "optimization_subset_score": None,
                }
                for model in selected_models
            ],
        }
    )
    run_config_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    LOGGER.info("Stage 9 complete: recipe=%s run_config=%s", recipe_path, run_config_path)
    return EnsembleOptimizerOutputs(recipe_path=recipe_path, run_config_path=run_config_path)


def run_ensemble_optimizer_pipeline(
    config: EnsembleOptimizerConfig,
    *,
    pipeline_runner: Callable[
        [EnsembleOptimizerConfig], EnsembleOptimizerOutputs
    ] = _execute_pipeline,
) -> EnsembleOptimizerOutputs:
    _prepare_output_dir(config)
    outputs = pipeline_runner(config)
    if outputs.run_config_path.exists():
        return outputs
    run_payload = _serialize_config(config)
    run_payload["recipe_path"] = str(outputs.recipe_path)
    outputs.run_config_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    return outputs
