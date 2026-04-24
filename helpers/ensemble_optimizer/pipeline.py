from __future__ import annotations

import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

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
    select_best_candidates_by_architecture,
)
from helpers.ensemble_optimizer.models import load_ensemble_models, load_single_model
from helpers.ensemble_optimizer.optimization import (
    predict_with_tta_batched,
    run_two_stream_optimization,
)
from helpers.ensemble_optimizer.reporting import (
    RecipeMetadataConfig,
    build_recipe_metadata,
    write_recipe_metadata,
)
from helpers.ensemble_optimizer.splitting import HoldoutSplit, build_holdout_split
from helpers.provenance import build_split_fingerprint
from helpers.training.gpu import GPUNormalizer
from helpers.training.metrics import AdvancedMetricTracker
from helpers.training.runtime import seed_everything
from helpers.training.utils import get_formatted_datetime_string

LOGGER = logging.getLogger(__name__)
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5
_ENSEMBLE_SELECTION_MIN_MODELS = 2
_PROBABILITY_THRESHOLD = 0.5


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under LoggerWriter."""

    return sys.__stderr__


def _progress_disabled() -> bool:
    isatty = getattr(_progress_file(), "isatty", None)
    return not bool(isatty() if callable(isatty) else False)


@dataclass(frozen=True)
class EnsembleOptimizerOutputs:
    recipe_path: Path
    run_config_path: Path


@dataclass(frozen=True)
class _CandidateSubsetEvaluation:
    score: float
    prediction_u16: npt.NDArray[np.uint16]
    truth_u8: npt.NDArray[np.uint8]
    patient_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SubsetPredictionCachePayload:
    prediction_u16: npt.NDArray[np.uint16]
    truth_u8: npt.NDArray[np.uint8]
    patient_ids: tuple[str, ...]


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


def _compute_candidate_subset_score(
    selected_model: SelectedModelMetadata,
    dataloader: Any,
    *,
    sort_metric: str,
    device: torch.device,
) -> _CandidateSubsetEvaluation:
    model = load_single_model(selected_model, device)
    tracker = AdvancedMetricTracker(device=device, from_logits=False)
    normalizer = GPUNormalizer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device,
    )
    tp = 0
    fp = 0
    fn = 0
    prediction_batches: list[npt.NDArray[np.uint16]] = []
    truth_batches: list[npt.NDArray[np.uint8]] = []
    patient_ids: list[str] = []

    with torch.inference_mode():
        for batch in dataloader:
            if batch is None:
                continue
            images, masks, _patient_ids = batch
            images = normalizer(images.to(device))
            mask_tensor = masks.to(device)
            probabilities = predict_with_tta_batched(model, images, selected_model.architecture)
            tracker.update_from_probs_fg(probabilities, mask_tensor)
            prediction_batches.append(
                np.clip(
                    np.nan_to_num(
                        probabilities.detach().float().cpu().numpy(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ),
                    0.0,
                    1.0,
                )
                .astype(np.float32)
                .__mul__(65535.0)
                .astype(np.uint16)
            )
            truth_batches.append(mask_tensor[:, 1, :, :].detach().cpu().numpy().astype(np.uint8))
            patient_ids.extend(str(patient_id) for patient_id in _patient_ids)
            pred_fg = probabilities >= _PROBABILITY_THRESHOLD
            true_fg = mask_tensor[:, 1, :, :] > _PROBABILITY_THRESHOLD
            tp += int((pred_fg & true_fg).sum().item())
            fp += int((pred_fg & ~true_fg).sum().item())
            fn += int((~pred_fg & true_fg).sum().item())

    metrics = tracker.compute_and_reset()
    model.cpu()

    if sort_metric == "best_val_auprc_pixel_score":
        if metrics is None:
            score = float("nan")
        else:
            score = float(metrics["val_auprc"])
    elif sort_metric == "best_validation_DICE":
        denominator = (2 * tp) + fp + fn
        score = float((2 * tp) / denominator) if denominator > 0 else 0.0
    else:
        raise ValueError(f"Unsupported optimizer sort metric: {sort_metric}")
    return _CandidateSubsetEvaluation(
        score=score,
        prediction_u16=np.concatenate(prediction_batches, axis=0)
        if prediction_batches
        else np.zeros((0, 0, 0), dtype=np.uint16),
        truth_u8=np.concatenate(truth_batches, axis=0)
        if truth_batches
        else np.zeros((0, 0, 0), dtype=np.uint8),
        patient_ids=tuple(patient_ids),
    )


def _select_models_from_optimization_subset(
    config: EnsembleOptimizerConfig,
    validation_layout: ValidationDatasetLayout,
    optimization_patients: set[str],
    device: torch.device,
) -> tuple[
    list[SelectedModelMetadata],
    dict[str, float],
    dict[str, str],
    dict[str, _SubsetPredictionCachePayload],
]:
    candidates = load_model_candidates(config.metadata_dir, config.sort_metric)
    dataloader = create_validation_dataloader(
        validation_layout,
        batch_size=config.batch_size,
        workers=config.workers,
        allowed_patients=optimization_patients,
    )
    if len(cast(Any, dataloader.dataset)) == 0:
        raise ValueError("Optimization subset is empty; cannot rank candidate models.")

    LOGGER.info("Ranking %s candidate models on the optimization subset.", len(candidates))
    subset_scores: dict[str, float] = {}
    requested_architectures = tuple(
        dict.fromkeys(config.semantic_architectures + config.spatial_architectures)
    )
    requested_architecture_set = {architecture.upper() for architecture in requested_architectures}
    best_evaluations_by_architecture: dict[str, tuple[str, _CandidateSubsetEvaluation]] = {}
    with tqdm(
        candidates,
        total=len(candidates),
        desc="Rank models",
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        file=_progress_file(),
        disable=_progress_disabled(),
    ) as progress_bar:
        for candidate in progress_bar:
            progress_bar.set_postfix(
                {"arch": candidate.architecture, "encoder": candidate.encoder},
                refresh=False,
            )
            try:
                evaluation = _compute_candidate_subset_score(
                    candidate,
                    dataloader,
                    sort_metric=config.sort_metric,
                    device=device,
                )
                subset_scores[candidate.metadata_filename] = evaluation.score
                architecture = candidate.architecture.upper()
                if (
                    architecture in requested_architecture_set
                    and np.isfinite(evaluation.score)
                    and (
                        architecture not in best_evaluations_by_architecture
                        or evaluation.score
                        > best_evaluations_by_architecture[architecture][1].score
                    )
                ):
                    best_evaluations_by_architecture[architecture] = (
                        candidate.metadata_filename,
                        evaluation,
                    )
            except Exception as error:
                LOGGER.warning(
                    "Skipping candidate %s (%s/%s): %s",
                    candidate.metadata_filename,
                    candidate.architecture,
                    candidate.encoder,
                    error,
                )
                subset_scores[candidate.metadata_filename] = float("nan")

    filtered_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(subset_scores.get(candidate.metadata_filename, float("nan")))
    ]
    selected_models, skipped_architectures = select_best_candidates_by_architecture(
        filtered_candidates,
        requested_architectures=requested_architectures,
        score_getter=lambda item: subset_scores[item.metadata_filename],
    )
    if len(selected_models) < _ENSEMBLE_SELECTION_MIN_MODELS:
        raise ValueError(
            "Fewer than 2 requested architectures had valid scored candidates. "
            f"Requested={list(requested_architectures)} skipped={skipped_architectures}"
        )
    for architecture, reason in skipped_architectures.items():
        LOGGER.warning("Skipping requested architecture %s: %s", architecture, reason)
    LOGGER.info(
        "Model ranking complete: %s valid candidates, %s requested architectures, %s selected.",
        len(filtered_candidates),
        len(requested_architectures),
        len(selected_models),
    )
    subset_prediction_caches: dict[str, _SubsetPredictionCachePayload] = {}
    for selected_model in selected_models:
        best_evaluation = best_evaluations_by_architecture.get(selected_model.architecture.upper())
        if best_evaluation is None or best_evaluation[0] != selected_model.metadata_filename:
            continue
        evaluation = best_evaluation[1]
        subset_prediction_caches[selected_model.metadata_filename] = _SubsetPredictionCachePayload(
            prediction_u16=evaluation.prediction_u16,
            truth_u8=evaluation.truth_u8,
            patient_ids=evaluation.patient_ids,
        )
    return selected_models, subset_scores, skipped_architectures, subset_prediction_caches


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    LOGGER.info(
        "Stage 10 starting on %s. validation staging=%s output_dir=%s",
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
    (
        selected_models,
        subset_scores,
        skipped_architectures,
        subset_prediction_caches,
    ) = _select_models_from_optimization_subset(
        config,
        validation_layout,
        split.optimization_patients,
        device,
    )
    LOGGER.info("Loading %s selected models.", len(selected_models))
    models, _ = load_ensemble_models(selected_models, device)
    if len(models) == len(selected_models):
        for model, selected_model in zip(models, selected_models, strict=True):
            cache_payload = subset_prediction_caches.get(selected_model.metadata_filename)
            if cache_payload is not None:
                cast(Any, model)._optimization_subset_cache = cache_payload
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
            "skipped_requested_architectures": skipped_architectures,
            "selected_models": [
                {
                    "architecture": model.architecture,
                    "encoder": model.encoder,
                    "checkpoint_path": model.checkpoint_path,
                    "sort_metric_value": model.sort_metric_value,
                    "metadata_filename": model.metadata_filename,
                    "optimization_subset_score": subset_scores[model.metadata_filename],
                }
                for model in selected_models
            ],
        }
    )
    run_config_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    LOGGER.info("Stage 10 complete: recipe=%s run_config=%s", recipe_path, run_config_path)
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
