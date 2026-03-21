from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.data import (
    create_validation_dataloader,
    setup_validation_hdf5,
    summarize_validation_hdf5,
)
from helpers.ensemble_optimizer.metadata import (
    SelectedModelMetadata,
    load_model_candidates,
    select_top_models,
)
from helpers.ensemble_optimizer.models import load_ensemble_models, load_single_model
from helpers.ensemble_optimizer.optimization import (
    predict_with_tta_batched,
    run_two_stream_optimization,
)
from helpers.ensemble_optimizer.reporting import build_recipe_metadata, write_recipe_metadata
from helpers.ensemble_optimizer.splitting import HoldoutSplit, build_holdout_split
from helpers.provenance import build_split_fingerprint, collect_hdf5_provenance
from helpers.training.gpu import GPUNormalizer
from helpers.training.metrics import AdvancedMetricTracker
from helpers.training.runtime import seed_everything
from helpers.training.utils import get_formatted_datetime_string


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
        "hdf5_drive_dir",
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
) -> float:
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

    with torch.inference_mode():
        for batch in dataloader:
            if batch is None:
                continue
            images, masks, _patient_ids = batch
            images = normalizer(images.to(device))
            mask_tensor = masks.to(device)
            probabilities = predict_with_tta_batched(model, images, selected_model.architecture)
            tracker.update_from_probs_fg(probabilities, mask_tensor)
            pred_fg = probabilities >= 0.5
            true_fg = mask_tensor[:, 1, :, :] > 0.5
            tp += int((pred_fg & true_fg).sum().item())
            fp += int((pred_fg & ~true_fg).sum().item())
            fn += int((~pred_fg & true_fg).sum().item())

    metrics = tracker.compute_and_reset()
    model.cpu()

    if sort_metric == "best_val_auprc_pixel_score":
        if metrics is None:
            return float("nan")
        return float(metrics["val_auprc"])
    if sort_metric == "best_validation_DICE":
        denominator = (2 * tp) + fp + fn
        return float((2 * tp) / denominator) if denominator > 0 else 0.0
    raise ValueError(f"Unsupported optimizer sort metric: {sort_metric}")


def _select_models_from_optimization_subset(
    config: EnsembleOptimizerConfig,
    validation_h5_path: Path,
    optimization_patients: set[str],
    device: torch.device,
) -> tuple[list[SelectedModelMetadata], dict[str, float]]:
    candidates = load_model_candidates(config.metadata_dir, config.sort_metric)
    dataloader = create_validation_dataloader(
        validation_h5_path,
        batch_size=config.batch_size,
        workers=config.workers,
        allowed_patients=optimization_patients,
    )
    if len(cast(Any, dataloader.dataset)) == 0:
        raise ValueError("Optimization subset is empty; cannot rank candidate models.")

    subset_scores: dict[str, float] = {}
    for candidate in candidates:
        subset_scores[candidate.metadata_filename] = _compute_candidate_subset_score(
            candidate,
            dataloader,
            sort_metric=config.sort_metric,
            device=device,
        )

    filtered_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(subset_scores.get(candidate.metadata_filename, float("nan")))
    ]
    if len(filtered_candidates) < 2:
        raise ValueError("Fewer than 2 valid models remained after optimization-subset scoring.")

    selected_models = select_top_models(
        filtered_candidates,
        n_top_models=config.top_models,
        score_getter=lambda item: subset_scores[item.metadata_filename],
    )
    return selected_models, subset_scores


def _build_validation_split(
    config: EnsembleOptimizerConfig, validation_h5_path: Path
) -> HoldoutSplit:
    ordered_patients, positive_patients = summarize_validation_hdf5(validation_h5_path)
    return build_holdout_split(
        ordered_patients,
        positive_patients,
        holdout_frac=config.val_holdout_frac,
        seed=config.seed + 123,
    )


def _execute_pipeline(config: EnsembleOptimizerConfig) -> EnsembleOptimizerOutputs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    validation_h5_path = setup_validation_hdf5(
        config.hdf5_drive_dir,
        config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    split = _build_validation_split(config, validation_h5_path)
    split_fingerprint = build_split_fingerprint(
        optimization_patients=split.optimization_patients,
        holdout_patients=split.holdout_patients,
    )
    selected_models, subset_scores = _select_models_from_optimization_subset(
        config,
        validation_h5_path,
        split.optimization_patients,
        device,
    )
    models, _ = load_ensemble_models(selected_models, device)
    dataloader = create_validation_dataloader(
        validation_h5_path,
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
    timestamp = get_formatted_datetime_string()
    compatibility_signature = str(selected_models[0].raw_metadata["compatibility_signature"])
    validation_provenance = collect_hdf5_provenance(validation_h5_path)
    payload = build_recipe_metadata(
        selected_models=selected_models,
        semantic_indices=optimization_result.semantic_indices,
        spatial_indices=optimization_result.spatial_indices,
        semantic_weights=optimization_result.semantic_weights,
        spatial_weights=optimization_result.spatial_weights,
        roi_context_scale=config.roi_context_scale,
        roi_threshold=optimization_result.roi_threshold,
        spill_penalty_lambda=config.spill_penalty_lambda,
        spatial_patient_policy=config.spatial_patient_policy,
        holdout_metrics=optimization_result.holdout_metrics,
        generated_at=timestamp,
        compatibility_signature=compatibility_signature,
        validation_provenance=validation_provenance,
        split_fingerprint=split_fingerprint,
    )
    recipe_path = write_recipe_metadata(payload, config.output_dir, timestamp)
    run_config_path = config.output_dir / "ensemble_optimizer_run_config.json"
    run_payload = _serialize_config(config)
    run_payload.update(
        {
            "validation_h5_path": str(validation_h5_path),
            "optimization_patients": sorted(split.optimization_patients),
            "holdout_patients": sorted(split.holdout_patients),
            "recipe_path": str(recipe_path),
            "generated_at": timestamp,
            "compatibility_signature": compatibility_signature,
            "validation_provenance": validation_provenance,
            "split_fingerprint": split_fingerprint,
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
