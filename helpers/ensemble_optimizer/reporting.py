from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from helpers.ensemble_optimizer.metadata import SelectedModelMetadata


def build_recipe_metadata(
    *,
    selected_models: list[SelectedModelMetadata],
    semantic_indices: list[int],
    spatial_indices: list[int],
    semantic_weights: list[float],
    spatial_weights: list[float],
    roi_context_scale: int,
    roi_threshold: float,
    spill_penalty_lambda: float,
    spatial_patient_policy: str,
    holdout_metrics: dict[str, float | int | str],
    generated_at: str,
) -> dict[str, Any]:
    final_semantic_weights = np.zeros(len(selected_models), dtype=np.float64)
    for index, global_index in enumerate(semantic_indices):
        final_semantic_weights[global_index] = semantic_weights[index]

    final_spatial_weights = np.zeros(len(selected_models), dtype=np.float64)
    for index, global_index in enumerate(spatial_indices):
        final_spatial_weights[global_index] = spatial_weights[index]

    model_registry: list[dict[str, Any]] = []
    for index, selected_model in enumerate(selected_models):
        raw_hyperparameters = selected_model.raw_metadata.get("hyperparameters", {})
        stream_role = "none"
        weight = 0.0
        if index in semantic_indices:
            stream_role = "semantic"
            weight = float(final_semantic_weights[index])
        elif index in spatial_indices:
            stream_role = "spatial"
            weight = float(final_spatial_weights[index])
        model_registry.append(
            {
                "model_id": f"model_{index}",
                "architecture": selected_model.architecture,
                "encoder": selected_model.encoder,
                "checkpoint_path": selected_model.checkpoint_path,
                "stream_role": stream_role,
                "weight": weight,
                "original_index_in_optimizer": index,
                "best_model_epoch": selected_model.raw_metadata.get("best_model_epoch"),
                "best_val_auprc_pixel_score": selected_model.raw_metadata.get(
                    "best_val_auprc_pixel_score"
                ),
                "training_monitoring_threshold": selected_model.raw_metadata.get(
                    "validation_monitoring_threshold_pixel_level"
                ),
                "Learning_rate": raw_hyperparameters.get("Learning_rate"),
                "Batch_Size": raw_hyperparameters.get("Batch_Size"),
                "Weight_Decay": raw_hyperparameters.get("Weight_Decay"),
                "Optimizer": raw_hyperparameters.get("Optimizer"),
                "Seed": raw_hyperparameters.get("Seed"),
                "Loss_Function": raw_hyperparameters.get("Loss_Function"),
            }
        )

    stream_order = {"semantic": 0, "spatial": 1, "none": 2}
    model_registry.sort(key=lambda item: (stream_order[item["stream_role"]], -item["weight"]))
    return {
        "experiment_id": f"two_stream_opt_{generated_at}",
        "datetime": generated_at,
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {
            "method": "lowpass_upsample_threshold",
            "scale": roi_context_scale,
            "threshold": roi_threshold,
        },
        "spatial_config": {
            "spill_lambda": spill_penalty_lambda,
            "patient_policy": spatial_patient_policy,
        },
        "model_registry": model_registry,
        "holdout_metrics": holdout_metrics,
    }


def write_recipe_metadata(payload: dict[str, Any], output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ENSEMBLE_TWO_STREAM_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return output_path
