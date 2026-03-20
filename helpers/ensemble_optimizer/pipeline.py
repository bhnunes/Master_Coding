from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.data import create_validation_dataloader, setup_validation_hdf5
from helpers.ensemble_optimizer.metadata import load_and_select_models
from helpers.ensemble_optimizer.models import load_ensemble_models
from helpers.ensemble_optimizer.optimization import run_two_stream_optimization
from helpers.ensemble_optimizer.reporting import build_recipe_metadata, write_recipe_metadata
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
    for key in ("hdf5_drive_dir", "metadata_dir", "output_dir", "local_data_dir", "pred_cache_dir"):
        payload[key] = str(payload[key])
    payload["semantic_architectures"] = list(config.semantic_architectures)
    payload["spatial_architectures"] = list(config.spatial_architectures)
    return payload


def _execute_pipeline(config: EnsembleOptimizerConfig) -> EnsembleOptimizerOutputs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    validation_h5_path = setup_validation_hdf5(
        config.hdf5_drive_dir,
        config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    selected_models = load_and_select_models(
        config.metadata_dir,
        n_top_models=config.top_models,
        sort_metric=config.sort_metric,
    )
    models, _ = load_ensemble_models(selected_models, device)
    dataloader = create_validation_dataloader(
        validation_h5_path,
        batch_size=config.batch_size,
        workers=config.workers,
    )
    optimization_result = run_two_stream_optimization(config, models, dataloader, device=device)
    timestamp = get_formatted_datetime_string()
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
    )
    recipe_path = write_recipe_metadata(payload, config.output_dir, timestamp)
    run_config_path = config.output_dir / "ensemble_optimizer_run_config.json"
    run_payload = _serialize_config(config)
    run_payload.update(
        {
            "validation_h5_path": str(validation_h5_path),
            "recipe_path": str(recipe_path),
            "generated_at": timestamp,
            "selected_models": [
                {
                    "architecture": model.architecture,
                    "encoder": model.encoder,
                    "checkpoint_path": model.checkpoint_path,
                    "sort_metric_value": model.sort_metric_value,
                    "metadata_filename": model.metadata_filename,
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
