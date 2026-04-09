from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import nn

from helpers.ensemble_optimizer.metadata import SelectedModelMetadata
from helpers.training.models import create_model

LOGGER = logging.getLogger(__name__)


def load_checkpoint_strict_without_aux(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("classification_head.")
    }
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(filtered_state_dict.keys())
    if model_keys and checkpoint_keys:
        model_has_prefix = model_keys[0].startswith("_orig_mod.")
        checkpoint_has_prefix = checkpoint_keys[0].startswith("_orig_mod.")
        if model_has_prefix and not checkpoint_has_prefix:
            filtered_state_dict = {
                f"_orig_mod.{key}": value for key, value in filtered_state_dict.items()
            }
        elif not model_has_prefix and checkpoint_has_prefix:
            filtered_state_dict = {
                key.replace("_orig_mod.", "", 1): value
                for key, value in filtered_state_dict.items()
            }
    model.load_state_dict(filtered_state_dict, strict=True)
    model.to(device)
    if checkpoint.get("is_compiled", False):
        try:
            model = cast(nn.Module, torch.compile(model))
        except Exception as error:
            LOGGER.warning("torch.compile failed for %s: %s", checkpoint_path, error)
    return model


def build_loaded_model_info(selected_model: SelectedModelMetadata) -> dict[str, Any]:
    hyperparameters = selected_model.raw_metadata.get("hyperparameters", {})
    return {
        "architecture": selected_model.architecture,
        "encoder": selected_model.encoder,
        "checkpoint_path": selected_model.checkpoint_path,
        "best_model_epoch": selected_model.raw_metadata.get("best_model_epoch"),
        "best_val_auprc_pixel_score": selected_model.raw_metadata.get("best_val_auprc_pixel_score"),
        "training_monitoring_threshold": selected_model.raw_metadata.get(
            "validation_monitoring_threshold_pixel_level"
        ),
        "Learning_rate": hyperparameters.get("Learning_rate"),
        "Batch_Size": hyperparameters.get("Batch_Size"),
        "Weight_Decay": hyperparameters.get("Weight_Decay"),
        "Optimizer": hyperparameters.get("Optimizer"),
        "Seed": hyperparameters.get("Seed"),
        "Loss_Function": hyperparameters.get("Loss_Function"),
    }


def load_single_model(selected_model: SelectedModelMetadata, device: torch.device) -> nn.Module:
    model = create_model(
        architecture=selected_model.architecture,
        encoder=selected_model.encoder,
        validation=True,
    )
    model = load_checkpoint_strict_without_aux(model, selected_model.checkpoint_path, device)
    cast(Any, model).arch_name = selected_model.architecture
    model.eval()
    return model


def load_ensemble_models(
    selected_models: list[SelectedModelMetadata],
    device: torch.device,
) -> tuple[list[nn.Module], list[dict[str, Any]]]:
    ensemble_models: list[nn.Module] = []
    constituent_model_info: list[dict[str, Any]] = []
    failed_loads = 0
    for selected_model in selected_models:
        try:
            model = load_single_model(selected_model, device)
            ensemble_models.append(model)
            constituent_model_info.append(build_loaded_model_info(selected_model))
        except Exception as error:
            failed_loads += 1
            LOGGER.warning("Failed to load model %s: %s", selected_model.checkpoint_path, error)
    if len(ensemble_models) < 2:
        raise RuntimeError("Fewer than 2 models loaded. Ensemble optimization requires at least 2.")
    LOGGER.info(
        "Loaded %s/%s selected models%s.",
        len(ensemble_models),
        len(selected_models),
        f" ({failed_loads} failed)" if failed_loads else "",
    )
    return ensemble_models, constituent_model_info
