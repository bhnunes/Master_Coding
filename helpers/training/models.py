from __future__ import annotations

from typing import cast

import schedulefree
import segmentation_models_pytorch as smp
import torch
from torch import nn

from helpers.training.registry import TrainingLossWeights, get_training_model_registry_entry

COMMON_MODEL_ARGS = {
    "in_channels": 3,
    "classes": 2,
    "activation": None,
}
MODEL_BUILDERS: dict[str, tuple[str, dict[str, object]]] = {
    "SWIN": ("Unet", {"decoder_attention_type": None}),
    "DEEPLABV3PLUS": ("DeepLabV3Plus", {"decoder_attention_type": None}),
    "DPT": ("DPT", {"decoder_readout": "ignore"}),
    "UNET++": ("UnetPlusPlus", {"decoder_attention_type": None}),
    "FPN": ("FPN", {"decoder_attention_type": None}),
    "SEGFORMER": ("Segformer", {"decoder_attention_type": None}),
    "MANET": ("MAnet", {"decoder_attention_type": None}),
    "UPERNET": ("UPerNet", {"decoder_attention_type": None}),
    "INCEPTIONRESNETV2": ("Unet", {"decoder_attention_type": None}),
}


def _resolve_encoder_weights(encoder: str, *, validation: bool) -> str | bool | None:
    if validation:
        return None
    if encoder.startswith("tu-"):
        return True
    return "imagenet"


def get_learning_rate(architecture: str) -> tuple[float, float]:
    """Return the existing learning-rate and weight-decay defaults by architecture."""

    config = get_training_model_registry_entry(architecture)
    return config.lr, config.wd


def get_loss_weights(architecture: str) -> TrainingLossWeights:
    """Return the registry-defined BCE+Dice loss weights for an architecture."""

    return get_training_model_registry_entry(architecture).loss


def create_model(architecture: str, encoder: str, validation: bool = False) -> nn.Module:
    """Create the configured segmentation model."""

    builder_config = MODEL_BUILDERS.get(architecture.upper())
    if builder_config is None:
        raise ValueError(f"Unknown architecture: {architecture}")

    builder_name, extra_args = builder_config
    common_args = {
        "encoder_name": encoder,
        "encoder_weights": _resolve_encoder_weights(encoder, validation=validation),
        **COMMON_MODEL_ARGS,
        **extra_args,
    }
    builder = getattr(smp, builder_name)
    return cast(nn.Module, builder(**common_args))


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer | schedulefree.AdamWScheduleFree:
    """Create the configured optimizer for a model."""

    if optimizer_name == "AdamWScheduleFree":
        return schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")
