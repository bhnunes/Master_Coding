from __future__ import annotations

from typing import cast

import schedulefree
import segmentation_models_pytorch as smp
import torch
from torch import nn

from helpers.training_registry import load_training_model_registry


def get_learning_rate(architecture: str) -> tuple[float, float]:
    """Return the existing learning-rate and weight-decay defaults by architecture."""

    configs = load_training_model_registry()

    try:
        config = configs[architecture.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown architecture: {architecture}") from error

    return cast(float, config["lr"]), cast(float, config["wd"])


def create_model(architecture: str, encoder: str, validation: bool = False) -> nn.Module:
    """Create the configured segmentation model."""

    encoder_weights = None if validation else "imagenet"
    common_args = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": 2,
        "activation": None,
    }

    arch = architecture.upper()
    if arch == "SWIN":
        return cast(nn.Module, smp.Unet(**common_args, decoder_attention_type=None))
    if arch == "DEEPLABV3PLUS":
        return cast(nn.Module, smp.DeepLabV3Plus(**common_args, decoder_attention_type=None))
    if arch == "DPT":
        return cast(nn.Module, smp.DPT(**common_args, decoder_readout="ignore"))
    if arch == "UNET++":
        return cast(nn.Module, smp.UnetPlusPlus(**common_args, decoder_attention_type=None))
    if arch == "FPN":
        return cast(nn.Module, smp.FPN(**common_args, decoder_attention_type=None))
    if arch == "SEGFORMER":
        return cast(nn.Module, smp.Segformer(**common_args, decoder_attention_type=None))
    if arch == "MANET":
        return cast(nn.Module, smp.MAnet(**common_args, decoder_attention_type=None))
    if arch == "UPERNET":
        return cast(nn.Module, smp.UPerNet(**common_args, decoder_attention_type=None))
    if arch == "INCEPTIONRESNETV2":
        return cast(nn.Module, smp.Unet(**common_args, decoder_attention_type=None))

    raise ValueError(f"Unknown architecture: {architecture}")


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
