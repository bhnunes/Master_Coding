from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingLossWeights:
    """Per-architecture BCE+Dice loss weights."""

    alpha_bce: float
    beta_dice_bg: float
    gamma_dice_fg: float


@dataclass(frozen=True)
class TrainingModelRegistryEntry:
    """Normalized registry entry for one approved architecture."""

    lr: float
    wd: float
    encoders: tuple[str, ...]
    loss: TrainingLossWeights


DEFAULT_MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "training_model_registry.json"
)


def get_model_registry_path() -> Path:
    """Return the configured path to the approved training model registry."""

    override_path = os.environ.get("TRAINING_MODEL_REGISTRY_PATH", "").strip()
    return Path(override_path) if override_path else DEFAULT_MODEL_REGISTRY_PATH


def load_training_model_registry() -> dict[str, TrainingModelRegistryEntry]:
    """Load and validate the approved research architecture registry."""

    registry_path = get_model_registry_path()
    try:
        raw_registry = json.loads(registry_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Training model registry not found: {registry_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Training model registry is invalid JSON: {registry_path}") from error

    if not isinstance(raw_registry, dict):
        raise ValueError("Training model registry must be a JSON object.")

    registry: dict[str, TrainingModelRegistryEntry] = {}
    for architecture, config in raw_registry.items():
        if not isinstance(config, dict) or not {"lr", "wd", "encoders", "loss"}.issubset(config):
            raise ValueError(
                f"Architecture '{architecture}' must define 'lr', 'wd', 'encoders', and 'loss'."
            )
        encoders = config["encoders"]
        if not isinstance(encoders, list) or not all(
            isinstance(encoder, str) for encoder in encoders
        ):
            raise ValueError(
                f"Architecture '{architecture}' must define 'encoders' as a list of strings."
            )
        loss = config["loss"]
        if not isinstance(loss, dict) or not {
            "alpha_bce",
            "beta_dice_bg",
            "gamma_dice_fg",
        }.issubset(loss):
            raise ValueError(
                f"Architecture '{architecture}' must define 'loss' with "
                "'alpha_bce', 'beta_dice_bg', and 'gamma_dice_fg'."
            )
        registry[str(architecture).upper()] = TrainingModelRegistryEntry(
            lr=float(config["lr"]),
            wd=float(config["wd"]),
            encoders=tuple(encoders),
            loss=TrainingLossWeights(
                alpha_bce=float(loss["alpha_bce"]),
                beta_dice_bg=float(loss["beta_dice_bg"]),
                gamma_dice_fg=float(loss["gamma_dice_fg"]),
            ),
        )

    return registry


def get_supported_encoders(architecture: str) -> tuple[str, ...]:
    """Return the approved research encoders for an architecture."""

    registry = load_training_model_registry()
    try:
        config = registry[architecture.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown architecture: {architecture}") from error
    return config.encoders


def get_training_model_registry_entry(architecture: str) -> TrainingModelRegistryEntry:
    """Return the normalized registry entry for one approved architecture."""

    registry = load_training_model_registry()
    try:
        return registry[architecture.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown architecture: {architecture}") from error


def validate_architecture_encoder_pair(architecture: str, encoder: str) -> None:
    """Ensure the selected architecture/encoder pair is approved for research use."""

    supported_encoders = get_supported_encoders(architecture)
    if encoder not in supported_encoders:
        supported_display = ", ".join(supported_encoders)
        raise ValueError(
            f"TRAINING_ENCODER must be one of [{supported_display}] "
            f"for architecture {architecture.upper()}."
        )
