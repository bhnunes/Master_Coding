from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

VALID_STREAM_ROLES = {"semantic", "spatial", "none"}


@dataclass(frozen=True)
class RecipeModelSpec:
    architecture: str
    encoder: str
    checkpoint_path: Path
    stream_role: str
    weight: float
    raw_metadata: dict[str, Any]


@dataclass(frozen=True)
class EnsembleRecipe:
    strategy: str
    roi_threshold: float
    roi_scale: int
    model_registry: list[RecipeModelSpec]
    raw_payload: dict[str, Any]


def load_recipe_payload(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def parse_ensemble_recipe(payload: dict[str, Any]) -> EnsembleRecipe:
    strategy = str(payload.get("ensemble_strategy", "")).strip()
    if strategy != "two_stream_spatial_gating":
        raise ValueError(
            "Stage 12 only supports 'two_stream_spatial_gating' ensemble recipes. "
            f"Got '{strategy}'."
        )

    if "roi_config" not in payload:
        raise KeyError("Recipe missing required 'roi_config'.")
    if "model_registry" not in payload:
        raise KeyError("Recipe missing required 'model_registry'.")

    roi_config = payload["roi_config"]
    roi_threshold = float(roi_config.get("threshold", 0.5))
    roi_scale = max(1, int(roi_config.get("scale", 4)))

    model_registry: list[RecipeModelSpec] = []
    for entry in payload["model_registry"]:
        role = str(entry.get("stream_role", "none")).strip().lower()
        if role not in VALID_STREAM_ROLES:
            raise ValueError(f"Invalid recipe stream_role '{role}'.")
        architecture = str(entry.get("architecture", "")).strip().upper()
        encoder = str(entry.get("encoder", "")).strip()
        checkpoint_raw = str(entry.get("checkpoint_path", "")).strip()
        if not architecture or not encoder or not checkpoint_raw:
            raise ValueError(
                "Each recipe model must define architecture, encoder, and checkpoint_path."
            )
        model_registry.append(
            RecipeModelSpec(
                architecture=architecture,
                encoder=encoder,
                checkpoint_path=Path(checkpoint_raw),
                stream_role=role,
                weight=float(entry.get("weight", 0.0)),
                raw_metadata=dict(entry),
            )
        )

    if not model_registry:
        raise ValueError("Recipe model_registry cannot be empty.")

    return EnsembleRecipe(
        strategy=strategy,
        roi_threshold=roi_threshold,
        roi_scale=roi_scale,
        model_registry=model_registry,
        raw_payload=payload,
    )
