from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from helpers.ensemble_postprocessing import PostprocessingConfig, postprocessing_config_from_payload

VALID_STREAM_ROLES = {"semantic", "spatial", "none"}
REQUIRED_RECIPE_SCHEMA_VERSION = 2
REQUIRED_CALIBRATION_OBJECTIVE = "balanced_rule6"


def _require_payload_key(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"Recipe missing required '{key}'.")
    return payload[key]


def _parse_threshold_config(payload: dict[str, Any], key: str) -> tuple[float, dict[str, Any]]:
    config = cast(dict[str, Any], _require_payload_key(payload, key))
    if "threshold" not in config:
        raise KeyError(f"Recipe {key} missing required 'threshold'.")
    return float(config["threshold"]), config


def _parse_recipe_model_spec(entry: dict[str, Any]) -> RecipeModelSpec:
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

    return RecipeModelSpec(
        architecture=architecture,
        encoder=encoder,
        checkpoint_path=Path(checkpoint_raw),
        stream_role=role,
        weight=float(entry.get("weight", 0.0)),
        raw_metadata=dict(entry),
    )


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
    decision_threshold: float
    postprocessing_config: PostprocessingConfig
    roi_scale: int
    model_registry: list[RecipeModelSpec]
    raw_payload: dict[str, Any]


def load_recipe_payload(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def parse_ensemble_recipe(payload: dict[str, Any]) -> EnsembleRecipe:
    schema_version = int(_require_payload_key(payload, "recipe_schema_version"))
    if schema_version != REQUIRED_RECIPE_SCHEMA_VERSION:
        raise ValueError(
            f"Recipe recipe_schema_version must be {REQUIRED_RECIPE_SCHEMA_VERSION}. "
            f"Got {schema_version}."
        )
    calibration_objective = str(_require_payload_key(payload, "calibration_objective")).strip()
    if calibration_objective != REQUIRED_CALIBRATION_OBJECTIVE:
        raise ValueError(
            "Recipe calibration_objective must be "
            f"'{REQUIRED_CALIBRATION_OBJECTIVE}'. Got '{calibration_objective}'."
        )
    strategy = str(payload.get("ensemble_strategy", "")).strip()
    if strategy != "two_stream_spatial_gating":
        raise ValueError(
            "Stage 10 only supports 'two_stream_spatial_gating' ensemble recipes. "
            f"Got '{strategy}'."
        )

    roi_threshold, roi_config = _parse_threshold_config(payload, "roi_config")
    roi_scale = max(1, int(roi_config.get("scale", 4)))
    decision_threshold, _ = _parse_threshold_config(payload, "decision_config")
    postprocessing_config = postprocessing_config_from_payload(
        cast(dict[str, Any], _require_payload_key(payload, "postprocessing_config"))
    )
    _require_payload_key(payload, "validation_calibration_summary")

    raw_registry = cast(list[dict[str, Any]], _require_payload_key(payload, "model_registry"))
    model_registry = [_parse_recipe_model_spec(entry) for entry in raw_registry]

    if not model_registry:
        raise ValueError("Recipe model_registry cannot be empty.")

    return EnsembleRecipe(
        strategy=strategy,
        roi_threshold=roi_threshold,
        decision_threshold=decision_threshold,
        postprocessing_config=postprocessing_config,
        roi_scale=roi_scale,
        model_registry=model_registry,
        raw_payload=payload,
    )
