from __future__ import annotations

import pytest

from helpers.ensemble_inference.recipe import load_recipe_payload, parse_ensemble_recipe


def test_parse_ensemble_recipe_preserves_contract_fields() -> None:
    recipe = {
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.33, "scale": 4},
        "decision_config": {"threshold": 0.57},
        "model_registry": [
            {
                "architecture": "SWIN",
                "encoder": "enc-a",
                "checkpoint_path": "/tmp/a.pth",
                "stream_role": "semantic",
                "weight": 1.0,
            },
            {
                "architecture": "FPN",
                "encoder": "enc-b",
                "checkpoint_path": "/tmp/b.pth",
                "stream_role": "spatial",
                "weight": 0.8,
            },
        ],
    }

    parsed = parse_ensemble_recipe(load_recipe_payload(recipe))

    assert parsed.strategy == "two_stream_spatial_gating"
    assert parsed.roi_threshold == 0.33
    assert parsed.decision_threshold == 0.57
    assert parsed.roi_scale == 4
    assert parsed.model_registry[0].stream_role == "semantic"
    assert parsed.model_registry[1].weight == 0.8


def test_parse_ensemble_recipe_rejects_invalid_strategy() -> None:
    recipe = {
        "ensemble_strategy": "weighted_average",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "decision_config": {"threshold": 0.6},
        "model_registry": [],
    }

    with pytest.raises(ValueError, match="two_stream_spatial_gating"):
        parse_ensemble_recipe(load_recipe_payload(recipe))


def test_parse_ensemble_recipe_rejects_missing_decision_config() -> None:
    recipe = {
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "model_registry": [
            {
                "architecture": "SWIN",
                "encoder": "enc-a",
                "checkpoint_path": "/tmp/a.pth",
                "stream_role": "semantic",
                "weight": 1.0,
            }
        ],
    }

    with pytest.raises(KeyError, match="decision_config"):
        parse_ensemble_recipe(load_recipe_payload(recipe))
