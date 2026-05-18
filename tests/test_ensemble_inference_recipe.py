from __future__ import annotations

import pytest

from helpers.ensemble_inference.recipe import load_recipe_payload, parse_ensemble_recipe

ROI_THRESHOLD = 0.33
DECISION_THRESHOLD = 0.57
ROI_SCALE = 4
SPATIAL_MODEL_WEIGHT = 0.8
MIN_COMPONENT_AREA_PX = 16
MIN_PATIENT_POSITIVE_PATCHES = 3


def test_parse_ensemble_recipe_preserves_contract_fields() -> None:
    recipe = {
        "recipe_schema_version": 2,
        "calibration_objective": "balanced_rule6",
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": ROI_THRESHOLD, "scale": ROI_SCALE},
        "decision_config": {"threshold": DECISION_THRESHOLD},
        "postprocessing_config": {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": MIN_COMPONENT_AREA_PX,
            "min_patient_positive_patches": MIN_PATIENT_POSITIVE_PATCHES,
        },
        "validation_calibration_summary": {"objective": "balanced_rule6"},
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
                "weight": SPATIAL_MODEL_WEIGHT,
            },
        ],
    }

    parsed = parse_ensemble_recipe(load_recipe_payload(recipe))

    assert parsed.strategy == "two_stream_spatial_gating"
    assert parsed.roi_threshold == ROI_THRESHOLD
    assert parsed.decision_threshold == DECISION_THRESHOLD
    assert parsed.postprocessing_config.min_component_area_px == MIN_COMPONENT_AREA_PX
    assert parsed.postprocessing_config.min_patient_positive_patches == MIN_PATIENT_POSITIVE_PATCHES
    assert parsed.roi_scale == ROI_SCALE
    assert parsed.model_registry[0].stream_role == "semantic"
    assert parsed.model_registry[1].weight == SPATIAL_MODEL_WEIGHT


def test_parse_ensemble_recipe_rejects_invalid_strategy() -> None:
    recipe = {
        "recipe_schema_version": 2,
        "calibration_objective": "balanced_rule6",
        "ensemble_strategy": "weighted_average",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "decision_config": {"threshold": 0.6},
        "postprocessing_config": {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": MIN_COMPONENT_AREA_PX,
            "min_patient_positive_patches": MIN_PATIENT_POSITIVE_PATCHES,
        },
        "validation_calibration_summary": {"objective": "balanced_rule6"},
        "model_registry": [],
    }

    with pytest.raises(ValueError, match="two_stream_spatial_gating"):
        parse_ensemble_recipe(load_recipe_payload(recipe))


def test_parse_ensemble_recipe_rejects_missing_decision_config() -> None:
    recipe = {
        "recipe_schema_version": 2,
        "calibration_objective": "balanced_rule6",
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "postprocessing_config": {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": MIN_COMPONENT_AREA_PX,
            "min_patient_positive_patches": MIN_PATIENT_POSITIVE_PATCHES,
        },
        "validation_calibration_summary": {"objective": "balanced_rule6"},
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


def test_parse_ensemble_recipe_rejects_missing_schema_v2_fields() -> None:
    recipe = {
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "decision_config": {"threshold": 0.6},
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

    with pytest.raises(KeyError, match="recipe_schema_version"):
        parse_ensemble_recipe(load_recipe_payload(recipe))


def test_parse_ensemble_recipe_rejects_missing_postprocessing_config() -> None:
    recipe = {
        "recipe_schema_version": 2,
        "calibration_objective": "balanced_rule6",
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.5, "scale": 4},
        "decision_config": {"threshold": 0.6},
        "validation_calibration_summary": {"objective": "balanced_rule6"},
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

    with pytest.raises(KeyError, match="postprocessing_config"):
        parse_ensemble_recipe(load_recipe_payload(recipe))
