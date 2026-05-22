from __future__ import annotations

import numpy as np
import pytest

from helpers.ensemble_postprocessing import (
    apply_patient_positive_patch_suppression,
    postprocessing_config_from_payload,
    threshold_and_filter_components,
)

PATIENT_AREA_FRACTION_CANDIDATE = 1e-4
COMPONENT_AREA_FRACTION_CANDIDATE = 0.001


def test_threshold_and_filter_components_removes_small_components() -> None:
    probabilities = np.array(
        [
            [0.9, 0.9, 0.0, 0.8],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.7],
            [0.0, 0.0, 0.7, 0.7],
        ],
        dtype=np.float32,
    )

    filtered = threshold_and_filter_components(
        probabilities,
        decision_threshold=0.5,
        min_component_area_px=4,
    )

    assert filtered.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]


def test_patient_positive_patch_suppression_zeroes_low_evidence_patients() -> None:
    stats = {
        "p1": [{"tp": 2, "fp": 1, "fn": 3, "tn": 4}],
        "p2": [{"tp": 0, "fp": 2, "fn": 0, "tn": 7}],
    }
    truth_counts = {
        "p1": {"positive": 5, "negative": 5},
        "p2": {"positive": 0, "negative": 9},
    }

    suppressed = apply_patient_positive_patch_suppression(
        stats,
        truth_counts,
        {"p1": 1, "p2": 0},
        min_patient_positive_patches=2,
    )

    assert suppressed["p1"] == [{"tp": 0, "fp": 0, "fn": 5, "tn": 5}]
    assert suppressed["p2"] == [{"tp": 0, "fp": 0, "fn": 0, "tn": 9}]


def test_patient_area_suppression_removes_tiny_patient_burden() -> None:
    stats = {
        "p1": [{"tp": 2, "fp": 0, "fn": 3, "tn": 95}],
        "p2": [{"tp": 0, "fp": 1, "fn": 0, "tn": 999}],
    }
    truth_counts = {
        "p1": {"positive": 5, "negative": 95},
        "p2": {"positive": 0, "negative": 1000},
    }

    suppressed = apply_patient_positive_patch_suppression(
        stats,
        truth_counts,
        {"p1": 3, "p2": 3},
        min_patient_positive_patches=2,
        positive_area_fractions={"p1": 0.02, "p2": 0.001},
        min_patient_positive_area_fraction=0.01,
    )

    assert suppressed["p1"] == stats["p1"]
    assert suppressed["p2"] == [{"tp": 0, "fp": 0, "fn": 0, "tn": 1000}]


def test_component_fraction_filtering_converts_to_patch_area() -> None:
    probabilities = np.array([[0.9, 0.9], [0.0, 0.0]], dtype=np.float32)

    filtered = threshold_and_filter_components(
        probabilities,
        decision_threshold=0.5,
        min_component_area_px=0,
        min_component_area_fraction_patch=0.75,
    )

    assert filtered.tolist() == [[0, 0], [0, 0]]


def test_postprocessing_config_requires_explicit_v3_near_fields() -> None:
    with pytest.raises(KeyError, match="min_patient_positive_area_fraction"):
        postprocessing_config_from_payload(
            {
                "method": "threshold_components_patient_suppression",
                "min_component_area_px": 16,
                "min_patient_positive_patches": 3,
            }
        )

    config = postprocessing_config_from_payload(
        {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": 64,
            "min_patient_positive_patches": 5,
            "min_patient_positive_area_fraction": PATIENT_AREA_FRACTION_CANDIDATE,
            "min_component_area_fraction_patch": COMPONENT_AREA_FRACTION_CANDIDATE,
        }
    )

    assert config.min_patient_positive_area_fraction == PATIENT_AREA_FRACTION_CANDIDATE
    assert config.min_component_area_fraction_patch == COMPONENT_AREA_FRACTION_CANDIDATE


def test_permissive_postprocessing_keeps_thresholded_mask_unchanged() -> None:
    probabilities = np.array([[0.2, 0.7], [0.8, 0.1]], dtype=np.float32)

    filtered = threshold_and_filter_components(
        probabilities,
        decision_threshold=0.5,
        min_component_area_px=0,
    )

    assert filtered.tolist() == [[0, 1], [1, 0]]
