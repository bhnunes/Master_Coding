from __future__ import annotations

import numpy as np

from helpers.ensemble_postprocessing import (
    apply_patient_positive_patch_suppression,
    threshold_and_filter_components,
)


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


def test_permissive_postprocessing_keeps_thresholded_mask_unchanged() -> None:
    probabilities = np.array([[0.2, 0.7], [0.8, 0.1]], dtype=np.float32)

    filtered = threshold_and_filter_components(
        probabilities,
        decision_threshold=0.5,
        min_component_area_px=0,
    )

    assert filtered.tolist() == [[0, 1], [1, 0]]
