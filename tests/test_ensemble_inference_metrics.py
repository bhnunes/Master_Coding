from __future__ import annotations

import math

import pytest
import torch

from helpers.ensemble_inference.metrics import (
    calculate_metrics,
    compute_auc_from_histograms,
    mask_to_binary_indices,
    summarize_patient_metrics,
)


def test_calculate_metrics_treats_empty_gt_and_empty_prediction_as_perfect() -> None:
    metrics = calculate_metrics(tp=0, fp=0, fn=0, tn=10)

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["tnr"] == 1.0


def test_calculate_metrics_treats_empty_gt_and_false_alarm_as_failure() -> None:
    metrics = calculate_metrics(tp=0, fp=3, fn=0, tn=7)

    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["fpr"] == 0.3


def test_mask_to_binary_indices_accepts_two_channel_masks() -> None:
    masks = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )

    result = mask_to_binary_indices(masks)

    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8))


def test_mask_to_binary_indices_accepts_three_dimensional_masks() -> None:
    masks = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)

    result = mask_to_binary_indices(masks)

    assert result.dtype == torch.uint8
    assert torch.equal(result, masks.to(torch.uint8))


def test_mask_to_binary_indices_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected masks shape"):
        mask_to_binary_indices(torch.zeros((1, 1, 2, 2), dtype=torch.float32))


def test_calculate_metrics_computes_standard_nonempty_case() -> None:
    metrics = calculate_metrics(tp=4, fp=1, fn=3, tn=2)

    assert metrics["dice"] == pytest.approx(2 * 4 / (2 * 4 + 1 + 3))
    assert metrics["iou"] == pytest.approx(4 / (4 + 1 + 3))
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["tpr"] == pytest.approx(4 / 7)
    assert metrics["tnr"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(4 / 5)
    assert metrics["fpr"] == pytest.approx(1 / 3)
    assert metrics["fnr"] == pytest.approx(3 / 7)


def test_calculate_metrics_returns_nan_when_rate_denominators_are_zero() -> None:
    metrics = calculate_metrics(tp=0, fp=0, fn=0, tn=0)

    assert math.isnan(metrics["accuracy"])
    assert math.isnan(metrics["tpr"])
    assert math.isnan(metrics["tnr"])
    assert math.isnan(metrics["precision"])
    assert math.isnan(metrics["fpr"])
    assert math.isnan(metrics["fnr"])


def test_compute_auc_from_histograms_returns_nan_without_both_classes() -> None:
    auc_value = compute_auc_from_histograms(
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([1, 0, 0], dtype=torch.int64),
    )

    assert math.isnan(auc_value)


def test_compute_auc_from_histograms_computes_expected_auc_for_separable_histograms() -> None:
    auc_value = compute_auc_from_histograms(
        torch.tensor([0, 0, 2], dtype=torch.int64),
        torch.tensor([2, 0, 0], dtype=torch.int64),
    )

    assert auc_value == pytest.approx(1.0)


def test_summarize_patient_metrics_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No patients were processed"):
        summarize_patient_metrics({}, seed=24)


def test_summarize_patient_metrics_without_bootstrap_reports_rule6_counts() -> None:
    stats_by_patient = {
        "pos": [{"tp": 2, "fp": 1, "fn": 1, "tn": 0}],
        "neg": [{"tp": 0, "fp": 0, "fn": 0, "tn": 4}],
    }

    summary = summarize_patient_metrics(stats_by_patient, seed=24)

    assert summary["bootstrap"] == {
        "ran": False,
        "n_patients": 2,
        "n_bootstrap_samples": 0,
        "seed": 24,
    }
    assert summary["confusion_matrix"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    assert summary["micro_averaged_metrics"]["point_estimate"]["dice"] == pytest.approx(2 / 3)
    assert summary["macro_dice_rule6_split"]["n_pos_patients"] == 1
    assert summary["macro_dice_rule6_split"]["n_neg_patients"] == 1
    assert summary["macro_dice_rule6_split"]["dice_pos_only"]["point_estimate"] == pytest.approx(
        2 / 3
    )
    assert summary["macro_dice_rule6_split"]["neg_clean_rate"]["point_estimate"] == 1.0
    assert math.isnan(summary["micro_averaged_metrics"]["ci"]["dice"][0])


def test_summarize_patient_metrics_with_bootstrap_populates_confidence_intervals() -> None:
    stats_by_patient = {
        f"patient_{index}": [
            {"tp": 1, "fp": 0, "fn": 0, "tn": 3}
            if index % 2 == 0
            else {"tp": 0, "fp": 0, "fn": 0, "tn": 4}
        ]
        for index in range(20)
    }

    summary = summarize_patient_metrics(stats_by_patient, seed=7, n_bootstrap_samples=8)

    assert summary["bootstrap"] == {
        "ran": True,
        "n_patients": 20,
        "n_bootstrap_samples": 8,
        "seed": 7,
    }
    assert len(summary["micro_averaged_metrics"]["ci"]["dice"]) == 2
    assert len(summary["macro_averaged_metrics"]["ci"]["dice"]) == 2
    assert not math.isnan(summary["macro_dice_rule6_split"]["dice_pos_only"]["ci"][0])
    assert not math.isnan(summary["macro_dice_rule6_split"]["neg_clean_rate"]["ci"][0])
