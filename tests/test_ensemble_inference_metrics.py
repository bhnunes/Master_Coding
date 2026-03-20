from __future__ import annotations

from helpers.ensemble_inference.metrics import calculate_metrics


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
