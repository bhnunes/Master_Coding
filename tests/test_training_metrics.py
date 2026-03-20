import torch

from helpers.training_metrics import (
    AdvancedMetricTracker,
    RunningWeightedMetric,
    TrainingHealthTracker,
)


def test_running_weighted_metric_computes_average() -> None:
    tracker = RunningWeightedMetric()

    tracker.update(6.0, 2)
    tracker.update(3.0, 1)

    assert tracker.get_average() == 3.0


def test_training_health_tracker_tracks_skips_and_nan_losses() -> None:
    tracker = TrainingHealthTracker(name="run")

    tracker.train_skip("naninf_loss")
    tracker.val_skip("bad_shape")
    tracker.train_naninf_loss()
    tracker.val_naninf_loss()

    assert tracker.epoch["train_skipped_batches"] == 1
    assert tracker.epoch["val_skipped_batches"] == 1
    assert tracker.run["train_naninf_loss"] == 1
    assert tracker.run["val_naninf_loss"] == 1
    assert tracker.run["train_skip_reasons"]["naninf_loss"] == 1
    assert tracker.run["val_skip_reasons"]["bad_shape"] == 1


def test_training_health_tracker_emergency_stop_uses_consecutive_collapses() -> None:
    tracker = TrainingHealthTracker(name="run", patience_collapse=2)

    tracker.mark_val_collapsed()
    assert tracker.should_emergency_stop() is False

    tracker.mark_val_invalid_metrics()
    assert tracker.should_emergency_stop() is True

    tracker.reset_collapse_counter()
    assert tracker.should_emergency_stop() is False


def test_advanced_metric_tracker_computes_metrics_from_probabilities() -> None:
    tracker = AdvancedMetricTracker(device=torch.device("cpu"), metric_bins=8)
    probs_fg = torch.tensor([[[0.9, 0.1], [0.8, 0.2]]], dtype=torch.float32)
    target = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.int64)

    tracker.update_from_probs_fg(probs_fg, target)

    result = tracker.compute_and_reset()

    assert result is not None
    assert result["val_auprc"] == 1.0
    assert result["val_auroc"] == 1.0
    assert result["val_mcc_star"] == 1.0
    assert result["fg_prevalence_at_05"] == 0.5


def test_advanced_metric_tracker_marks_health_on_collapse() -> None:
    tracker = AdvancedMetricTracker(
        device=torch.device("cpu"),
        metric_bins=8,
        collapse_low=0.3,
        collapse_high=0.7,
    )
    health = TrainingHealthTracker(name="run")
    probs_fg = torch.zeros((1, 2, 2), dtype=torch.float32)
    target = torch.zeros((1, 2, 2), dtype=torch.int64)

    tracker.update_from_probs_fg(probs_fg, target)

    result = tracker.compute_and_reset(health=health)

    assert result is None
    assert health.run["val_collapse_epochs"] == 1
    assert health.current_consecutive_collapses == 1
