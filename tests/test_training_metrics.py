from __future__ import annotations

import os

import pytest
import torch

from helpers.runtime_platform import COLAB_INLINE_MATPLOTLIB_BACKEND, HEADLESS_MATPLOTLIB_BACKEND
from helpers.training.metrics import (
    AdvancedMetricTracker,
    RunningWeightedMetric,
    TrainingHealthTracker,
)

WEIGHTED_SCORE = 3.0
COLLAPSE_PATIENCE = 2
PROBABILITY_THRESHOLD = 0.5
LOGIT_PROBABILITY_CUTOFF = 0.5
HISTOGRAM_TEST_BINS = 8
HISTOGRAM_TEST_CLASS_PIXELS = 2


def test_advanced_metric_tracker_replaces_colab_inline_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", COLAB_INLINE_MATPLOTLIB_BACKEND)

    AdvancedMetricTracker(device=torch.device("cpu"), metric_bins=8)

    assert os.environ["MPLBACKEND"] == HEADLESS_MATPLOTLIB_BACKEND


def test_running_weighted_metric_computes_average() -> None:
    tracker = RunningWeightedMetric()

    tracker.update(6.0, 2)
    tracker.update(3.0, 1)

    assert tracker.get_average() == WEIGHTED_SCORE


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
    tracker = TrainingHealthTracker(name="run", patience_collapse=COLLAPSE_PATIENCE)

    tracker.mark_val_collapsed()
    assert tracker.should_emergency_stop() is False

    tracker.mark_val_invalid_metrics()
    assert tracker.should_emergency_stop() is True

    tracker.reset_collapse_counter()
    assert tracker.should_emergency_stop() is False


def test_training_health_tracker_formats_and_logs_reasons(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = TrainingHealthTracker(name="run")

    tracker.train_skip("b")
    tracker.train_skip("a")
    tracker.train_skip("a")
    tracker.val_skip("z")
    tracker.log_epoch(epoch_num=3, prefix="[T]")
    tracker.log_run(prefix="[R]")

    assert TrainingHealthTracker._fmt_reasons({}) == "-"
    assert TrainingHealthTracker._fmt_reasons({"b": 1, "a": 2}) == "a:2, b:1"
    output = capsys.readouterr().out
    assert "[T] Epoch 3" in output
    assert "reasons: a:2, b:1" in output
    assert "[R] Run totals" in output


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
    assert result["fg_prevalence_at_05"] == PROBABILITY_THRESHOLD


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


def test_advanced_metric_tracker_handles_high_prevalence_collapse() -> None:
    tracker = AdvancedMetricTracker(
        device=torch.device("cpu"),
        metric_bins=8,
        collapse_low=0.1,
        collapse_high=0.7,
    )
    health = TrainingHealthTracker(name="run")

    tracker.update_from_probs_fg(torch.ones((1, 2, 2), dtype=torch.float32), torch.ones((1, 2, 2)))

    assert tracker.compute_and_reset(health=health) is None
    assert health.run["val_collapse_epochs"] == 1


def test_advanced_metric_tracker_rejects_invalid_logits_shapes() -> None:
    with pytest.raises(ValueError, match="Expected pred_logits"):
        AdvancedMetricTracker._extract_probs_fg(torch.zeros((1, 2, 2)))

    with pytest.raises(ValueError, match="Expected C=1 or C=2"):
        AdvancedMetricTracker._extract_probs_fg(torch.zeros((1, 3, 2, 2)))


def test_advanced_metric_tracker_extracts_probs_for_binary_logits() -> None:
    single_channel = AdvancedMetricTracker._extract_probs_fg(torch.tensor([[[[0.0, 2.0]]]]))
    dual_channel = AdvancedMetricTracker._extract_probs_fg(
        torch.tensor([[[[0.0, 1.0]], [[2.0, 1.0]]]])
    )

    assert torch.allclose(single_channel, torch.tensor([[[0.5, 0.8808]]]), atol=1e-4)
    assert dual_channel.shape == (1, 1, 2)
    assert dual_channel[0, 0, 0] > LOGIT_PROBABILITY_CUTOFF


def test_advanced_metric_tracker_extracts_targets_and_probabilities() -> None:
    target = torch.tensor([[[[0, 1]], [[1, 0]]]], dtype=torch.float32)
    extracted = AdvancedMetricTracker._extract_target_fg(target)
    probs = AdvancedMetricTracker._extract_probs_from_probs_fg(
        torch.tensor([[[[1.2, -0.1], [0.4, 0.9]]]], dtype=torch.float32)
    )

    assert extracted.dtype is torch.bool
    assert extracted.tolist() == [[[True, False]]]
    assert probs.tolist() == [[[1.0, 0.0], [0.4000000059604645, 0.8999999761581421]]]

    with pytest.raises(ValueError, match="Unsupported target shape"):
        AdvancedMetricTracker._extract_target_fg(torch.zeros((1, 3, 2, 2)))
    with pytest.raises(ValueError, match="Expected probs_fg"):
        AdvancedMetricTracker._extract_probs_from_probs_fg(torch.zeros((1, 1, 1, 1, 1)))


def test_advanced_metric_tracker_returns_none_when_no_pixels_processed() -> None:
    tracker = AdvancedMetricTracker(device=torch.device("cpu"), metric_bins=8)

    assert tracker.compute_and_reset() is None


def test_advanced_metric_tracker_marks_invalid_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = AdvancedMetricTracker(device=torch.device("cpu"), metric_bins=8)
    health = TrainingHealthTracker(name="run")
    tracker.update_from_probs_fg(
        torch.tensor([[[0.9, 0.1], [0.8, 0.2]]], dtype=torch.float32),
        torch.tensor([[[1, 0], [1, 0]]], dtype=torch.int64),
    )
    monkeypatch.setattr(
        tracker,
        "_compute_threshold_metrics",
        lambda: (
            torch.tensor(float("nan")),
            torch.tensor(1.0),
            torch.tensor([1.0]),
        ),
    )

    assert tracker.compute_and_reset(health=health) is None
    assert health.run["val_invalid_metric_epochs"] == 1


def test_advanced_metric_tracker_uses_fixed_size_histograms() -> None:
    tracker = AdvancedMetricTracker(device=torch.device("cpu"), metric_bins=HISTOGRAM_TEST_BINS)

    tracker.update_from_probs_fg(
        torch.tensor([[[0.95, 0.75], [0.25, 0.05]]], dtype=torch.float32),
        torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int64),
    )

    assert int(tracker._positive_hist.sum().item()) == HISTOGRAM_TEST_CLASS_PIXELS
    assert int(tracker._negative_hist.sum().item()) == HISTOGRAM_TEST_CLASS_PIXELS
    assert tracker._positive_hist.numel() == HISTOGRAM_TEST_BINS
    assert tracker._negative_hist.numel() == HISTOGRAM_TEST_BINS
