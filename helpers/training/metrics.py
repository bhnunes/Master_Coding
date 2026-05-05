from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

import torch

from helpers.runtime_platform import ensure_headless_matplotlib_backend

LOGIT_TENSOR_NDIM = 4
MASK_TENSOR_NDIM = 3
BINARY_CLASS_COUNT = 2
SINGLE_CHANNEL_COUNT = 1
FOREGROUND_CHANNEL_INDEX = 1
SINGLE_CHANNEL_INDEX = 0
DEFAULT_PROBABILITY_THRESHOLD = 0.5
MIN_METRIC_BINS = 2


class TrainingHealthTracker:
    """Track train/validation instability counters and collapse state."""

    def __init__(self, name: str = "run", patience_collapse: int = 3) -> None:
        self.name = name
        self.patience_collapse = patience_collapse
        self.current_consecutive_collapses = 0
        self.reset_run()
        self.reset_epoch()

    def reset_run(self) -> None:
        self.run: dict[str, Any] = {
            "train_naninf_loss": 0,
            "val_naninf_loss": 0,
            "train_skipped_batches": 0,
            "val_skipped_batches": 0,
            "train_skip_reasons": defaultdict(int),
            "val_skip_reasons": defaultdict(int),
            "val_collapse_epochs": 0,
            "val_invalid_metric_epochs": 0,
        }

    def reset_epoch(self) -> None:
        self.epoch: dict[str, Any] = {
            "train_naninf_loss": 0,
            "val_naninf_loss": 0,
            "train_skipped_batches": 0,
            "val_skipped_batches": 0,
            "train_skip_reasons": defaultdict(int),
            "val_skip_reasons": defaultdict(int),
            "val_collapsed": 0,
            "val_invalid_metrics": 0,
        }

    def train_skip(self, reason: str) -> None:
        self.epoch["train_skipped_batches"] += 1
        self.epoch["train_skip_reasons"][reason] += 1
        self.run["train_skipped_batches"] += 1
        self.run["train_skip_reasons"][reason] += 1

    def val_skip(self, reason: str) -> None:
        self.epoch["val_skipped_batches"] += 1
        self.epoch["val_skip_reasons"][reason] += 1
        self.run["val_skipped_batches"] += 1
        self.run["val_skip_reasons"][reason] += 1

    def train_naninf_loss(self) -> None:
        self.epoch["train_naninf_loss"] += 1
        self.run["train_naninf_loss"] += 1

    def val_naninf_loss(self) -> None:
        self.epoch["val_naninf_loss"] += 1
        self.run["val_naninf_loss"] += 1

    def mark_val_collapsed(self) -> None:
        self.epoch["val_collapsed"] = 1
        self.run["val_collapse_epochs"] += 1
        self.current_consecutive_collapses += 1

    def mark_val_invalid_metrics(self) -> None:
        self.epoch["val_invalid_metrics"] = 1
        self.run["val_invalid_metric_epochs"] += 1
        self.current_consecutive_collapses += 1

    def reset_collapse_counter(self) -> None:
        self.current_consecutive_collapses = 0

    def should_emergency_stop(self) -> bool:
        return self.current_consecutive_collapses >= self.patience_collapse

    @staticmethod
    def _fmt_reasons(reason_dict: Mapping[str, int]) -> str:
        if not reason_dict:
            return "-"
        items = sorted(reason_dict.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{key}:{value}" for key, value in items)

    def log_epoch(self, epoch_num: int, prefix: str = "[Health]") -> None:
        train_reasons = self._fmt_reasons(cast(Mapping[str, int], self.epoch["train_skip_reasons"]))
        val_reasons = self._fmt_reasons(cast(Mapping[str, int], self.epoch["val_skip_reasons"]))
        print(
            f"{prefix} Epoch {epoch_num} | "
            f"Train skip={self.epoch['train_skipped_batches']} "
            f"(reasons: {train_reasons}) | "
            f"Train NaN/Inf loss={self.epoch['train_naninf_loss']} | "
            f"Val skip={self.epoch['val_skipped_batches']} "
            f"(reasons: {val_reasons}) | "
            f"Val NaN/Inf loss={self.epoch['val_naninf_loss']} | "
            f"Val collapsed={self.epoch['val_collapsed']} | "
            f"Val invalid_metrics={self.epoch['val_invalid_metrics']}"
        )

    def log_run(self, prefix: str = "[Health-SUMMARY]") -> None:
        train_reasons = self._fmt_reasons(cast(Mapping[str, int], self.run["train_skip_reasons"]))
        val_reasons = self._fmt_reasons(cast(Mapping[str, int], self.run["val_skip_reasons"]))
        print(
            f"{prefix} Run totals | "
            f"Train skip={self.run['train_skipped_batches']} "
            f"(reasons: {train_reasons}) | "
            f"Train NaN/Inf loss={self.run['train_naninf_loss']} | "
            f"Val skip={self.run['val_skipped_batches']} "
            f"(reasons: {val_reasons}) | "
            f"Val NaN/Inf loss={self.run['val_naninf_loss']} | "
            f"Val collapse_epochs={self.run['val_collapse_epochs']} | "
            f"Val invalid_metric_epochs={self.run['val_invalid_metric_epochs']}"
        )


class RunningWeightedMetric:
    """Accumulate sample-weighted batch metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cumulative_score = 0.0
        self.sample_count = 0

    def update(self, batch_score_sum: float, batch_sample_count: int) -> None:
        self.cumulative_score += batch_score_sum
        self.sample_count += batch_sample_count

    def get_average(self) -> float:
        return self.cumulative_score / self.sample_count if self.sample_count > 0 else 0.0


class AdvancedMetricTracker:
    """Incremental validation tracker for binary segmentation metrics."""

    def __init__(
        self,
        device: torch.device,
        metric_bins: int = 2048,
        mcc_thresholds: torch.Tensor | list[float] | None = None,
        collapse_low: float = 0.001,
        collapse_high: float = 0.99,
        from_logits: bool = True,
    ) -> None:
        ensure_headless_matplotlib_backend()

        self.device = device
        self.from_logits = from_logits
        self.metric_bins = int(metric_bins)
        if self.metric_bins < MIN_METRIC_BINS:
            raise ValueError("metric_bins must be at least 2.")

        if mcc_thresholds is None:
            mcc_thresholds = torch.arange(0.1, 1.0, 0.1)
        self.mcc_thresholds = torch.as_tensor(mcc_thresholds, dtype=torch.float32).to(device)

        self.collapse_low = float(collapse_low)
        self.collapse_high = float(collapse_high)
        self.reset()

    def reset(self) -> None:
        self._positive_hist = torch.zeros(self.metric_bins, dtype=torch.int64, device=self.device)
        self._negative_hist = torch.zeros(self.metric_bins, dtype=torch.int64, device=self.device)
        self._pred_pos_at_05 = torch.zeros((), dtype=torch.int64, device=self.device)
        self._total_pixels = 0

    @staticmethod
    def _extract_probs_fg(pred_logits: torch.Tensor) -> torch.Tensor:
        if pred_logits.ndim != LOGIT_TENSOR_NDIM:
            raise ValueError(f"Expected pred_logits [B,C,H,W], got {tuple(pred_logits.shape)}")

        _, channels, _, _ = pred_logits.shape
        if channels == SINGLE_CHANNEL_COUNT:
            probs = torch.sigmoid(pred_logits[:, SINGLE_CHANNEL_INDEX, ...])
        elif channels == BINARY_CLASS_COUNT:
            probs = torch.softmax(pred_logits, dim=1)[:, FOREGROUND_CHANNEL_INDEX, ...]
        else:
            raise ValueError(f"Expected C=1 or C=2 for binary segmentation, got C={channels}")

        return probs.to(dtype=torch.float32)

    @staticmethod
    def _extract_target_fg(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == MASK_TENSOR_NDIM:
            foreground = target
        elif target.ndim == LOGIT_TENSOR_NDIM and target.shape[1] == BINARY_CLASS_COUNT:
            foreground = target[:, FOREGROUND_CHANNEL_INDEX, ...]
        elif target.ndim == LOGIT_TENSOR_NDIM and target.shape[1] == SINGLE_CHANNEL_COUNT:
            foreground = target[:, SINGLE_CHANNEL_INDEX, ...]
        else:
            raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")

        if foreground.dtype != torch.bool:
            foreground = foreground > DEFAULT_PROBABILITY_THRESHOLD
        return foreground

    @staticmethod
    def _extract_probs_from_probs_fg(probs_fg: torch.Tensor) -> torch.Tensor:
        if probs_fg.ndim == LOGIT_TENSOR_NDIM and probs_fg.shape[1] == SINGLE_CHANNEL_COUNT:
            probs_fg = probs_fg[:, SINGLE_CHANNEL_INDEX, ...]
        if probs_fg.ndim != MASK_TENSOR_NDIM:
            raise ValueError(f"Expected probs_fg [B,H,W], got {tuple(probs_fg.shape)}")
        return probs_fg.to(dtype=torch.float32).clamp(0.0, 1.0)

    def _reset_after_failure(
        self,
        message: str,
        health: TrainingHealthTracker | None,
        *,
        invalid_metrics: bool = False,
    ) -> None:
        print(message)
        if health is not None:
            if invalid_metrics:
                health.mark_val_invalid_metrics()
            else:
                health.mark_val_collapsed()
        self.reset()

    def _check_fg_prevalence(
        self,
        fg_prevalence: float,
        health: TrainingHealthTracker | None,
    ) -> bool:
        if fg_prevalence < self.collapse_low:
            self._reset_after_failure(
                "[VAL GUARDRAIL] COLLAPSE DETECTED - "
                f"FG prevalence @0.5 too LOW: {fg_prevalence:.6f} "
                f"(threshold {self.collapse_low:.6f})",
                health,
            )
            return False
        if fg_prevalence > self.collapse_high:
            self._reset_after_failure(
                "[VAL GUARDRAIL] COLLAPSE DETECTED - "
                f"FG prevalence @0.5 too HIGH: {fg_prevalence:.6f} "
                f"(threshold {self.collapse_high:.6f})",
                health,
            )
            return False
        return True

    def _validate_metric_tensor(
        self,
        metric_tensor: torch.Tensor,
        metric_name: str,
        health: TrainingHealthTracker | None,
    ) -> bool:
        if (metric_tensor.numel() == 0) or (not torch.isfinite(metric_tensor).all()):
            self._reset_after_failure(
                f"[VAL GUARDRAIL] INVALID METRIC - {metric_name} is NaN/Inf: {metric_tensor}",
                health,
                invalid_metrics=True,
            )
            return False
        return True

    def _build_final_metrics(
        self,
        auprc_tensor: torch.Tensor,
        auroc_tensor: torch.Tensor,
        mcc_values: torch.Tensor,
        fg_prevalence: float,
        health: TrainingHealthTracker | None,
    ) -> dict[str, float] | None:
        val_auprc = float(auprc_tensor.item())
        val_auroc = float(auroc_tensor.item())
        if not self._validate_metric_tensor(mcc_values, "MCC*", health):
            return None

        val_mcc_star = float(torch.max(mcc_values).item())
        if not (
            math.isfinite(val_auprc) and math.isfinite(val_auroc) and math.isfinite(val_mcc_star)
        ):
            self._reset_after_failure(
                f"[VAL GUARDRAIL] INVALID METRIC - AUPRC={val_auprc}, AUROC={val_auroc}, "
                f"MCC*={val_mcc_star}",
                health,
                invalid_metrics=True,
            )
            return None

        metrics = {
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "val_mcc_star": val_mcc_star,
            "fg_prevalence_at_05": float(fg_prevalence),
        }
        self.reset()
        return metrics

    @staticmethod
    def _mcc_from_counts(
        tp: torch.Tensor,
        fp: torch.Tensor,
        tn: torch.Tensor,
        fn: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        tp = tp.to(torch.float64)
        fp = fp.to(torch.float64)
        tn = tn.to(torch.float64)
        fn = fn.to(torch.float64)
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        return (numerator / denominator).to(torch.float32)

    def _update_score_histograms(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = probabilities.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        targets = targets.to(device=self.device, dtype=torch.bool)
        bin_indices = torch.clamp(
            (probabilities * self.metric_bins).to(torch.int64),
            max=self.metric_bins - 1,
        )
        combined_indices = bin_indices + targets.to(torch.int64) * self.metric_bins
        counts = torch.bincount(combined_indices, minlength=self.metric_bins * 2)
        self._negative_hist += counts[: self.metric_bins]
        self._positive_hist += counts[self.metric_bins :]
        self._pred_pos_at_05 += torch.count_nonzero(probabilities >= DEFAULT_PROBABILITY_THRESHOLD)
        self._total_pixels += int(probabilities.numel())

    def _counts_at_or_above_thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive_at_or_above = torch.flip(
            torch.cumsum(torch.flip(self._positive_hist, dims=(0,)), dim=0),
            dims=(0,),
        )
        negative_at_or_above = torch.flip(
            torch.cumsum(torch.flip(self._negative_hist, dims=(0,)), dim=0),
            dims=(0,),
        )
        threshold_bins = torch.clamp(
            torch.floor(self.mcc_thresholds * self.metric_bins).to(torch.int64),
            min=0,
            max=self.metric_bins - 1,
        )
        return positive_at_or_above[threshold_bins], negative_at_or_above[threshold_bins]

    def _compute_mcc_values_from_histograms(self) -> torch.Tensor:
        tp, fp = self._counts_at_or_above_thresholds()
        total_positive = torch.sum(self._positive_hist)
        total_negative = torch.sum(self._negative_hist)
        fn = total_positive - tp
        tn = total_negative - fp
        return self._mcc_from_counts(tp, fp, tn, fn)

    def _compute_auroc_from_histograms(self) -> torch.Tensor:
        total_positive = torch.sum(self._positive_hist)
        total_negative = torch.sum(self._negative_hist)
        if int(total_positive.item()) == 0 or int(total_negative.item()) == 0:
            return torch.tensor(float("nan"), device=self.device)

        positive_desc = torch.flip(self._positive_hist, dims=(0,)).to(torch.float64)
        negative_desc = torch.flip(self._negative_hist, dims=(0,)).to(torch.float64)
        true_positive_rate = torch.cumsum(positive_desc, dim=0) / total_positive.to(torch.float64)
        false_positive_rate = torch.cumsum(negative_desc, dim=0) / total_negative.to(torch.float64)
        zero = torch.zeros(1, dtype=torch.float64, device=self.device)
        curve_tpr = torch.cat((zero, true_positive_rate))
        curve_fpr = torch.cat((zero, false_positive_rate))
        return torch.trapezoid(curve_tpr, curve_fpr).to(torch.float32)

    def _compute_auprc_from_histograms(self) -> torch.Tensor:
        total_positive = torch.sum(self._positive_hist)
        if int(total_positive.item()) == 0:
            return torch.tensor(float("nan"), device=self.device)

        positive_desc = torch.flip(self._positive_hist, dims=(0,)).to(torch.float64)
        negative_desc = torch.flip(self._negative_hist, dims=(0,)).to(torch.float64)
        true_positive = torch.cumsum(positive_desc, dim=0)
        false_positive = torch.cumsum(negative_desc, dim=0)
        precision = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
        recall = true_positive / total_positive.to(torch.float64)
        previous_recall = torch.cat(
            (torch.zeros(1, dtype=torch.float64, device=self.device), recall[:-1])
        )
        recall_delta = recall - previous_recall
        return torch.sum(recall_delta * precision).to(torch.float32)

    def _compute_threshold_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._compute_auprc_from_histograms(),
            self._compute_auroc_from_histograms(),
            self._compute_mcc_values_from_histograms(),
        )

    def _update_from_flat_probs(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        self._update_score_histograms(probabilities, targets)

    @torch.no_grad()
    def update(self, pred_logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = self._extract_probs_fg(pred_logits).reshape(-1)
        targets = self._extract_target_fg(target).reshape(-1)
        self._update_from_flat_probs(probabilities, targets)

    @torch.no_grad()
    def update_from_probs_fg(self, probs_fg: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = self._extract_probs_from_probs_fg(probs_fg).reshape(-1)
        targets = self._extract_target_fg(target).reshape(-1)
        self._update_from_flat_probs(probabilities, targets)

    def compute_and_reset(
        self, health: TrainingHealthTracker | None = None
    ) -> dict[str, float] | None:
        if self._total_pixels == 0:
            print("[VAL GUARDRAIL] No pixels processed in validation epoch.")
            self.reset()
            return None

        fg_prevalence = int(self._pred_pos_at_05.detach().cpu().item()) / float(self._total_pixels)
        if not self._check_fg_prevalence(fg_prevalence, health):
            return None

        auprc_tensor, auroc_tensor, mcc_values = self._compute_threshold_metrics()
        auprc_tensor = auprc_tensor.detach().cpu()
        auroc_tensor = auroc_tensor.detach().cpu()
        mcc_values = mcc_values.detach().cpu()
        if not self._validate_metric_tensor(auprc_tensor, "AUPRC", health):
            return None
        if not self._validate_metric_tensor(auroc_tensor, "AUROC", health):
            return None

        return self._build_final_metrics(
            auprc_tensor,
            auroc_tensor,
            mcc_values,
            fg_prevalence,
            health,
        )
