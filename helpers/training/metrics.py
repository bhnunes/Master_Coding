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
        from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

        self.device = device
        self.from_logits = from_logits
        self.auprc = BinaryAveragePrecision(thresholds=metric_bins).to(device)
        self.auroc = BinaryAUROC(thresholds=metric_bins).to(device)

        if mcc_thresholds is None:
            mcc_thresholds = torch.arange(0.1, 1.0, 0.1)
        self.mcc_thresholds = torch.as_tensor(mcc_thresholds, dtype=torch.float32).to(device)

        self.collapse_low = float(collapse_low)
        self.collapse_high = float(collapse_high)
        self.reset()

    def reset(self) -> None:
        self.auprc.reset()
        self.auroc.reset()
        self._pred_pos_at_05 = 0
        self._total_pixels = 0

        threshold_count = int(self.mcc_thresholds.numel())
        self._tp = torch.zeros(threshold_count, dtype=torch.int64, device=self.device)
        self._fp = torch.zeros(threshold_count, dtype=torch.int64, device=self.device)
        self._tn = torch.zeros(threshold_count, dtype=torch.int64, device=self.device)
        self._fn = torch.zeros(threshold_count, dtype=torch.int64, device=self.device)

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
        fg_prevalence: float,
        health: TrainingHealthTracker | None,
    ) -> dict[str, float] | None:
        val_auprc = float(auprc_tensor.item())
        val_auroc = float(auroc_tensor.item())
        mcc_values = self._mcc_from_counts(
            self._tp.cpu(),
            self._fp.cpu(),
            self._tn.cpu(),
            self._fn.cpu(),
        )
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

    def _update_from_flat_probs(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        self.auprc.update(probabilities, targets)
        self.auroc.update(probabilities, targets)

        self._pred_pos_at_05 += int((probabilities >= DEFAULT_PROBABILITY_THRESHOLD).sum().item())
        self._total_pixels += int(probabilities.numel())

        preds_k = probabilities.unsqueeze(0) >= self.mcc_thresholds.unsqueeze(1)
        targets_k = targets.unsqueeze(0)
        self._tp += (preds_k & targets_k).sum(dim=1)
        self._fp += (preds_k & ~targets_k).sum(dim=1)
        self._tn += (~preds_k & ~targets_k).sum(dim=1)
        self._fn += (~preds_k & targets_k).sum(dim=1)

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

        fg_prevalence = self._pred_pos_at_05 / float(self._total_pixels)
        if not self._check_fg_prevalence(fg_prevalence, health):
            return None

        auprc_tensor = self.auprc.compute().detach().cpu()
        auroc_tensor = self.auroc.compute().detach().cpu()
        if not self._validate_metric_tensor(auprc_tensor, "AUPRC", health):
            return None
        if not self._validate_metric_tensor(auroc_tensor, "AUROC", health):
            return None

        return self._build_final_metrics(auprc_tensor, auroc_tensor, fg_prevalence, health)
