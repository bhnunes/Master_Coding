from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CurveStats:
    min_loss: float
    min_loss_lr: float


def moving_average(values: npt.NDArray[np.float64], window: int = 5) -> npt.NDArray[np.float64]:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def compute_curve_stats(
    lrs: npt.NDArray[np.float64],
    losses: npt.NDArray[np.float64],
    skip_start: int = 10,
    skip_end: int = 5,
) -> CurveStats:
    learning_rates = np.asarray(lrs, dtype=np.float64)
    curve_losses = np.asarray(losses, dtype=np.float64)
    lower_bound = skip_start
    upper_bound = max(lower_bound + 1, len(learning_rates) - skip_end)
    if lower_bound >= upper_bound:
        return CurveStats(min_loss=float("inf"), min_loss_lr=0.0)

    trimmed_lrs = learning_rates[lower_bound:upper_bound]
    trimmed_losses = curve_losses[lower_bound:upper_bound]
    finite_mask = np.isfinite(trimmed_lrs) & np.isfinite(trimmed_losses) & (trimmed_lrs > 0)
    trimmed_lrs = trimmed_lrs[finite_mask]
    trimmed_losses = trimmed_losses[finite_mask]
    if len(trimmed_lrs) < 5:
        return CurveStats(min_loss=float("inf"), min_loss_lr=0.0)

    smooth_losses = moving_average(trimmed_losses, window=5)
    minimum_index = int(np.argmin(smooth_losses))
    return CurveStats(
        min_loss=float(smooth_losses[minimum_index]),
        min_loss_lr=float(trimmed_lrs[minimum_index]),
    )
