from __future__ import annotations

import numpy as np

from helpers.lr_finder.analysis import compute_curve_stats
from helpers.lr_finder.search_space import BCEDiceSearchSpace, sample_bcedice_params


def test_sample_bcedice_params_is_deterministic_and_bounded() -> None:
    space = BCEDiceSearchSpace(
        alpha_min=0.1,
        alpha_max=0.9,
        beta_min=0.2,
        beta_max=0.4,
        gamma_min=0.05,
        gamma_max=0.25,
    )

    first = sample_bcedice_params(5, space, seed=24)
    second = sample_bcedice_params(5, space, seed=24)

    assert first == second
    assert len(first) == 5
    for params in first:
        assert 0.1 <= params.alpha <= 0.9
        assert 0.2 <= params.beta <= 0.4
        assert 0.05 <= params.gamma <= 0.25


def test_compute_curve_stats_uses_trimmed_smoothed_minimum() -> None:
    lrs = np.logspace(-6, -1, num=12)
    losses = np.array([10.0, 9.0, 1.3, 1.2, 1.1, 0.4, 0.5, 0.6, 0.8, 5.0, 6.0, 7.0])

    stats = compute_curve_stats(lrs, losses, skip_start=2, skip_end=2)

    assert stats.min_loss < 1.0
    assert stats.min_loss_lr == lrs[6]
