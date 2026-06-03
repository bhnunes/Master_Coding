from __future__ import annotations

import numpy as np

from helpers.lr_finder.analysis import compute_curve_stats
from helpers.lr_finder.search_space import BCEDiceSearchSpace, sample_bcedice_params

SAMPLE_COUNT = 5
ALPHA_MIN = 0.1
ALPHA_MAX = 0.9
BETA_MIN = 0.2
BETA_MAX = 0.4
GAMMA_MIN = 0.05
GAMMA_MAX = 0.25
LOW_MIN_LOSS_BOUND = 1.0
EXPECTED_MIN_LOSS_INDEX = 6


def test_sample_bcedice_params_is_deterministic_and_bounded() -> None:
    space = BCEDiceSearchSpace(
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        gamma_min=GAMMA_MIN,
        gamma_max=GAMMA_MAX,
    )

    first = sample_bcedice_params(SAMPLE_COUNT, space, seed=24)
    second = sample_bcedice_params(SAMPLE_COUNT, space, seed=24)

    assert first == second
    assert len(first) == SAMPLE_COUNT
    for params in first:
        assert ALPHA_MIN <= params.alpha <= ALPHA_MAX
        assert BETA_MIN <= params.beta <= BETA_MAX
        assert GAMMA_MIN <= params.gamma <= GAMMA_MAX


def test_compute_curve_stats_uses_trimmed_smoothed_minimum() -> None:
    lrs = np.logspace(-6, -1, num=12)
    losses = np.array([10.0, 9.0, 1.3, 1.2, 1.1, 0.4, 0.5, 0.6, 0.8, 5.0, 6.0, 7.0])

    stats = compute_curve_stats(lrs, losses, skip_start=2, skip_end=2)

    assert stats.min_loss < LOW_MIN_LOSS_BOUND
    assert stats.min_loss_lr == lrs[EXPECTED_MIN_LOSS_INDEX]
