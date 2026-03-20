from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BCEDiceSearchSpace:
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 0.5
    gamma_min: float = 0.1
    gamma_max: float = 0.5
    gamma_log: bool = False


@dataclass(frozen=True)
class BCEDiceParams:
    alpha: float
    beta: float
    gamma: float


def latin_hypercube(n_samples: int, n_dims: int, seed: int = 24) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dims), dtype=np.float64)
    for dimension in range(n_dims):
        cuts = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.random(n_samples)
        samples[:, dimension] = cuts[:-1] + u * (cuts[1:] - cuts[:-1])
        rng.shuffle(samples[:, dimension])
    return samples


def sample_bcedice_params(
    n_samples: int, space: BCEDiceSearchSpace, seed: int
) -> list[BCEDiceParams]:
    lhs_samples = latin_hypercube(n_samples, 3, seed)
    params: list[BCEDiceParams] = []
    for alpha_u, beta_u, gamma_u in lhs_samples:
        alpha = space.alpha_min + alpha_u * (space.alpha_max - space.alpha_min)
        beta = space.beta_min + beta_u * (space.beta_max - space.beta_min)
        if space.gamma_log:
            gamma_low = math.log(space.gamma_min)
            gamma_high = math.log(space.gamma_max)
            gamma = math.exp(gamma_low + gamma_u * (gamma_high - gamma_low))
        else:
            gamma = space.gamma_min + gamma_u * (space.gamma_max - space.gamma_min)
        params.append(BCEDiceParams(alpha=float(alpha), beta=float(beta), gamma=float(gamma)))
    return params
