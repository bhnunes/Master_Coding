"""
GIST: Greedy Independent Set Thresholding for Diverse Data Summarization
========================================================================
Implementation based on:
  - Paper: "GIST: Greedy Independent Set Thresholding for Diverse Data Summarization"
           Fahrbach et al., arXiv:2405.18754v2
  - Blog:  "Introducing GIST: The Next Stage in Smart Sampling"
           Google Research, January 2026

Problem:
  Min-Distance Diversification with Monotone Submodular Utility (MDMS).

  Given n points V in a metric space and a monotone submodular utility
  function g : 2^V → R≥0, select a subset S ⊆ V with |S| ≤ k that
  maximises:

      f(S) = g(S) + div(S)

  where div(S) = min_{u,v ∈ S, u≠v} dist(u, v)  (max-min diversity).

Algorithm overview:
  1. Run GreedyIndependentSet with threshold d=0 (classic greedy).
  2. Initialise with the two diametrically opposite points.
  3. Sweep over O(ε⁻¹ log ε⁻¹) distance thresholds D.
  4. For each threshold d ∈ D, call GreedyIndependentSet to obtain a
     candidate set T; keep the best f(T) seen so far.

Approximation guarantee: (1/2 − ε) · OPT  for any ε > 0.
For linear utility functions the guarantee strengthens to 2/3 · OPT.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.float64]
Points = FloatArray  # shape (n, d) – rows are point vectors
UtilityFn = Callable[[list[int], int | None], float]
# g(S_indices, new_index) → marginal gain g(new | S)
# When new_index is None → g(S_indices) absolute value


# ---------------------------------------------------------------------------
# Built-in utility functions
# ---------------------------------------------------------------------------


def linear_utility(weights: FloatArray) -> UtilityFn:
    """
    g(S) = Σ w(v)  for v in S  (linear / modular function).

    Marginal gain: g(v | S) = w(v)  (independent of S).
    """

    def _g(selected: list[int], new_idx: int | None) -> float:
        if new_idx is None:
            return float(np.sum(weights[selected]))
        return float(weights[new_idx])

    return _g


def facility_location_utility(
    points: Points,
    reference: Points | None = None,
) -> UtilityFn:
    """
    g(S) = Σ_{v ∈ V} max_{u ∈ S} sim(u, v)  (facility-location).

    A classic monotone submodular function widely used in data summarisation.
    sim(u, v) = 1 - dist(u, v) / diam(V)  (normalised similarity).

    Args:
        points:    The full dataset (n, d).
        reference: Optional set of query / reference points (m, d).
                   Defaults to `points` itself.
    """
    if reference is None:
        reference = points
    # Pre-compute all pairwise similarities (n_ref × n_points)
    n_ref = len(reference)
    dists = cast(FloatArray, np.linalg.norm(reference[:, None, :] - points[None, :, :], axis=-1))
    diam = dists.max() or 1.0
    sims = 1.0 - dists / diam  # (n_ref, n_pts)

    # covered[i] = current best similarity for reference point i
    covered: FloatArray = np.zeros(n_ref, dtype=np.float64)

    def _g(selected: list[int], new_idx: int | None) -> float:
        if new_idx is None:
            if not selected:
                return 0.0
            best = np.max(sims[:, selected], axis=1)  # (n_ref,)
            return float(best.sum())
        # Marginal gain
        new_sim = sims[:, new_idx]
        gain = np.maximum(new_sim, covered) - covered
        return float(gain.sum())

    # Patch to update `covered` after each selection (stateful)
    _g._covered = covered  # type: ignore[attr-defined]
    _g._sims = sims  # type: ignore[attr-defined]
    return _g


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _pairwise_distance_matrix(points: Points) -> FloatArray:
    """Compute full n×n Euclidean distance matrix."""
    diff = points[:, None, :] - points[None, :, :]  # (n, n, d)
    return cast(FloatArray, np.linalg.norm(diff, axis=-1))  # (n, n)


def _min_dist_to_set(v: int, S: list[int], dist_mat: FloatArray) -> float:
    """dist(v, S) = min_{u ∈ S} dist(v, u).  Returns ∞ if S is empty."""
    if not S:
        return math.inf
    return float(dist_mat[v, S].min())


def _diversity(S: list[int], dist_mat: FloatArray) -> float:
    """div(S) = min_{u≠v ∈ S} dist(u, v).  Returns d_max if |S| ≤ 1."""
    d_max = float(dist_mat.max())
    if len(S) <= 1:
        return d_max
    pairs = dist_mat[np.ix_(S, S)]
    np.fill_diagonal(pairs, math.inf)
    return float(pairs.min())


# ---------------------------------------------------------------------------
# Core subroutines
# ---------------------------------------------------------------------------


class GIST:
    """
    Greedy Independent Set Thresholding (GIST) for diverse data summarisation.

    Solves the MDMS problem:
        argmax_{S ⊆ V, |S| ≤ k}  g(S) + div(S)

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
        Dataset in a metric (Euclidean) space.
    utility_fn : UtilityFn
        Monotone submodular utility.  Signature:
            utility_fn(selected_indices, new_index) -> float (marginal gain)
            utility_fn(selected_indices, None)      -> float (absolute value)
    k : int
        Cardinality budget.
    eps : float, optional
        Approximation parameter ε ∈ (0, 0.5).  Smaller → more thresholds,
        better approximation, higher runtime.  Default: 0.1.
    dist_matrix : np.ndarray or None, optional
        Pre-computed n×n distance matrix.  Computed from `points` if None.
    verbose : bool, optional
        Print progress information.  Default: False.

    References
    ----------
    Fahrbach et al., "GIST: Greedy Independent Set Thresholding for Diverse
    Data Summarization," arXiv:2405.18754v2, 2025.
    """

    def __init__(
        self,
        points: Points,
        utility_fn: UtilityFn,
        k: int,
        eps: float = 0.1,
        dist_matrix: FloatArray | None = None,
        verbose: bool = False,
    ) -> None:
        if not (0 < eps < 0.5):
            raise ValueError("eps must be in (0, 0.5).")
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.points = points
        self.g = utility_fn
        self.k = k
        self.eps = eps
        self.verbose = verbose
        self.n = len(points)

        # Distance matrix
        if dist_matrix is not None:
            self.dist_mat = dist_matrix
        else:
            self.dist_mat = _pairwise_distance_matrix(points)

        self.d_max: float = float(self.dist_mat.max())

        # Results (populated after fit())
        self.selected_indices_: list[int] = []
        self.objective_value_: float = -math.inf
        self._all_candidates_: list[tuple[float, list[int]]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> GIST:
        """
        Run the GIST algorithm and store results in `selected_indices_` and
        `objective_value_`.

        Returns
        -------
        self
        """
        best_S: list[int] = []
        best_f: float = -math.inf

        def evaluate(candidate: list[int]) -> float:
            utility = self.g(candidate, None)
            diversity = _diversity(candidate, self.dist_mat)
            return utility + diversity

        def update_best(candidate: list[int]) -> None:
            nonlocal best_S, best_f
            if not candidate:
                return
            f_val = evaluate(candidate)
            self._all_candidates_.append((f_val, list(candidate)))
            if f_val > best_f:
                best_f = f_val
                best_S = list(candidate)
                if self.verbose:
                    print(f"  New best  f={best_f:.4f}  |S|={len(best_S)}")

        # ── Step 1: Classic greedy (d = 0) ────────────────────────────
        if self.verbose:
            print("Step 1: Classic greedy (d=0)")
        S0 = self._greedy_independent_set(d=0.0)
        update_best(S0)

        # ── Step 2: Diametrical pair ───────────────────────────────────
        if self.k >= 2:
            if self.verbose:
                print("Step 2: Diametrical pair")
            u, v = self._diametrical_pair()
            update_best([u, v])

        # ── Step 3: Threshold sweep ────────────────────────────────────
        thresholds = self._compute_thresholds()
        if self.verbose:
            print(f"Step 3: Sweeping {len(thresholds)} thresholds …")
        for idx, d in enumerate(thresholds):
            T = self._greedy_independent_set(d=d)
            update_best(T)
            if self.verbose:
                print(f"  [{idx + 1}/{len(thresholds)}]  d={d:.4f}  |T|={len(T)}")

        self.selected_indices_ = best_S
        self.objective_value_ = best_f
        return self

    @property
    def selected_points(self) -> Points:
        """Return the selected data points as a numpy array."""
        if not self.selected_indices_:
            raise RuntimeError("Call fit() before accessing selected_points.")
        return self.points[self.selected_indices_]

    def diversity(self) -> float:
        """Min-pairwise distance of the selected set."""
        return _diversity(self.selected_indices_, self.dist_mat)

    def utility(self) -> float:
        """Submodular utility value g(S) of the selected set."""
        return self.g(self.selected_indices_, None)

    def summary(self) -> dict[str, float | int | list[int]]:
        """Return a summary dictionary of the solution."""
        return {
            "selected_indices": self.selected_indices_,
            "k": len(self.selected_indices_),
            "utility": self.utility(),
            "diversity": self.diversity(),
            "objective": self.objective_value_,
            "n_candidates_evaluated": len(self._all_candidates_),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(self) -> list[float]:
        """
        Build the set D of distance thresholds (Algorithm 1, line 7):

            D = { (1 + ε)^i · ε · d_max / 2 : (1 + ε)^i ≤ 2/ε, i ∈ Z≥0 }
        """
        thresholds: list[float] = []
        eps = self.eps
        d_start = eps * self.d_max / 2.0
        limit = 2.0 / eps
        multiplier = 1.0
        while multiplier <= limit + 1e-12:
            thresholds.append(multiplier * d_start)
            multiplier *= 1.0 + eps
        return thresholds

    def _greedy_independent_set(self, d: float) -> list[int]:
        """
        GreedyIndependentSet (Algorithm 1, inner function):

        Builds S greedily by adding the point with the highest marginal
        utility gain among candidates that are at distance ≥ d from all
        already-selected points.

        Stops when |S| = k or no eligible candidate remains (maximal
        independent set of the intersection graph G_d(V)).

        Parameters
        ----------
        d : float
            Distance threshold.

        Returns
        -------
        List[int]
            Indices of selected points.
        """
        S: list[int] = []
        # Track minimum distance of each candidate to the current set S
        min_dist_to_S = np.full(self.n, math.inf)

        for _ in range(self.k):
            # Eligible candidates: not yet in S and dist(v, S) ≥ d
            if S:
                eligible_mask = min_dist_to_S >= d
            else:
                eligible_mask = np.ones(self.n, dtype=bool)

            # Exclude already-selected points
            eligible_mask[S] = False
            eligible_indices = np.where(eligible_mask)[0].tolist()

            if not eligible_indices:
                break  # Maximal independent set reached

            # Greedy: pick argmax marginal gain g(v | S)
            best_v: int = -1
            best_gain: float = -math.inf
            for v in eligible_indices:
                gain = self.g(S, v)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v

            if best_v == -1:
                break

            S.append(best_v)

            # Update min distances incrementally
            if d > 0:
                new_dists = self.dist_mat[best_v, :]
                min_dist_to_S = np.minimum(min_dist_to_S, new_dists)

        return S

    def _diametrical_pair(self) -> tuple[int, int]:
        """Return the pair of points (u, v) achieving d_max = max dist(u, v)."""
        idx = int(np.argmax(self.dist_mat))
        u = idx // self.n
        v = idx % self.n
        return u, v


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def gist_linear(
    points: Points,
    weights: FloatArray,
    k: int,
    eps: float = 0.1,
    verbose: bool = False,
) -> GIST:
    """
    Shortcut: run GIST with a linear utility function g(S) = Σ w(v).

    Approximation guarantee: 2/3 · OPT (stronger than the general 1/2 bound).
    """
    g = linear_utility(weights)
    return GIST(points, g, k, eps=eps, verbose=verbose).fit()


def gist_facility_location(
    points: Points,
    k: int,
    eps: float = 0.1,
    verbose: bool = False,
) -> GIST:
    """
    Shortcut: run GIST with a facility-location utility function.

    g(S) = Σ_{v ∈ V} max_{u ∈ S} sim(u, v)
    """
    g = facility_location_utility(points)
    return GIST(points, g, k, eps=eps, verbose=verbose).fit()


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("GIST Demo — 50 random 2-D points, k=8, ε=0.1")
    print("=" * 60)

    n, d, k = 50, 2, 8
    pts = np.random.rand(n, d)

    # ── Linear utility ────────────────────────────────────────────────
    weights = np.random.rand(n)
    gist_lin = gist_linear(pts, weights, k, eps=0.1, verbose=True)
    print("\n[Linear utility]")
    print(gist_lin.summary())

    # ── Facility-location utility ─────────────────────────────────────
    gist_fl = gist_facility_location(pts, k, eps=0.1, verbose=False)
    print("\n[Facility-location utility]")
    print(gist_fl.summary())

    # ── Custom monotone submodular utility ────────────────────────────
    # Coverage: g(S) = |∪_{v ∈ S} N(v)|  where N(v) is a random neighbourhood
    rng = np.random.default_rng(0)
    neighbourhoods = [
        set(rng.choice(100, size=rng.integers(5, 20), replace=False).tolist()) for _ in range(n)
    ]

    def coverage_utility(selected: list[int], new_idx: int | None) -> float:
        covered: set[int] = set()
        for i in selected:
            covered |= neighbourhoods[i]
        if new_idx is None:
            return float(len(covered))
        return float(len(covered | neighbourhoods[new_idx]) - len(covered))

    gist_cov = GIST(pts, coverage_utility, k, eps=0.1, verbose=False).fit()
    print("\n[Coverage utility]")
    print(gist_cov.summary())
