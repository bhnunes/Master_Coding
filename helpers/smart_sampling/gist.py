from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

_DEFAULT_GIST_EPS = 0.1
MIN_PAIRWISE_CANDIDATE_SIZE = 2
_PROGRESS_LOG_MIN_CANDIDATES = 1024


@dataclass(frozen=True)
class GistSelectionResult:
    selected_positions: npt.NDArray[np.int64]
    objective_trace: list[tuple[int, float]]


def select_gist_facility_location(
    embeddings: npt.NDArray[np.float32],
    *,
    max_selected: int,
) -> GistSelectionResult:
    """Select a diverse subset with a GIST-style facility-location objective."""

    n_samples = len(embeddings)
    if n_samples == 0 or max_selected <= 0:
        return GistSelectionResult(
            selected_positions=np.empty(0, dtype=np.int64),
            objective_trace=[],
        )

    if n_samples <= max_selected:
        return GistSelectionResult(
            selected_positions=np.arange(n_samples, dtype=np.int64),
            objective_trace=[(n_samples, 0.0)],
        )

    points = np.asarray(embeddings, dtype=np.float32)
    if n_samples >= _PROGRESS_LOG_MIN_CANDIDATES:
        logging.info(
            "Running GIST facility-location selector on %d candidates with target %d",
            n_samples,
            max_selected,
        )
    dist_mat = _pairwise_distance_matrix(points)
    sims = _facility_location_similarity(points, dist_mat)
    selector = _GistSelector(
        dist_mat=dist_mat,
        sims=sims,
        k=max_selected,
        eps=_DEFAULT_GIST_EPS,
    )
    selected_positions, objective_trace = selector.fit()
    if n_samples >= _PROGRESS_LOG_MIN_CANDIDATES:
        logging.info(
            "GIST facility-location selector finished with %d selected positions "
            "after evaluating %d threshold candidate sets",
            len(selected_positions),
            len(objective_trace),
        )
    return GistSelectionResult(
        selected_positions=np.asarray(selected_positions, dtype=np.int64),
        objective_trace=objective_trace,
    )


def _pairwise_distance_matrix(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    points = np.asarray(points, dtype=np.float32)
    squared_norms = np.sum(points * points, axis=1, dtype=np.float32)
    squared_distances = squared_norms[:, None] + squared_norms[None, :]
    squared_distances -= np.asarray(2.0 * (points @ points.T), dtype=np.float32)
    np.maximum(squared_distances, 0.0, out=squared_distances)
    distances = np.sqrt(squared_distances, out=squared_distances).astype(np.float32, copy=False)
    np.fill_diagonal(distances, 0.0)
    return cast(npt.NDArray[np.float32], distances)


def _facility_location_similarity(
    points: npt.NDArray[np.float32],
    dist_mat: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    del points
    diameter = float(np.max(dist_mat))
    if diameter <= 0.0:
        diameter = 1.0
    return (1.0 - (dist_mat / diameter)).astype(np.float32, copy=False)


class _GistSelector:
    def __init__(
        self,
        *,
        dist_mat: npt.NDArray[np.float32],
        sims: npt.NDArray[np.float32],
        k: int,
        eps: float,
    ) -> None:
        self.dist_mat = dist_mat
        self.sims = sims
        self.k = k
        self.eps = eps
        self.n = int(dist_mat.shape[0])
        self.d_max = float(np.max(dist_mat)) if self.n > 0 else 0.0
        self.candidate_order = self._rank_candidates_by_singleton_coverage()

    def fit(self) -> tuple[list[int], list[tuple[int, float]]]:
        best_selected: list[int] = []
        best_objective = -math.inf
        objective_trace: list[tuple[int, float]] = []

        for candidate in self._candidate_sets():
            if not candidate:
                continue
            objective = self._objective(candidate)
            objective_trace.append((len(candidate), objective))
            if objective > best_objective:
                best_objective = objective
                best_selected = candidate

        if not best_selected:
            best_selected = [0]
            objective_trace.append((1, self._objective(best_selected)))
        return best_selected, objective_trace

    def _candidate_sets(self) -> list[list[int]]:
        candidates = [self._greedy_independent_set(0.0)]
        if self.k >= MIN_PAIRWISE_CANDIDATE_SIZE and self.n >= MIN_PAIRWISE_CANDIDATE_SIZE:
            u, v = self._diametrical_pair()
            if u != v:
                candidates.append([u, v])
        for threshold in self._compute_thresholds():
            candidates.append(self._greedy_independent_set(threshold))
        return candidates

    def _compute_thresholds(self) -> list[float]:
        if self.d_max <= 0.0:
            return []
        thresholds: list[float] = []
        d_start = self.eps * self.d_max / 2.0
        limit = 2.0 / self.eps
        multiplier = 1.0
        while multiplier <= limit + 1e-12:
            thresholds.append(multiplier * d_start)
            multiplier *= 1.0 + self.eps
        return thresholds

    def _greedy_independent_set(self, threshold: float) -> list[int]:
        selected: list[int] = []
        min_dist_to_selected = np.full(self.n, np.inf, dtype=np.float32)

        for position in self.candidate_order.tolist():
            if len(selected) >= self.k:
                break
            if selected and min_dist_to_selected[position] < threshold:
                continue
            selected.append(int(position))
            if threshold > 0.0:
                min_dist_to_selected = np.minimum(min_dist_to_selected, self.dist_mat[position])

        return selected

    def _objective(self, selected: list[int]) -> float:
        return self._facility_location_value(selected) + self._diversity(selected)

    def _facility_location_value(self, selected: list[int]) -> float:
        if not selected:
            return 0.0
        return float(np.sum(np.max(self.sims[:, selected], axis=1), dtype=np.float64))

    def _diversity(self, selected: list[int]) -> float:
        if len(selected) <= 1:
            return self.d_max
        pairs = self.dist_mat[np.ix_(selected, selected)].copy()
        np.fill_diagonal(pairs, np.inf)
        return float(np.min(pairs))

    def _diametrical_pair(self) -> tuple[int, int]:
        index = int(np.argmax(self.dist_mat))
        return index // self.n, index % self.n

    def _rank_candidates_by_singleton_coverage(self) -> npt.NDArray[np.int64]:
        singleton_coverage = np.sum(self.sims, axis=0, dtype=np.float64)
        ranked = np.lexsort((np.arange(self.n, dtype=np.int64), -singleton_coverage)).astype(
            np.int64,
            copy=False,
        )
        return cast(npt.NDArray[np.int64], ranked)
