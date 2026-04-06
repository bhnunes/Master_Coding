from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_DEFAULT_GIST_EPS = 0.1


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
    dist_mat = _pairwise_distance_matrix(points)
    sims = _facility_location_similarity(points, dist_mat)
    selector = _GistSelector(
        dist_mat=dist_mat,
        sims=sims,
        k=max_selected,
        eps=_DEFAULT_GIST_EPS,
    )
    selected_positions, objective_trace = selector.fit()
    return GistSelectionResult(
        selected_positions=np.asarray(selected_positions, dtype=np.int64),
        objective_trace=objective_trace,
    )


def _pairwise_distance_matrix(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return np.asarray(distances, dtype=np.float32)


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
        if self.k >= 2 and self.n >= 2:
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
        covered = np.zeros(self.n, dtype=np.float32)
        min_dist_to_selected = np.full(self.n, np.inf, dtype=np.float32)

        for _ in range(self.k):
            eligible_mask = np.ones(self.n, dtype=bool)
            if selected:
                eligible_mask = min_dist_to_selected >= threshold
                eligible_mask[np.asarray(selected, dtype=np.int64)] = False

            eligible_positions = np.flatnonzero(eligible_mask)
            if len(eligible_positions) == 0:
                break

            best_position = -1
            best_gain = -math.inf
            for position in eligible_positions.tolist():
                gain = self._facility_location_gain(covered, position)
                if gain > best_gain:
                    best_gain = gain
                    best_position = position

            if best_position < 0:
                break

            selected.append(best_position)
            covered = np.maximum(covered, self.sims[:, best_position])
            if threshold > 0.0:
                min_dist_to_selected = np.minimum(
                    min_dist_to_selected, self.dist_mat[best_position]
                )

        return selected

    def _facility_location_gain(self, covered: npt.NDArray[np.float32], position: int) -> float:
        new_similarity = self.sims[:, position]
        gain = np.maximum(new_similarity, covered) - covered
        return float(np.sum(gain, dtype=np.float64))

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
