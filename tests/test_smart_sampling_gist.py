from __future__ import annotations

import numpy as np

from helpers.smart_sampling.gist import (
    _pairwise_distance_matrix,
    select_gist_facility_location,
)

EXPECTED_UNIT_DISTANCE = 1.0
EXPECTED_DIAGONAL_DISTANCE = 0.0
SELECTED_LIMIT = 4


def test_pairwise_distance_matrix_matches_euclidean_distances() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    distances = _pairwise_distance_matrix(points)

    assert distances.dtype == np.float32
    assert distances[0, 0] == EXPECTED_DIAGONAL_DISTANCE
    assert distances[0, 1] == EXPECTED_UNIT_DISTANCE
    assert distances[0, 2] == EXPECTED_UNIT_DISTANCE
    assert np.allclose(distances, distances.T)


def test_select_gist_facility_location_respects_budget() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [4.0, 4.0],
            [4.2, 4.0],
            [8.0, 8.0],
            [8.2, 8.0],
            [12.0, 12.0],
        ],
        dtype=np.float32,
    )

    result = select_gist_facility_location(embeddings, max_selected=SELECTED_LIMIT)

    assert 1 <= len(result.selected_positions) <= SELECTED_LIMIT
    assert set(result.selected_positions.tolist()).issubset(set(range(len(embeddings))))
    assert result.objective_trace
