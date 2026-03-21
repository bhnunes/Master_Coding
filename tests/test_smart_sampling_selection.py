from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from helpers.smart_sampling.selection import (
    compute_k,
    select_diverse_samples,
    select_patient_samples,
)


def test_compute_k_respects_bounds() -> None:
    config = SimpleNamespace(K_MIN=20, K_MAX=80)

    assert compute_k(0, config) == 1
    assert compute_k(5, config) == 2
    assert compute_k(25, config) == 20
    assert compute_k(400, config) == 20
    assert compute_k(10000, config) == 80


def test_select_diverse_samples_keeps_all_when_target_exceeds_pool() -> None:
    config = SimpleNamespace(K_MIN=20, K_MAX=80, SEED=42)
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    global_indices = np.array([10, 20], dtype=np.int64)

    selected_indices, k_used, method = select_diverse_samples(
        embeddings,
        global_indices,
        m_target=5,
        config=config,
    )

    assert np.array_equal(selected_indices, global_indices)
    assert k_used == 2
    assert method == "keep_all"


def test_select_patient_samples_reuses_cached_embeddings_for_overlapping_subsets() -> None:
    requested_indices: list[list[int]] = []

    class FakeExtractor:
        def get_embeddings(
            self, h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            requested_indices.append(indices.tolist())
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 1.0], axis=1)

    config = SimpleNamespace(
        n_start=4,
        n_max=8,
        growth_factor=2.0,
        stability_threshold=0.9,
        stability_repeats=3,
        max_steps=1,
        intersection_ratio_threshold=0.2,
        k_min=2,
        k_max=4,
        m_max=2,
        seed=7,
    )
    patient_indices = np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.int64)

    result = select_patient_samples(
        "ignored.h5", 1, patient_indices, FakeExtractor(), cast(Any, config)
    )

    assert result.chosen_n_embed == 4
    assert len(requested_indices) == 2
    assert len(requested_indices[0]) == 4
    assert 1 <= len(requested_indices[1]) < 4
    assert set(requested_indices[0]) | set(requested_indices[1]) == set(patient_indices.tolist())


def test_select_patient_samples_computes_stability_once_when_repeats_are_deterministic(
    monkeypatch: Any,
) -> None:
    stability_calls = 0

    def fake_stability(*args: object, **kwargs: object) -> float:
        nonlocal stability_calls
        stability_calls += 1
        return 1.0

    monkeypatch.setattr(
        "helpers.smart_sampling.selection.calculate_stability_score", fake_stability
    )

    class FakeExtractor:
        def get_embeddings(
            self, h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 1.0], axis=1)

    config = SimpleNamespace(
        n_start=4,
        n_max=8,
        growth_factor=2.0,
        stability_threshold=0.5,
        stability_repeats=5,
        max_steps=2,
        intersection_ratio_threshold=0.2,
        k_min=2,
        k_max=4,
        m_max=2,
        seed=5,
    )

    _ = select_patient_samples(
        "ignored.h5",
        3,
        np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        FakeExtractor(),
        cast(Any, config),
    )

    assert stability_calls == 1
