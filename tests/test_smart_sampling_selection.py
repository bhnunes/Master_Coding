from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from helpers.smart_sampling.selection import (
    compute_k,
    select_diverse_samples,
    select_diverse_samples_gist,
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
    config = SimpleNamespace(
        K_MIN=20,
        K_MAX=80,
        SEED=42,
        ADAPTIVE_KEEP_ENABLED=True,
        KEEP_MIN=2,
        KEEP_STEP=1,
        KEEP_IMPROVEMENT_THRESHOLD=0.02,
        KEEP_PATIENCE=2,
    )
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    global_indices = np.array([10, 20], dtype=np.int64)

    selected_indices, k_used, method, retention_history = select_diverse_samples(
        embeddings,
        global_indices,
        m_target=5,
        config=config,
    )

    assert np.array_equal(selected_indices, global_indices)
    assert k_used == 2
    assert method == "keep_all"
    assert retention_history == [(2, 0.0)]


def test_select_diverse_samples_stops_when_coverage_improvement_plateaus() -> None:
    config = SimpleNamespace(
        K_MIN=2,
        K_MAX=4,
        SEED=42,
        ADAPTIVE_KEEP_ENABLED=True,
        KEEP_MIN=2,
        KEEP_STEP=1,
        KEEP_IMPROVEMENT_THRESHOLD=0.3,
        KEEP_PATIENCE=1,
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [10.0, 10.0],
            [10.05, 10.0],
            [20.0, 20.0],
            [20.05, 20.0],
        ],
        dtype=np.float32,
    )
    global_indices = np.arange(len(embeddings), dtype=np.int64)

    selected_indices, k_used, method, retention_history = select_diverse_samples(
        embeddings,
        global_indices,
        m_target=5,
        config=config,
    )

    assert method == "adaptive_plateau"
    assert k_used >= 2
    assert len(selected_indices) < 5
    assert len(retention_history) >= 2


def test_select_patient_samples_reuses_cached_embeddings_for_overlapping_subsets() -> None:
    requested_indices: list[list[int]] = []

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
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
        adaptive_keep_enabled=True,
        keep_min=2,
        keep_step=1,
        keep_improvement_threshold=0.02,
        keep_patience=2,
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

    def fake_stability(*_args: object, **_kwargs: object) -> float:
        nonlocal stability_calls
        stability_calls += 1
        return 1.0

    monkeypatch.setattr(
        "helpers.smart_sampling.selection.calculate_stability_score", fake_stability
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
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
        adaptive_keep_enabled=True,
        keep_min=2,
        keep_step=1,
        keep_improvement_threshold=0.02,
        keep_patience=2,
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


def test_select_patient_samples_reports_retention_history() -> None:
    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    config = SimpleNamespace(
        n_start=6,
        n_max=6,
        growth_factor=2.0,
        stability_threshold=0.5,
        stability_repeats=2,
        max_steps=1,
        intersection_ratio_threshold=0.2,
        k_min=2,
        k_max=4,
        adaptive_keep_enabled=True,
        keep_min=2,
        keep_step=1,
        keep_improvement_threshold=0.1,
        keep_patience=1,
        m_max=6,
        seed=11,
    )

    result = select_patient_samples(
        "ignored.h5",
        1,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        FakeExtractor(),
        cast(Any, config),
    )

    assert result.retention_history
    assert result.selection_method in {"adaptive_plateau", "adaptive_keep_all", "keep_all"}


def test_select_diverse_samples_gist_is_deterministic_and_respects_budget() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    global_indices = np.array([10, 11, 12, 13, 14], dtype=np.int64)

    selected_a, k_a, method_a, history_a = select_diverse_samples_gist(
        embeddings,
        global_indices,
        m_target=3,
        config=SimpleNamespace(
            K_MIN=2,
            K_MAX=4,
            SEED=42,
            ADAPTIVE_KEEP_ENABLED=True,
            KEEP_MIN=2,
            KEEP_STEP=1,
            KEEP_IMPROVEMENT_THRESHOLD=0.02,
            KEEP_PATIENCE=2,
        ),
    )
    selected_b, k_b, method_b, history_b = select_diverse_samples_gist(
        embeddings,
        global_indices,
        m_target=3,
        config=SimpleNamespace(
            K_MIN=2,
            K_MAX=4,
            SEED=42,
            ADAPTIVE_KEEP_ENABLED=True,
            KEEP_MIN=2,
            KEEP_STEP=1,
            KEEP_IMPROVEMENT_THRESHOLD=0.02,
            KEEP_PATIENCE=2,
        ),
    )

    assert np.array_equal(selected_a, selected_b)
    assert 1 <= len(selected_a) <= 3
    assert set(selected_a.tolist()).issubset(set(global_indices.tolist()))
    assert k_a == 0
    assert k_b == 0
    assert method_a == "gist_facility_location"
    assert method_b == "gist_facility_location"
    assert history_a
    assert history_b == history_a


def test_select_patient_samples_uses_gist_when_enabled() -> None:
    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    config = SimpleNamespace(
        n_start=8,
        n_max=8,
        growth_factor=2.0,
        stability_threshold=0.5,
        stability_repeats=2,
        max_steps=1,
        intersection_ratio_threshold=0.2,
        k_min=2,
        k_max=4,
        adaptive_keep_enabled=True,
        keep_min=2,
        keep_step=1,
        keep_improvement_threshold=0.1,
        keep_patience=1,
        m_max=3,
        seed=11,
        use_gist=True,
    )

    result = select_patient_samples(
        "ignored.h5",
        1,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        FakeExtractor(),
        cast(Any, config),
    )

    assert result.selection_method == "gist_facility_location"
    assert 1 <= len(result.selected_indices) <= 3
    assert result.retention_history
