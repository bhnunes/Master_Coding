from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from helpers.smart_sampling.selection import (
    compute_k,
    select_diverse_samples,
    select_diverse_samples_gist,
    select_patient_samples,
)

PATCH_SIDE = 4
RGB_CHANNELS = 3
DEFAULT_K_MIN = 20
DEFAULT_K_MAX = 80
SMALL_K_MIN = 2
SMALL_K_MAX = 4
FULL_RETENTION_COUNT = 2
INITIAL_EMBEDDING_COUNT = 4
FINAL_EMBEDDING_COUNT = 5
RETENTION_HISTORY_MIN_LENGTH = 2
PATIENT_PATCH_COUNT = 7
AVERAGED_STABILITY_REPEATS = 3
REDUCIBLE_POOL_COUNT = 10
PARTIAL_SELECTION_LIMIT = 5
ADAPTIVE_M_MAX = 6
FULL_SELECTION_EMBED_COUNT = 8
HELDOUT_PATCH_COUNT = 2
GIST_SELECTION_BUDGET = 3
MASK_PROTECTED_INDEX = 2
SINGLE_REDUCIBLE_SELECTION = 1
DOUBLE_REDUCIBLE_SELECTION = 2
TOTAL_SELECTED_WITH_PROTECTED = 3


def _selection_config(**overrides: Any) -> Any:
    values = {
        "n_start": 4,
        "n_max": 8,
        "growth_factor": 2.0,
        "stability_threshold": 0.9,
        "stability_repeats": 3,
        "max_steps": 1,
        "intersection_ratio_threshold": 0.2,
        "k_min": 2,
        "k_max": 4,
        "adaptive_keep_enabled": True,
        "keep_min": 2,
        "keep_step": 1,
        "keep_improvement_threshold": 0.02,
        "keep_patience": 2,
        "m_max": 2,
        "seed": 7,
        "use_gist": False,
        "protect_positive_labels": True,
        "protect_mask_positive": True,
        "positive_mask_fraction_threshold": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_patient_h5(
    path: Path,
    *,
    labels: np.ndarray[Any, np.dtype[np.uint8]],
    masks: np.ndarray[Any, np.dtype[np.uint8]],
) -> None:
    count = len(labels)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "images", data=np.zeros((count, PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS), dtype=np.uint8)
        )
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("patient_ids", data=np.ones(count, dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.asarray([f"patch_{index}.png".encode() for index in range(count)]),
        )


def test_compute_k_respects_bounds() -> None:
    config = SimpleNamespace(K_MIN=DEFAULT_K_MIN, K_MAX=DEFAULT_K_MAX)

    assert compute_k(0, config) == 1
    assert compute_k(5, config) == FULL_RETENTION_COUNT
    assert compute_k(25, config) == DEFAULT_K_MIN
    assert compute_k(400, config) == DEFAULT_K_MIN
    assert compute_k(10000, config) == DEFAULT_K_MAX


def test_select_diverse_samples_keeps_all_when_target_exceeds_pool() -> None:
    config = SimpleNamespace(
        K_MIN=DEFAULT_K_MIN,
        K_MAX=DEFAULT_K_MAX,
        SEED=42,
        ADAPTIVE_KEEP_ENABLED=True,
        KEEP_MIN=FULL_RETENTION_COUNT,
        KEEP_STEP=1,
        KEEP_IMPROVEMENT_THRESHOLD=0.02,
        KEEP_PATIENCE=2,
    )
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    global_indices = np.array([10, 20], dtype=np.int64)

    decision = select_diverse_samples(
        embeddings,
        global_indices,
        m_ceiling=5,
        config=config,
    )

    assert np.array_equal(decision.selected_indices, global_indices)
    assert decision.k_clusters == FULL_RETENTION_COUNT
    assert decision.selection_method == "keep_all"
    assert decision.retention_history == [(FULL_RETENTION_COUNT, 0.0)]


def test_select_diverse_samples_stops_when_coverage_improvement_plateaus() -> None:
    config = SimpleNamespace(
        K_MIN=SMALL_K_MIN,
        K_MAX=SMALL_K_MAX,
        SEED=42,
        ADAPTIVE_KEEP_ENABLED=True,
        KEEP_MIN=FULL_RETENTION_COUNT,
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

    decision = select_diverse_samples(
        embeddings,
        global_indices,
        m_ceiling=5,
        config=config,
        evaluation_embeddings=np.array(
            [[0.02, 0.0], [10.02, 10.0], [20.02, 20.0]], dtype=np.float32
        ),
    )

    assert decision.selection_method == "adaptive_plateau"
    assert decision.k_clusters >= FULL_RETENTION_COUNT
    assert len(decision.selected_indices) < PARTIAL_SELECTION_LIMIT
    assert len(decision.retention_history) >= RETENTION_HISTORY_MIN_LENGTH
    assert decision.plateau_evaluation_mode == "within_patient_patch_holdout"


def test_select_patient_samples_reuses_cached_embeddings_for_overlapping_subsets(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(PATIENT_PATCH_COUNT, dtype=np.uint8),
        masks=np.zeros((PATIENT_PATCH_COUNT, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )
    requested_indices: list[list[int]] = []

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            requested_indices.append(indices.tolist())
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 1.0], axis=1)

    patient_indices = np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.int64) - 10

    result = select_patient_samples(
        str(h5_path), 1, patient_indices, FakeExtractor(), _selection_config()
    )

    assert result.chosen_n_embed == FINAL_EMBEDDING_COUNT
    assert len(requested_indices) == FULL_RETENTION_COUNT
    assert len(requested_indices[0]) == INITIAL_EMBEDDING_COUNT
    assert 1 <= len(requested_indices[1]) < INITIAL_EMBEDDING_COUNT
    assert set(requested_indices[0]) | set(requested_indices[1]) == set(patient_indices.tolist())


def test_select_patient_samples_averages_stability_across_repeats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(PATIENT_PATCH_COUNT, dtype=np.uint8),
        masks=np.zeros((PATIENT_PATCH_COUNT, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )
    stability_scores = iter([0.4, 0.8, 1.0])
    stability_calls = 0

    def fake_stability(*_args: object, **_kwargs: object) -> float:
        nonlocal stability_calls
        stability_calls += 1
        return next(stability_scores)

    monkeypatch.setattr(
        "helpers.smart_sampling.selection.calculate_stability_score", fake_stability
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 1.0], axis=1)

    _ = select_patient_samples(
        str(h5_path),
        3,
        np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
        FakeExtractor(),
        _selection_config(stability_threshold=0.7, stability_repeats=3, seed=5),
    )

    assert stability_calls == AVERAGED_STABILITY_REPEATS


def test_select_patient_samples_reports_retention_history(tmp_path: Path) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(6, dtype=np.uint8),
        masks=np.zeros((6, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    result = select_patient_samples(
        str(h5_path),
        1,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        FakeExtractor(),
        _selection_config(
            n_start=6, n_max=6, m_max=6, keep_improvement_threshold=0.1, keep_patience=1, seed=11
        ),
    )

    assert result.retention_history
    assert result.selection_method in {
        "adaptive_plateau",
        "adaptive_keep_all",
        "adaptive_holdout_unavailable",
        "keep_all",
    }


def test_select_patient_samples_records_holdout_and_plateau_provenance(tmp_path: Path) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(REDUCIBLE_POOL_COUNT, dtype=np.uint8),
        masks=np.zeros((REDUCIBLE_POOL_COUNT, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.25], axis=1)

    result = select_patient_samples(
        str(h5_path),
        4,
        np.arange(10, dtype=np.int64),
        FakeExtractor(),
        _selection_config(n_start=6, n_max=10, m_max=6, keep_min=2, keep_patience=1, seed=3),
    )

    assert result.heldout_count > 0
    assert result.plateau_threshold is not None
    assert result.plateau_evaluation_mode == "within_patient_patch_holdout"
    assert result.adaptive_m_target <= ADAPTIVE_M_MAX


def test_select_patient_samples_runs_final_selection_on_full_reducible_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(REDUCIBLE_POOL_COUNT, dtype=np.uint8),
        masks=np.zeros((REDUCIBLE_POOL_COUNT, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )

    monkeypatch.setattr(
        "helpers.smart_sampling.selection.calculate_stability_score", lambda *_args, **_kwargs: 1.0
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    result = select_patient_samples(
        str(h5_path),
        9,
        np.arange(10, dtype=np.int64),
        FakeExtractor(),
        _selection_config(n_start=4, n_max=8, m_max=6, keep_min=2, seed=5),
    )

    assert result.chosen_n_embed == FULL_SELECTION_EMBED_COUNT
    assert result.heldout_count == HELDOUT_PATCH_COUNT


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

    config = SimpleNamespace(
        K_MIN=SMALL_K_MIN,
        K_MAX=SMALL_K_MAX,
        SEED=42,
        ADAPTIVE_KEEP_ENABLED=True,
        KEEP_MIN=FULL_RETENTION_COUNT,
        KEEP_STEP=1,
        KEEP_IMPROVEMENT_THRESHOLD=0.02,
        KEEP_PATIENCE=2,
    )
    decision_a = select_diverse_samples_gist(
        embeddings,
        global_indices,
        m_ceiling=GIST_SELECTION_BUDGET,
        config=config,
    )
    decision_b = select_diverse_samples_gist(
        embeddings,
        global_indices,
        m_ceiling=GIST_SELECTION_BUDGET,
        config=config,
    )

    assert np.array_equal(decision_a.selected_indices, decision_b.selected_indices)
    assert 1 <= len(decision_a.selected_indices) <= GIST_SELECTION_BUDGET
    assert set(decision_a.selected_indices.tolist()).issubset(set(global_indices.tolist()))
    assert decision_a.k_clusters == 0
    assert decision_b.k_clusters == 0
    assert decision_a.selection_method == "gist_facility_location"
    assert decision_b.selection_method == "gist_facility_location"
    assert decision_a.retention_history
    assert decision_b.retention_history == decision_a.retention_history


def test_select_patient_samples_uses_gist_when_enabled(tmp_path: Path) -> None:
    h5_path = tmp_path / "patient.h5"
    _write_patient_h5(
        h5_path,
        labels=np.zeros(5, dtype=np.uint8),
        masks=np.zeros((5, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    result = select_patient_samples(
        str(h5_path),
        1,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        FakeExtractor(),
        _selection_config(n_start=8, n_max=8, m_max=3, seed=11, use_gist=True),
    )

    assert result.selection_method == "gist_facility_location"
    assert 1 <= len(result.sampled_indices) <= GIST_SELECTION_BUDGET
    assert result.retention_history


def test_select_patient_samples_keeps_positive_label_rows_outside_reducible_budget(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "patient.h5"
    masks = np.zeros((5, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
    _write_patient_h5(
        h5_path,
        labels=np.array([1, 0, 0, 0, 0], dtype=np.uint8),
        masks=masks,
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    result = select_patient_samples(
        str(h5_path),
        1,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        FakeExtractor(),
        _selection_config(n_start=8, n_max=8, m_max=2),
    )

    assert np.array_equal(result.protected_indices, np.array([0], dtype=np.int64))
    assert result.selected_reducible_count == DOUBLE_REDUCIBLE_SELECTION
    assert result.rejected_reducible_count == DOUBLE_REDUCIBLE_SELECTION
    assert len(result.selected_indices) == TOTAL_SELECTED_WITH_PROTECTED
    assert 0 in result.selected_indices.tolist()


def test_select_patient_samples_keeps_mask_positive_rows_above_threshold(tmp_path: Path) -> None:
    h5_path = tmp_path / "patient.h5"
    masks = np.zeros((PATCH_SIDE, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
    masks[2, :2, :2] = 1
    _write_patient_h5(
        h5_path,
        labels=np.zeros(4, dtype=np.uint8),
        masks=masks,
    )

    class FakeExtractor:
        def get_embeddings(
            self, _h5_path: str, indices: np.ndarray[Any, np.dtype[np.int64]]
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
            values = indices.astype(np.float32).reshape(-1, 1)
            return np.concatenate([values, values + 0.5], axis=1)

    result = select_patient_samples(
        str(h5_path),
        1,
        np.array([0, 1, 2, 3], dtype=np.int64),
        FakeExtractor(),
        _selection_config(
            protect_positive_labels=False,
            protect_mask_positive=True,
            positive_mask_fraction_threshold=0.2,
            m_max=1,
            n_start=8,
            n_max=8,
        ),
    )

    assert np.array_equal(
        result.protected_indices,
        np.array([MASK_PROTECTED_INDEX], dtype=np.int64),
    )
    assert result.selected_reducible_count == SINGLE_REDUCIBLE_SELECTION
    assert len(result.selected_indices) == FULL_RETENTION_COUNT
