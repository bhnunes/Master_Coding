from types import SimpleNamespace

import numpy as np

from helpers.smart_sampling.selection import compute_k, select_diverse_samples


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
