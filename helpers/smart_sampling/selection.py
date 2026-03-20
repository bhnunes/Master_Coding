from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from helpers.smart_sampling.config import SmartSamplerConfig


class EmbeddingProvider(Protocol):
    def get_embeddings(
        self, h5_path: str, indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]: ...


@dataclass(frozen=True)
class PatientSelectionResult:
    patient_id: int
    selected_indices: npt.NDArray[np.int64]
    chosen_n_embed: int
    k_clusters: int
    selection_method: str
    stability_history: list[tuple[int, float]]


def compute_k(n_samples: int, config: Any) -> int:
    if n_samples == 0:
        return 1
    raw_k = int(float(np.sqrt(n_samples)))
    k = max(config.K_MIN, min(raw_k, config.K_MAX))
    if n_samples < config.K_MIN:
        k = max(2, int(n_samples / 10))
        if k < 2:
            k = n_samples
    return int(k)


def calculate_stability_score(
    emb_s1: npt.NDArray[np.float32],
    emb_s2: npt.NDArray[np.float32],
    idx_s1: npt.NDArray[np.int64],
    idx_s2: npt.NDArray[np.int64],
    config: Any,
) -> float:
    n_samples = len(emb_s1)
    k = compute_k(n_samples, config)
    if n_samples <= 1 or k <= 1:
        return 1.0

    mbk1 = MiniBatchKMeans(
        n_clusters=k,
        batch_size=512,
        n_init=3,
        random_state=config.SEED,
        max_iter=100,
    ).fit(emb_s1)
    mbk2 = MiniBatchKMeans(
        n_clusters=k,
        batch_size=512,
        n_init=3,
        random_state=config.SEED + 1,
        max_iter=100,
    ).fit(emb_s2)

    intersect_mask_1 = np.isin(idx_s1, idx_s2)
    intersection_size = int(np.sum(intersect_mask_1))
    if intersection_size > (config.INTERSECTION_RATIO_THRESHOLD * n_samples):
        emb_intersection = emb_s1[intersect_mask_1]
        labels1 = mbk1.predict(emb_intersection)
        labels2 = mbk2.predict(emb_intersection)
        return float(adjusted_rand_score(labels1, labels2))

    labels1 = mbk1.labels_
    labels2 = mbk2.predict(emb_s1)
    return float(adjusted_rand_score(labels1, labels2))


def select_diverse_samples(
    embeddings: npt.NDArray[np.float32],
    global_indices: npt.NDArray[np.int64],
    m_target: int,
    config: Any,
) -> tuple[npt.NDArray[np.int64], int, str]:
    n_samples = len(embeddings)
    if n_samples <= m_target:
        return global_indices, n_samples, "keep_all"

    k = compute_k(n_samples, config)
    if k <= 1:
        return global_indices[:m_target], k, "uniform"

    clusterer = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        n_init=3,
        random_state=config.SEED,
    ).fit(embeddings)
    labels = clusterer.labels_
    dataframe = pd.DataFrame({"idx": global_indices, "label": labels})
    quota = int(np.ceil(m_target / k))

    selected_indices: list[int] = []
    overflow_pool: list[int] = []
    for _, group in dataframe.groupby("label"):
        if len(group) <= quota:
            selected_indices.extend(group["idx"].tolist())
        else:
            selected = group.sample(n=quota, random_state=config.SEED)
            selected_indices.extend(selected["idx"].tolist())
            overflow_pool.extend(group.loc[~group.index.isin(selected.index), "idx"].tolist())

    if len(selected_indices) < m_target and overflow_pool:
        needed = m_target - len(selected_indices)
        selected_indices.extend(sorted(overflow_pool)[:needed])

    return np.asarray(sorted(selected_indices), dtype=np.int64), k, "uniform"


def select_patient_samples(
    h5_path: str,
    patient_id: int,
    patient_indices: npt.NDArray[np.int64],
    extractor: EmbeddingProvider,
    config: SmartSamplerConfig,
) -> PatientSelectionResult:
    patient_indices = np.asarray(patient_indices, dtype=np.int64)
    patch_count = len(patient_indices)
    n_curr = min(config.n_start, patch_count)
    stability_history: list[tuple[int, float]] = []
    final_embeddings: npt.NDArray[np.float32] | None = None
    candidate_pool: npt.NDArray[np.int64] = patient_indices

    rng = np.random.default_rng(config.seed + patient_id)

    if patch_count <= int(config.n_start * 1.5):
        final_embeddings = extractor.get_embeddings(h5_path, patient_indices)
        candidate_pool = patient_indices
        stability_history.append((patch_count, 1.0))
    else:
        step = 0
        while step < config.max_steps:
            if n_curr >= patch_count:
                candidate_pool = patient_indices
                final_embeddings = extractor.get_embeddings(h5_path, candidate_pool)
                break

            idx_s1 = np.sort(rng.choice(patient_indices, n_curr, replace=False))
            idx_s2 = np.sort(rng.choice(patient_indices, n_curr, replace=False))
            emb_s1 = extractor.get_embeddings(h5_path, idx_s1)
            emb_s2 = extractor.get_embeddings(h5_path, idx_s2)
            scores = [
                calculate_stability_score(
                    emb_s1, emb_s2, idx_s1, idx_s2, _selection_namespace(config)
                )
                for _ in range(config.stability_repeats)
            ]
            avg_score = float(np.median(scores))
            stability_history.append((n_curr, avg_score))

            if avg_score >= config.stability_threshold:
                final_embeddings = emb_s1
                candidate_pool = idx_s1
                break

            new_n = int(n_curr * config.growth_factor)
            if new_n >= config.n_max or new_n >= patch_count:
                n_curr = min(patch_count, config.n_max)
                candidate_pool = np.sort(rng.choice(patient_indices, n_curr, replace=False))
                final_embeddings = extractor.get_embeddings(h5_path, candidate_pool)
                break
            n_curr = new_n
            step += 1

    if final_embeddings is None:
        n_curr = min(patch_count, config.n_start)
        candidate_pool = np.sort(rng.choice(patient_indices, n_curr, replace=False))
        final_embeddings = extractor.get_embeddings(h5_path, candidate_pool)

    selected_indices, k_used, method = select_diverse_samples(
        final_embeddings,
        candidate_pool,
        min(config.m_max, patch_count),
        _selection_namespace(config),
    )

    return PatientSelectionResult(
        patient_id=patient_id,
        selected_indices=selected_indices,
        chosen_n_embed=len(final_embeddings),
        k_clusters=k_used,
        selection_method=method,
        stability_history=stability_history,
    )


def _selection_namespace(config: SmartSamplerConfig) -> Any:
    class _Namespace:
        K_MIN = config.k_min
        K_MAX = config.k_max
        SEED = config.seed
        INTERSECTION_RATIO_THRESHOLD = config.intersection_ratio_threshold

    return _Namespace()
