from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.gist import select_gist_facility_location

_GIST_MAX_CANDIDATE_POOL = 4096


class EmbeddingProvider(Protocol):
    def get_embeddings(
        self, h5_path: str, indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]: ...


@dataclass(frozen=True)
class PatientSelectionResult:
    patient_id: int
    selected_indices: npt.NDArray[np.int64]
    protected_indices: npt.NDArray[np.int64]
    sampled_indices: npt.NDArray[np.int64]
    chosen_n_embed: int
    k_clusters: int
    selection_method: str
    stability_history: list[tuple[int, float]]
    retention_history: list[tuple[int, float]]
    protected_count: int
    reducible_count: int
    selected_reducible_count: int
    rejected_reducible_count: int


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
) -> tuple[npt.NDArray[np.int64], int, str, list[tuple[int, float]]]:
    n_samples = len(embeddings)
    if n_samples <= m_target:
        return global_indices, n_samples, "keep_all", [(n_samples, 0.0)]

    k = compute_k(n_samples, config)
    if k <= 1:
        kept = global_indices[:m_target]
        return kept, k, "uniform", [(len(kept), 0.0)]

    clusterer = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        n_init=3,
        random_state=config.SEED,
    ).fit(embeddings)
    labels = clusterer.labels_
    if getattr(config, "ADAPTIVE_KEEP_ENABLED", False):
        return _select_diverse_samples_adaptive(
            embeddings,
            global_indices,
            labels,
            k,
            m_target,
            config,
        )

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

    return np.asarray(sorted(selected_indices), dtype=np.int64), k, "uniform", [(m_target, 0.0)]


def select_diverse_samples_gist(
    embeddings: npt.NDArray[np.float32],
    global_indices: npt.NDArray[np.int64],
    m_target: int,
    config: Any,
) -> tuple[npt.NDArray[np.int64], int, str, list[tuple[int, float]]]:
    n_samples = len(embeddings)
    if n_samples <= m_target:
        return global_indices, 0, "gist_keep_all", [(n_samples, 0.0)]

    if n_samples > _GIST_MAX_CANDIDATE_POOL:
        logging.warning(
            "GIST candidate pool size %d exceeds safe limit %d; falling back to legacy selector",
            n_samples,
            _GIST_MAX_CANDIDATE_POOL,
        )
        selected_indices, k_used, method, retention_history = select_diverse_samples(
            embeddings,
            global_indices,
            m_target,
            config,
        )
        return selected_indices, k_used, f"{method}_gist_fallback", retention_history

    gist_result = select_gist_facility_location(embeddings, max_selected=m_target)
    selected_indices = global_indices[gist_result.selected_positions]
    return (
        np.asarray(sorted(selected_indices.tolist()), dtype=np.int64),
        0,
        "gist_facility_location",
        gist_result.objective_trace,
    )


def select_patient_samples(
    h5_path: str,
    patient_id: int,
    patient_indices: npt.NDArray[np.int64],
    extractor: EmbeddingProvider,
    config: SmartSamplerConfig,
) -> PatientSelectionResult:
    patient_indices = np.asarray(patient_indices, dtype=np.int64)
    protected_mask = _load_protected_row_mask(h5_path, patient_indices, config)
    protected_indices = patient_indices[protected_mask]
    reducible_indices = patient_indices[~protected_mask]
    reducible_count = len(reducible_indices)

    if reducible_count == 0:
        return PatientSelectionResult(
            patient_id=patient_id,
            selected_indices=np.asarray(sorted(protected_indices.tolist()), dtype=np.int64),
            protected_indices=np.asarray(sorted(protected_indices.tolist()), dtype=np.int64),
            sampled_indices=np.empty(0, dtype=np.int64),
            chosen_n_embed=0,
            k_clusters=0,
            selection_method="protected_only",
            stability_history=[],
            retention_history=[],
            protected_count=len(protected_indices),
            reducible_count=0,
            selected_reducible_count=0,
            rejected_reducible_count=0,
        )

    n_curr = min(config.n_start, reducible_count)
    stability_history: list[tuple[int, float]] = []
    final_embeddings: npt.NDArray[np.float32] | None = None
    candidate_pool: npt.NDArray[np.int64] = reducible_indices
    embedding_cache: dict[int, npt.NDArray[np.float32]] = {}

    rng = np.random.default_rng(config.seed + patient_id)

    def fetch_embeddings(indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        ordered_indices = np.asarray(indices, dtype=np.int64)
        missing = np.asarray(
            [index for index in ordered_indices.tolist() if int(index) not in embedding_cache],
            dtype=np.int64,
        )
        if len(missing) > 0:
            missing_embeddings = extractor.get_embeddings(h5_path, missing)
            for index, embedding in zip(missing.tolist(), missing_embeddings, strict=False):
                embedding_cache[int(index)] = np.asarray(embedding, dtype=np.float32)
        return np.stack([embedding_cache[int(index)] for index in ordered_indices.tolist()]).astype(
            np.float32,
            copy=False,
        )

    if reducible_count <= int(config.n_start * 1.5):
        final_embeddings = fetch_embeddings(reducible_indices)
        candidate_pool = reducible_indices
        stability_history.append((reducible_count, 1.0))
    else:
        step = 0
        while step < config.max_steps:
            if n_curr >= reducible_count:
                candidate_pool = reducible_indices
                final_embeddings = fetch_embeddings(candidate_pool)
                break

            idx_s1 = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
            idx_s2 = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
            emb_s1 = fetch_embeddings(idx_s1)
            emb_s2 = fetch_embeddings(idx_s2)
            avg_score = float(
                calculate_stability_score(
                    emb_s1, emb_s2, idx_s1, idx_s2, _selection_namespace(config)
                )
            )
            stability_history.append((n_curr, avg_score))

            if avg_score >= config.stability_threshold:
                final_embeddings = emb_s1
                candidate_pool = idx_s1
                break

            new_n = int(n_curr * config.growth_factor)
            if new_n >= config.n_max or new_n >= reducible_count:
                n_curr = min(reducible_count, config.n_max)
                candidate_pool = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
                final_embeddings = fetch_embeddings(candidate_pool)
                break
            n_curr = new_n
            step += 1

    if final_embeddings is None:
        n_curr = min(reducible_count, config.n_start)
        candidate_pool = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
        final_embeddings = fetch_embeddings(candidate_pool)

    m_target = min(config.m_max, reducible_count)
    if getattr(config, "use_gist", False):
        sampled_indices, k_used, method, retention_history = select_diverse_samples_gist(
            final_embeddings,
            candidate_pool,
            m_target,
            _selection_namespace(config),
        )
    else:
        sampled_indices, k_used, method, retention_history = select_diverse_samples(
            final_embeddings,
            candidate_pool,
            m_target,
            _selection_namespace(config),
        )

    selected_indices = np.asarray(
        sorted(np.concatenate([protected_indices, sampled_indices]).tolist()),
        dtype=np.int64,
    )
    sampled_indices = np.asarray(sorted(sampled_indices.tolist()), dtype=np.int64)
    protected_indices = np.asarray(sorted(protected_indices.tolist()), dtype=np.int64)
    return PatientSelectionResult(
        patient_id=patient_id,
        selected_indices=selected_indices,
        protected_indices=protected_indices,
        sampled_indices=sampled_indices,
        chosen_n_embed=len(final_embeddings),
        k_clusters=k_used,
        selection_method=method,
        stability_history=stability_history,
        retention_history=retention_history,
        protected_count=len(protected_indices),
        reducible_count=reducible_count,
        selected_reducible_count=len(sampled_indices),
        rejected_reducible_count=reducible_count - len(sampled_indices),
    )


def _load_protected_row_mask(
    h5_path: str,
    patient_indices: npt.NDArray[np.int64],
    config: SmartSamplerConfig,
) -> npt.NDArray[np.bool_]:
    protected_mask = np.zeros(len(patient_indices), dtype=bool)
    if len(patient_indices) == 0:
        return protected_mask

    with h5py.File(h5_path, "r") as handle:
        if getattr(config, "protect_positive_labels", True):
            labels = np.asarray(handle["labels"][patient_indices], dtype=np.int64)
            protected_mask |= labels > 0
        if getattr(config, "protect_mask_positive", True):
            masks = np.asarray(handle["masks"][patient_indices])
            mask_axes = tuple(range(1, masks.ndim))
            positive_fraction = np.mean(masks > 0, axis=mask_axes)
            protected_mask |= positive_fraction > getattr(
                config, "positive_mask_fraction_threshold", 0.0
            )
    return np.asarray(protected_mask, dtype=bool)


def _select_diverse_samples_adaptive(
    embeddings: npt.NDArray[np.float32],
    global_indices: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    k: int,
    m_target: int,
    config: Any,
) -> tuple[npt.NDArray[np.int64], int, str, list[tuple[int, float]]]:
    max_keep = min(m_target, len(global_indices))
    min_keep = min(max_keep, getattr(config, "KEEP_MIN", max_keep))
    keep_step = max(1, getattr(config, "KEEP_STEP", 1))
    keep_patience = max(1, getattr(config, "KEEP_PATIENCE", 1))
    improvement_threshold = float(getattr(config, "KEEP_IMPROVEMENT_THRESHOLD", 0.0))
    selection_order = _build_cluster_balanced_order(global_indices, labels, seed=config.SEED)

    if min_keep >= max_keep:
        return selection_order[:max_keep], k, "adaptive_keep_all", [(max_keep, 0.0)]

    retention_history: list[tuple[int, float]] = []
    previous_score: float | None = None
    previous_count: int | None = None
    plateau_steps = 0
    selected_count = min_keep
    best_count = max_keep

    while selected_count <= max_keep:
        score = _coverage_score(embeddings, selection_order[:selected_count], global_indices)
        retention_history.append((selected_count, score))
        if previous_score is not None:
            relative_improvement = (previous_score - score) / max(previous_score, 1e-12)
            if relative_improvement < improvement_threshold:
                plateau_steps += 1
            else:
                plateau_steps = 0
            if plateau_steps >= keep_patience:
                best_count = previous_count if previous_count is not None else selected_count
                break
        previous_score = score
        previous_count = selected_count
        if selected_count == max_keep:
            best_count = max_keep
            break
        selected_count = min(max_keep, selected_count + keep_step)

    return selection_order[:best_count], k, "adaptive_plateau", retention_history


def _build_cluster_balanced_order(
    global_indices: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    *,
    seed: int,
) -> npt.NDArray[np.int64]:
    dataframe = pd.DataFrame({"idx": global_indices, "label": labels})
    rng = np.random.default_rng(seed)
    grouped_indices: list[list[int]] = []
    for _, group in dataframe.groupby("label", sort=True):
        values = group["idx"].tolist()
        shuffled = np.asarray(values, dtype=np.int64)
        rng.shuffle(shuffled)
        grouped_indices.append(shuffled.tolist())

    ordered: list[int] = []
    while grouped_indices:
        next_groups: list[list[int]] = []
        for group in grouped_indices:
            if not group:
                continue
            ordered.append(int(group.pop(0)))
            if group:
                next_groups.append(group)
        grouped_indices = next_groups
    return np.asarray(ordered, dtype=np.int64)


def _coverage_score(
    embeddings: npt.NDArray[np.float32],
    selected_indices: npt.NDArray[np.int64],
    global_indices: npt.NDArray[np.int64],
) -> float:
    selection_mask = np.isin(global_indices, selected_indices)
    selected_embeddings = embeddings[selection_mask]
    if len(selected_embeddings) == 0:
        return float("inf")
    distances = np.linalg.norm(
        embeddings[:, None, :] - selected_embeddings[None, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1)))


def _selection_namespace(config: SmartSamplerConfig) -> Any:
    class _Namespace:
        K_MIN = config.k_min
        K_MAX = config.k_max
        SEED = config.seed
        INTERSECTION_RATIO_THRESHOLD = config.intersection_ratio_threshold
        ADAPTIVE_KEEP_ENABLED = config.adaptive_keep_enabled
        KEEP_MIN = config.keep_min
        KEEP_STEP = config.keep_step
        KEEP_IMPROVEMENT_THRESHOLD = config.keep_improvement_threshold
        KEEP_PATIENCE = config.keep_patience

    return _Namespace()
