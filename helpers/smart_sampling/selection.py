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
MIN_REDUCED_SAMPLE_COUNT = 2


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
    protected_positive_label_count: int
    protected_mask_positive_count: int
    reducible_count: int
    total_positive_label_count: int
    total_negative_label_count: int
    selected_positive_label_count: int
    selected_negative_label_count: int
    adaptive_m_target: int
    heldout_count: int
    plateau_threshold: float | None
    plateau_trigger_improvement: float | None
    plateau_trigger_keep_count: int | None
    plateau_trigger_step: int | None
    plateau_stop_reason: str | None
    plateau_evaluation_mode: str | None
    selected_reducible_count: int
    rejected_reducible_count: int


@dataclass(frozen=True)
class SelectionDecision:
    selected_indices: npt.NDArray[np.int64]
    k_clusters: int
    adaptive_m_target: int
    selection_method: str
    retention_history: list[tuple[int, float]]
    heldout_count: int
    plateau_threshold: float | None
    plateau_trigger_improvement: float | None
    plateau_trigger_keep_count: int | None
    plateau_trigger_step: int | None
    plateau_stop_reason: str | None
    plateau_evaluation_mode: str | None


@dataclass(frozen=True)
class PatientProtectionMetadata:
    protected_mask: npt.NDArray[np.bool_]
    labels: npt.NDArray[np.int64]
    mask_positive_mask: npt.NDArray[np.bool_]


@dataclass
class _EmbeddingCache:
    h5_path: str
    extractor: EmbeddingProvider
    cache: dict[int, npt.NDArray[np.float32]]

    def fetch(self, indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        ordered_indices = np.asarray(indices, dtype=np.int64)
        missing = np.asarray(
            [index for index in ordered_indices.tolist() if int(index) not in self.cache],
            dtype=np.int64,
        )
        if len(missing) > 0:
            missing_embeddings = self.extractor.get_embeddings(self.h5_path, missing)
            for index, embedding in zip(missing.tolist(), missing_embeddings, strict=False):
                self.cache[int(index)] = np.asarray(embedding, dtype=np.float32)
        return np.stack([self.cache[int(index)] for index in ordered_indices.tolist()]).astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class _PatientSelectionCounts:
    protected_positive_label_count: int
    protected_mask_positive_count: int
    reducible_count: int
    total_positive_label_count: int
    total_negative_label_count: int
    selected_positive_label_count: int
    selected_negative_label_count: int


@dataclass(frozen=True)
class _PatientSelectionBuildRequest:
    patient_id: int
    selected_indices: npt.NDArray[np.int64]
    protected_indices: npt.NDArray[np.int64]
    sampled_indices: npt.NDArray[np.int64]
    chosen_n_embed: int


def compute_k(n_samples: int, config: Any) -> int:
    if n_samples == 0:
        return 1
    raw_k = int(float(np.sqrt(n_samples)))
    k = max(config.K_MIN, min(raw_k, config.K_MAX))
    if n_samples < config.K_MIN:
        k = max(MIN_REDUCED_SAMPLE_COUNT, int(n_samples / 10))
        if k < MIN_REDUCED_SAMPLE_COUNT:
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
    m_ceiling: int,
    config: Any,
    *,
    evaluation_embeddings: npt.NDArray[np.float32] | None = None,
) -> SelectionDecision:
    n_samples = len(embeddings)
    if n_samples <= m_ceiling:
        return SelectionDecision(
            selected_indices=global_indices,
            k_clusters=n_samples,
            adaptive_m_target=n_samples,
            selection_method="keep_all",
            retention_history=[(n_samples, 0.0)],
            heldout_count=0 if evaluation_embeddings is None else len(evaluation_embeddings),
            plateau_threshold=None,
            plateau_trigger_improvement=None,
            plateau_trigger_keep_count=None,
            plateau_trigger_step=None,
            plateau_stop_reason="keep_all",
            plateau_evaluation_mode=None,
        )

    k = compute_k(n_samples, config)
    adaptive_m_target = _compute_adaptive_m_target(
        reducible_count=n_samples,
        m_ceiling=m_ceiling,
        k_used=k,
        config=config,
    )
    if k <= 1:
        kept = global_indices[:adaptive_m_target]
        return SelectionDecision(
            selected_indices=kept,
            k_clusters=k,
            adaptive_m_target=len(kept),
            selection_method="uniform",
            retention_history=[(len(kept), 0.0)],
            heldout_count=0 if evaluation_embeddings is None else len(evaluation_embeddings),
            plateau_threshold=None,
            plateau_trigger_improvement=None,
            plateau_trigger_keep_count=None,
            plateau_trigger_step=None,
            plateau_stop_reason="uniform_single_cluster",
            plateau_evaluation_mode=None,
        )

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
            adaptive_m_target,
            config,
            evaluation_embeddings=evaluation_embeddings,
        )

    dataframe = pd.DataFrame({"idx": global_indices, "label": labels})
    quota = int(np.ceil(adaptive_m_target / k))

    selected_indices: list[int] = []
    overflow_pool: list[int] = []
    for _, group in dataframe.groupby("label"):
        if len(group) <= quota:
            selected_indices.extend(group["idx"].tolist())
        else:
            selected = group.sample(n=quota, random_state=config.SEED)
            selected_indices.extend(selected["idx"].tolist())
            overflow_pool.extend(group.loc[~group.index.isin(selected.index), "idx"].tolist())

    if len(selected_indices) < adaptive_m_target and overflow_pool:
        needed = adaptive_m_target - len(selected_indices)
        selected_indices.extend(sorted(overflow_pool)[:needed])

    return SelectionDecision(
        selected_indices=np.asarray(sorted(selected_indices), dtype=np.int64),
        k_clusters=k,
        adaptive_m_target=adaptive_m_target,
        selection_method="uniform",
        retention_history=[(adaptive_m_target, 0.0)],
        heldout_count=0 if evaluation_embeddings is None else len(evaluation_embeddings),
        plateau_threshold=None,
        plateau_trigger_improvement=None,
        plateau_trigger_keep_count=None,
        plateau_trigger_step=None,
        plateau_stop_reason="uniform_budget",
        plateau_evaluation_mode=None,
    )


def select_diverse_samples_gist(
    embeddings: npt.NDArray[np.float32],
    global_indices: npt.NDArray[np.int64],
    m_ceiling: int,
    config: Any,
    *,
    evaluation_embeddings: npt.NDArray[np.float32] | None = None,
) -> SelectionDecision:
    n_samples = len(embeddings)
    adaptive_m_target = _compute_adaptive_m_target(
        reducible_count=n_samples,
        m_ceiling=m_ceiling,
        k_used=compute_k(n_samples, config),
        config=config,
    )
    if n_samples <= adaptive_m_target:
        return SelectionDecision(
            selected_indices=global_indices,
            k_clusters=0,
            adaptive_m_target=n_samples,
            selection_method="gist_keep_all",
            retention_history=[(n_samples, 0.0)],
            heldout_count=0 if evaluation_embeddings is None else len(evaluation_embeddings),
            plateau_threshold=None,
            plateau_trigger_improvement=None,
            plateau_trigger_keep_count=None,
            plateau_trigger_step=None,
            plateau_stop_reason="keep_all",
            plateau_evaluation_mode=None,
        )

    if n_samples > _GIST_MAX_CANDIDATE_POOL:
        logging.warning(
            "GIST candidate pool size %d exceeds safe limit %d; falling back to legacy selector",
            n_samples,
            _GIST_MAX_CANDIDATE_POOL,
        )
        fallback_decision = select_diverse_samples(
            embeddings,
            global_indices,
            adaptive_m_target,
            config,
            evaluation_embeddings=evaluation_embeddings,
        )
        return SelectionDecision(
            selected_indices=fallback_decision.selected_indices,
            k_clusters=fallback_decision.k_clusters,
            adaptive_m_target=fallback_decision.adaptive_m_target,
            selection_method=f"{fallback_decision.selection_method}_gist_fallback",
            retention_history=fallback_decision.retention_history,
            heldout_count=fallback_decision.heldout_count,
            plateau_threshold=fallback_decision.plateau_threshold,
            plateau_trigger_improvement=fallback_decision.plateau_trigger_improvement,
            plateau_trigger_keep_count=fallback_decision.plateau_trigger_keep_count,
            plateau_trigger_step=fallback_decision.plateau_trigger_step,
            plateau_stop_reason=fallback_decision.plateau_stop_reason,
            plateau_evaluation_mode=fallback_decision.plateau_evaluation_mode,
        )

    gist_result = select_gist_facility_location(embeddings, max_selected=adaptive_m_target)
    selected_indices = global_indices[gist_result.selected_positions]
    return SelectionDecision(
        selected_indices=np.asarray(sorted(selected_indices.tolist()), dtype=np.int64),
        k_clusters=0,
        adaptive_m_target=adaptive_m_target,
        selection_method="gist_facility_location",
        retention_history=gist_result.objective_trace,
        heldout_count=0 if evaluation_embeddings is None else len(evaluation_embeddings),
        plateau_threshold=None,
        plateau_trigger_improvement=None,
        plateau_trigger_keep_count=None,
        plateau_trigger_step=None,
        plateau_stop_reason="gist_objective",
        plateau_evaluation_mode=None,
    )


def select_patient_samples(
    h5_path: str,
    patient_id: int,
    patient_indices: npt.NDArray[np.int64],
    extractor: EmbeddingProvider,
    config: SmartSamplerConfig,
) -> PatientSelectionResult:
    patient_indices = np.asarray(patient_indices, dtype=np.int64)
    protection_metadata = _load_patient_protection_metadata(h5_path, patient_indices, config)
    protected_mask = protection_metadata.protected_mask
    patient_labels = protection_metadata.labels
    mask_positive_mask = protection_metadata.mask_positive_mask
    protected_indices = patient_indices[protected_mask]
    reducible_indices = patient_indices[~protected_mask]
    reducible_count = len(reducible_indices)
    total_positive_label_count = int(np.sum(patient_labels > 0))
    total_negative_label_count = int(len(patient_labels) - total_positive_label_count)
    protected_positive_label_count = int(np.sum(patient_labels[protected_mask] > 0))
    protected_mask_positive_count = int(np.sum(mask_positive_mask[protected_mask]))

    if reducible_count == 0:
        return _protected_only_result(
            patient_id=patient_id,
            protected_indices=protected_indices,
            protected_positive_label_count=protected_positive_label_count,
            protected_mask_positive_count=protected_mask_positive_count,
            total_positive_label_count=total_positive_label_count,
            total_negative_label_count=total_negative_label_count,
        )

    rng = np.random.default_rng(config.seed + patient_id)
    embedding_cache = _EmbeddingCache(h5_path=h5_path, extractor=extractor, cache={})
    stability_history, _ = _resolve_stability_subset_embeddings(
        reducible_indices=reducible_indices,
        config=config,
        rng=rng,
        embedding_cache=embedding_cache,
    )
    selection_decision, sampled_indices, selected_indices, chosen_n_embed = _run_patient_selection(
        reducible_indices=reducible_indices,
        protected_indices=protected_indices,
        config=config,
        rng=rng,
        embedding_cache=embedding_cache,
    )
    selected_mask = np.isin(patient_indices, selected_indices)
    selected_positive_label_count = int(np.sum(patient_labels[selected_mask] > 0))
    selected_negative_label_count = int(np.sum(patient_labels[selected_mask] <= 0))
    return _build_patient_selection_result(
        request=_PatientSelectionBuildRequest(
            patient_id=patient_id,
            selected_indices=selected_indices,
            protected_indices=protected_indices,
            sampled_indices=sampled_indices,
            chosen_n_embed=chosen_n_embed,
        ),
        selection_decision=selection_decision,
        stability_history=stability_history,
        counts=_PatientSelectionCounts(
            protected_positive_label_count=protected_positive_label_count,
            protected_mask_positive_count=protected_mask_positive_count,
            reducible_count=reducible_count,
            total_positive_label_count=total_positive_label_count,
            total_negative_label_count=total_negative_label_count,
            selected_positive_label_count=selected_positive_label_count,
            selected_negative_label_count=selected_negative_label_count,
        ),
    )


def _protected_only_result(
    *,
    patient_id: int,
    protected_indices: npt.NDArray[np.int64],
    protected_positive_label_count: int,
    protected_mask_positive_count: int,
    total_positive_label_count: int,
    total_negative_label_count: int,
) -> PatientSelectionResult:
    sorted_protected = np.asarray(sorted(protected_indices.tolist()), dtype=np.int64)
    return PatientSelectionResult(
        patient_id=patient_id,
        selected_indices=sorted_protected,
        protected_indices=sorted_protected,
        sampled_indices=np.empty(0, dtype=np.int64),
        chosen_n_embed=0,
        k_clusters=0,
        selection_method="protected_only",
        stability_history=[],
        retention_history=[],
        protected_count=len(sorted_protected),
        protected_positive_label_count=protected_positive_label_count,
        protected_mask_positive_count=protected_mask_positive_count,
        reducible_count=0,
        total_positive_label_count=total_positive_label_count,
        total_negative_label_count=total_negative_label_count,
        selected_positive_label_count=total_positive_label_count,
        selected_negative_label_count=total_negative_label_count,
        adaptive_m_target=0,
        heldout_count=0,
        plateau_threshold=None,
        plateau_trigger_improvement=None,
        plateau_trigger_keep_count=None,
        plateau_trigger_step=None,
        plateau_stop_reason=None,
        plateau_evaluation_mode=None,
        selected_reducible_count=0,
        rejected_reducible_count=0,
    )


def _resolve_stability_subset_embeddings(
    *,
    reducible_indices: npt.NDArray[np.int64],
    config: SmartSamplerConfig,
    rng: np.random.Generator,
    embedding_cache: _EmbeddingCache,
) -> tuple[list[tuple[int, float]], npt.NDArray[np.float32]]:
    reducible_count = len(reducible_indices)
    if reducible_count <= int(config.n_start * 1.5):
        return [(reducible_count, 1.0)], embedding_cache.fetch(reducible_indices)

    n_curr = min(config.n_start, reducible_count)
    stability_history: list[tuple[int, float]] = []
    step = 0
    while step < config.max_steps:
        if n_curr >= reducible_count:
            return stability_history, embedding_cache.fetch(reducible_indices)

        avg_score, accepted_embeddings = _measure_stability_step(
            reducible_indices=reducible_indices,
            n_curr=n_curr,
            config=config,
            rng=rng,
            embedding_cache=embedding_cache,
        )
        stability_history.append((n_curr, avg_score))
        if avg_score >= config.stability_threshold:
            return stability_history, accepted_embeddings

        new_n = int(n_curr * config.growth_factor)
        if new_n >= config.n_max or new_n >= reducible_count:
            return stability_history, embedding_cache.fetch(reducible_indices)
        n_curr = new_n
        step += 1

    return stability_history, embedding_cache.fetch(reducible_indices)


def _measure_stability_step(
    *,
    reducible_indices: npt.NDArray[np.int64],
    n_curr: int,
    config: SmartSamplerConfig,
    rng: np.random.Generator,
    embedding_cache: _EmbeddingCache,
) -> tuple[float, npt.NDArray[np.float32]]:
    repeat_scores: list[float] = []
    accepted_embeddings: npt.NDArray[np.float32] | None = None
    for _ in range(config.stability_repeats):
        idx_s1 = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
        idx_s2 = np.sort(rng.choice(reducible_indices, n_curr, replace=False))
        emb_s1 = embedding_cache.fetch(idx_s1)
        emb_s2 = embedding_cache.fetch(idx_s2)
        repeat_scores.append(
            float(
                calculate_stability_score(
                    emb_s1,
                    emb_s2,
                    idx_s1,
                    idx_s2,
                    _selection_namespace(config),
                )
            )
        )
        if accepted_embeddings is None:
            accepted_embeddings = emb_s1
    assert accepted_embeddings is not None
    return float(np.mean(repeat_scores)), accepted_embeddings


def _run_patient_selection(
    *,
    reducible_indices: npt.NDArray[np.int64],
    protected_indices: npt.NDArray[np.int64],
    config: SmartSamplerConfig,
    rng: np.random.Generator,
    embedding_cache: _EmbeddingCache,
) -> tuple[SelectionDecision, npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
    selection_pool, heldout_indices = _split_selection_and_holdout(
        reducible_indices=reducible_indices,
        config=config,
        rng=rng,
    )
    selection_embeddings = embedding_cache.fetch(selection_pool)
    heldout_embeddings = _heldout_embeddings(
        heldout_indices=heldout_indices,
        selection_embeddings=selection_embeddings,
        embedding_cache=embedding_cache,
    )
    evaluation_embeddings = heldout_embeddings if len(heldout_indices) > 0 else None
    selection_decision = _select_reducible_pool(
        selection_embeddings=selection_embeddings,
        selection_pool=selection_pool,
        reducible_count=len(reducible_indices),
        config=config,
        evaluation_embeddings=evaluation_embeddings,
    )
    sampled_indices = np.asarray(
        sorted(selection_decision.selected_indices.tolist()),
        dtype=np.int64,
    )
    sorted_protected = np.asarray(sorted(protected_indices.tolist()), dtype=np.int64)
    selected_indices = np.asarray(
        sorted(np.concatenate([sorted_protected, sampled_indices]).tolist()),
        dtype=np.int64,
    )
    return selection_decision, sampled_indices, selected_indices, len(selection_embeddings)


def _heldout_embeddings(
    *,
    heldout_indices: npt.NDArray[np.int64],
    selection_embeddings: npt.NDArray[np.float32],
    embedding_cache: _EmbeddingCache,
) -> npt.NDArray[np.float32]:
    if len(heldout_indices) == 0:
        return np.empty((0, selection_embeddings.shape[1]), dtype=np.float32)
    return embedding_cache.fetch(heldout_indices)


def _select_reducible_pool(
    *,
    selection_embeddings: npt.NDArray[np.float32],
    selection_pool: npt.NDArray[np.int64],
    reducible_count: int,
    config: SmartSamplerConfig,
    evaluation_embeddings: npt.NDArray[np.float32] | None,
) -> SelectionDecision:
    m_ceiling = min(config.m_max, reducible_count)
    selector = (
        select_diverse_samples_gist
        if getattr(config, "use_gist", False)
        else select_diverse_samples
    )
    return selector(
        selection_embeddings,
        selection_pool,
        m_ceiling,
        _selection_namespace(config),
        evaluation_embeddings=evaluation_embeddings,
    )


def _build_patient_selection_result(
    *,
    request: _PatientSelectionBuildRequest,
    selection_decision: SelectionDecision,
    stability_history: list[tuple[int, float]],
    counts: _PatientSelectionCounts,
) -> PatientSelectionResult:
    return PatientSelectionResult(
        patient_id=request.patient_id,
        selected_indices=request.selected_indices,
        protected_indices=np.asarray(sorted(request.protected_indices.tolist()), dtype=np.int64),
        sampled_indices=request.sampled_indices,
        chosen_n_embed=request.chosen_n_embed,
        k_clusters=selection_decision.k_clusters,
        selection_method=selection_decision.selection_method,
        stability_history=stability_history,
        retention_history=selection_decision.retention_history,
        protected_count=len(request.protected_indices),
        protected_positive_label_count=counts.protected_positive_label_count,
        protected_mask_positive_count=counts.protected_mask_positive_count,
        reducible_count=counts.reducible_count,
        total_positive_label_count=counts.total_positive_label_count,
        total_negative_label_count=counts.total_negative_label_count,
        selected_positive_label_count=counts.selected_positive_label_count,
        selected_negative_label_count=counts.selected_negative_label_count,
        adaptive_m_target=selection_decision.adaptive_m_target,
        heldout_count=selection_decision.heldout_count,
        plateau_threshold=selection_decision.plateau_threshold,
        plateau_trigger_improvement=selection_decision.plateau_trigger_improvement,
        plateau_trigger_keep_count=selection_decision.plateau_trigger_keep_count,
        plateau_trigger_step=selection_decision.plateau_trigger_step,
        plateau_stop_reason=selection_decision.plateau_stop_reason,
        plateau_evaluation_mode=selection_decision.plateau_evaluation_mode,
        selected_reducible_count=len(request.sampled_indices),
        rejected_reducible_count=counts.reducible_count - len(request.sampled_indices),
    )


def _load_patient_protection_metadata(
    h5_path: str,
    patient_indices: npt.NDArray[np.int64],
    config: SmartSamplerConfig,
) -> PatientProtectionMetadata:
    protected_mask = np.zeros(len(patient_indices), dtype=bool)
    labels = np.zeros(len(patient_indices), dtype=np.int64)
    mask_positive_mask = np.zeros(len(patient_indices), dtype=bool)
    if len(patient_indices) == 0:
        return PatientProtectionMetadata(
            protected_mask=np.asarray(protected_mask, dtype=bool),
            labels=np.asarray(labels, dtype=np.int64),
            mask_positive_mask=np.asarray(mask_positive_mask, dtype=bool),
        )

    with h5py.File(h5_path, "r") as handle:
        labels = np.asarray(handle["labels"][patient_indices], dtype=np.int64)
        if getattr(config, "protect_positive_labels", True):
            protected_mask |= labels > 0
        if getattr(config, "protect_mask_positive", True):
            masks = np.asarray(handle["masks"][patient_indices])
            mask_axes = tuple(range(1, masks.ndim))
            positive_fraction = np.mean(masks > 0, axis=mask_axes)
            mask_positive_mask = positive_fraction > getattr(
                config, "positive_mask_fraction_threshold", 0.0
            )
            protected_mask |= mask_positive_mask
    return PatientProtectionMetadata(
        protected_mask=np.asarray(protected_mask, dtype=bool),
        labels=np.asarray(labels, dtype=np.int64),
        mask_positive_mask=np.asarray(mask_positive_mask, dtype=bool),
    )


def _select_diverse_samples_adaptive(
    embeddings: npt.NDArray[np.float32],
    global_indices: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    k: int,
    m_target: int,
    config: Any,
    *,
    evaluation_embeddings: npt.NDArray[np.float32] | None,
) -> SelectionDecision:
    max_keep = min(m_target, len(global_indices))
    min_keep = min(max_keep, getattr(config, "KEEP_MIN", max_keep))
    keep_step = max(1, getattr(config, "KEEP_STEP", 1))
    keep_patience = max(1, getattr(config, "KEEP_PATIENCE", 1))
    improvement_threshold = float(getattr(config, "KEEP_IMPROVEMENT_THRESHOLD", 0.0))
    selection_order = _build_cluster_balanced_order(global_indices, labels, seed=config.SEED)
    heldout_count = 0 if evaluation_embeddings is None else len(evaluation_embeddings)
    evaluation_mode = "within_patient_patch_holdout" if heldout_count > 0 else "candidate_pool"

    if min_keep >= max_keep:
        return SelectionDecision(
            selected_indices=selection_order[:max_keep],
            k_clusters=k,
            adaptive_m_target=max_keep,
            selection_method="adaptive_keep_all",
            retention_history=[(max_keep, 0.0)],
            heldout_count=heldout_count,
            plateau_threshold=improvement_threshold,
            plateau_trigger_improvement=None,
            plateau_trigger_keep_count=max_keep,
            plateau_trigger_step=0,
            plateau_stop_reason="max_keep_reached",
            plateau_evaluation_mode=evaluation_mode,
        )

    if evaluation_embeddings is not None and len(evaluation_embeddings) == 0:
        evaluation_embeddings = None

    if evaluation_embeddings is None:
        return SelectionDecision(
            selected_indices=selection_order[:max_keep],
            k_clusters=k,
            adaptive_m_target=max_keep,
            selection_method="adaptive_holdout_unavailable",
            retention_history=[],
            heldout_count=0,
            plateau_threshold=improvement_threshold,
            plateau_trigger_improvement=None,
            plateau_trigger_keep_count=max_keep,
            plateau_trigger_step=None,
            plateau_stop_reason="holdout_unavailable",
            plateau_evaluation_mode="holdout_unavailable",
        )

    selection_order_positions = _selection_order_positions(global_indices, selection_order)
    target_embeddings = embeddings if evaluation_embeddings is None else evaluation_embeddings
    current_min_distances = _prefix_coverage_distances(
        embeddings,
        selection_order_positions[:min_keep],
        target_embeddings=target_embeddings,
    )
    retention_history: list[tuple[int, float]] = []
    previous_score: float | None = None
    previous_count: int | None = None
    plateau_steps = 0
    selected_count = min_keep
    best_count = max_keep

    while selected_count <= max_keep:
        score = (
            float(np.mean(current_min_distances))
            if len(current_min_distances)
            else float("inf")
        )
        retention_history.append((selected_count, score))
        if previous_score is not None:
            relative_improvement = (previous_score - score) / max(previous_score, 1e-12)
            if relative_improvement < improvement_threshold:
                plateau_steps += 1
            else:
                plateau_steps = 0
            if plateau_steps >= keep_patience:
                best_count = previous_count if previous_count is not None else selected_count
                return SelectionDecision(
                    selected_indices=selection_order[:best_count],
                    k_clusters=k,
                    adaptive_m_target=max_keep,
                    selection_method="adaptive_plateau",
                    retention_history=retention_history,
                    heldout_count=heldout_count,
                    plateau_threshold=improvement_threshold,
                    plateau_trigger_improvement=relative_improvement,
                    plateau_trigger_keep_count=best_count,
                    plateau_trigger_step=len(retention_history) - 1,
                    plateau_stop_reason="plateau_threshold",
                    plateau_evaluation_mode=evaluation_mode,
                )
        previous_score = score
        previous_count = selected_count
        if selected_count == max_keep:
            best_count = max_keep
            break
        next_count = min(max_keep, selected_count + keep_step)
        current_min_distances = _update_prefix_coverage_distances(
            embeddings,
            current_min_distances,
            selection_order_positions[selected_count:next_count],
            target_embeddings=target_embeddings,
        )
        selected_count = next_count

    return SelectionDecision(
        selected_indices=selection_order[:best_count],
        k_clusters=k,
        adaptive_m_target=max_keep,
        selection_method="adaptive_plateau",
        retention_history=retention_history,
        heldout_count=heldout_count,
        plateau_threshold=improvement_threshold,
        plateau_trigger_improvement=None,
        plateau_trigger_keep_count=best_count,
        plateau_trigger_step=len(retention_history) - 1 if retention_history else None,
        plateau_stop_reason="max_keep_reached",
        plateau_evaluation_mode=evaluation_mode,
    )


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
    *,
    evaluation_embeddings: npt.NDArray[np.float32] | None = None,
) -> float:
    selection_mask = np.isin(global_indices, selected_indices)
    selected_embeddings = embeddings[selection_mask]
    if len(selected_embeddings) == 0:
        return float("inf")
    target_embeddings = embeddings if evaluation_embeddings is None else evaluation_embeddings
    distances = np.linalg.norm(
        target_embeddings[:, None, :] - selected_embeddings[None, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1)))


def _selection_order_positions(
    global_indices: npt.NDArray[np.int64],
    selection_order: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    positions_by_index = {
        int(index): position for position, index in enumerate(global_indices.tolist())
    }
    return np.asarray(
        [positions_by_index[int(index)] for index in selection_order.tolist()],
        dtype=np.int64,
    )


def _prefix_coverage_distances(
    embeddings: npt.NDArray[np.float32],
    selected_positions: npt.NDArray[np.int64],
    *,
    target_embeddings: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if len(selected_positions) == 0:
        return np.full(len(target_embeddings), np.inf, dtype=np.float32)
    selected_embeddings = embeddings[selected_positions]
    distances = np.linalg.norm(
        target_embeddings[:, None, :] - selected_embeddings[None, :, :],
        axis=2,
    )
    return np.asarray(np.min(distances, axis=1), dtype=np.float32)


def _update_prefix_coverage_distances(
    embeddings: npt.NDArray[np.float32],
    current_min_distances: npt.NDArray[np.float32],
    added_positions: npt.NDArray[np.int64],
    *,
    target_embeddings: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if len(added_positions) == 0:
        return current_min_distances
    added_embeddings = embeddings[added_positions]
    added_distances = np.linalg.norm(
        target_embeddings[:, None, :] - added_embeddings[None, :, :],
        axis=2,
    )
    added_min_distances = np.asarray(np.min(added_distances, axis=1), dtype=np.float32)
    return np.minimum(current_min_distances, added_min_distances)


def _compute_adaptive_m_target(
    *,
    reducible_count: int,
    m_ceiling: int,
    k_used: int,
    config: Any,
) -> int:
    bounded_ceiling = min(reducible_count, m_ceiling)
    if bounded_ceiling <= 0:
        return 0

    min_target = min(bounded_ceiling, max(1, getattr(config, "KEEP_MIN", 1)))
    k_max = max(1, int(getattr(config, "K_MAX", max(k_used, 1))))
    if bounded_ceiling <= min_target or k_max <= 1:
        return bounded_ceiling

    heterogeneity_ratio = min(1.0, max(0.0, float(k_used - 1) / float(k_max - 1)))
    scaled_target = min_target + (bounded_ceiling - min_target) * heterogeneity_ratio
    return int(min(bounded_ceiling, max(min_target, np.ceil(scaled_target))))


def _split_selection_and_holdout(
    *,
    reducible_indices: npt.NDArray[np.int64],
    config: SmartSamplerConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    if not getattr(config, "adaptive_keep_enabled", False):
        return np.asarray(reducible_indices, dtype=np.int64), np.empty(0, dtype=np.int64)

    selection_pool = np.asarray(reducible_indices, dtype=np.int64)
    holdout_indices = np.empty(0, dtype=np.int64)
    desired_holdout = int(np.ceil(len(reducible_indices) * 0.2))
    if desired_holdout <= 0:
        return selection_pool, np.asarray(holdout_indices, dtype=np.int64)

    min_selection_pool = max(1, min(len(selection_pool), min(config.keep_min, config.m_max)))
    removable = max(0, len(selection_pool) - min_selection_pool)
    extra_needed = max(0, desired_holdout - len(holdout_indices))
    extra_holdout = min(removable, extra_needed)
    if extra_holdout <= 0:
        return selection_pool, np.asarray(sorted(holdout_indices.tolist()), dtype=np.int64)

    holdout_positions = np.sort(rng.choice(len(selection_pool), size=extra_holdout, replace=False))
    holdout_mask = np.zeros(len(selection_pool), dtype=bool)
    holdout_mask[holdout_positions] = True
    promoted_holdout = selection_pool[holdout_mask]
    final_selection_pool = selection_pool[~holdout_mask]
    final_holdout = np.asarray(
        sorted(np.concatenate([holdout_indices, promoted_holdout]).tolist()),
        dtype=np.int64,
    )
    return np.asarray(sorted(final_selection_pool.tolist()), dtype=np.int64), final_holdout


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
