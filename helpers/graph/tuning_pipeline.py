from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from skopt import gp_minimize
from skopt.space import Integer

from helpers.graph.contamination import (
    GraphContaminationParameters,
    calculate_roi_contamination,
    calculate_roi_contamination_from_arrays,
    load_graph_sources,
)
from helpers.graph.parameter_store import GraphCleaningParameterArtifact
from helpers.optimization_sampling.sampling import (
    ImageMaskPair,
    discover_hdf5_image_mask_pairs,
)

APPROVED_LABEL = "Approved"
REJECTED_LABEL = "Rejected"
REVIEW_EXTENSIONS = {".png", ".jpg", ".tif"}
THRESHOLDS = np.arange(0.05, 0.96, 0.01)
ZERO_DIVISION = 0.0

ProgressItem = TypeVar("ProgressItem")
ProgressFactory = Callable[[Iterable[ProgressItem]], Iterable[ProgressItem]]
Scorer = Callable[[Path | str, Path | str, GraphContaminationParameters], float | None]
RecordProgressFactory = Callable[[Iterable["LabeledSourceRecord"]], Iterable["LabeledSourceRecord"]]


@dataclass(frozen=True)
class LabeledSourceRecord:
    """A review label matched back to a source image/mask pair."""

    pair: ImageMaskPair
    label: str
    group_id: str


@dataclass(frozen=True)
class PreloadedGraphSources:
    """In-memory image and mask arrays for one reviewed source record."""

    image: npt.NDArray[np.uint8]
    mask: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class GraphTuningResult:
    """Best graph search result returned by Bayesian optimization."""

    best_score: float
    best_params: GraphContaminationParameters


@dataclass(frozen=True)
class GraphTuningSummary:
    """Execution summary for Stage 4.2 graph tuning."""

    total_labeled_pairs: int
    training_pairs: int
    test_pairs: int
    best_cross_validated_f1: float
    best_params: GraphContaminationParameters
    final_tau: float


@dataclass(frozen=True)
class GraphTuningPipelineConfig:
    source_hdf5_path: Path
    review_base_dir: Path
    test_set_size: float
    n_splits_inner_cv: int
    n_bayesian_calls: int
    n_initial_points: int
    random_state: int
    bg_intensity_range: tuple[int, int]
    k_range: tuple[int, int]
    min_size_range: tuple[int, int]
    erosion_range: tuple[int, int]
    logger: logging.Logger
    scorer: Scorer = calculate_roi_contamination
    optimizer: Callable[..., GraphTuningResult] | None = None
    progress_factory: RecordProgressFactory | None = None


def collect_review_labels(review_base_dir: Path) -> dict[str, str]:
    """Load approved/rejected labels from the manual-review folders."""

    approved_dir = review_base_dir / "APPROVED"
    rejected_dir = review_base_dir / "REJECTED"
    approved_files = _list_review_files(approved_dir)
    rejected_files = _list_review_files(rejected_dir)
    if not approved_files and not rejected_files:
        raise FileNotFoundError("Both APPROVED and REJECTED folders are empty. Nothing to tune.")
    return {
        **{path.stem: APPROVED_LABEL for path in approved_files},
        **{path.stem: REJECTED_LABEL for path in rejected_files},
    }


def resolve_labeled_source_records(
    *,
    labels_by_stem: dict[str, str],
    source_hdf5_path: Path,
    logger: logging.Logger,
) -> list[LabeledSourceRecord]:
    """Resolve review labels to existing source image/mask pairs."""

    source_pairs = discover_hdf5_image_mask_pairs(source_hdf5_path)
    records: list[LabeledSourceRecord] = []
    missing_files = False
    for stem, label in labels_by_stem.items():
        pair = source_pairs.get(stem)
        if pair is None:
            logger.warning("Source image/mask pair not found, skipping review item: %s", stem)
            missing_files = True
            continue
        records.append(
            LabeledSourceRecord(
                pair=pair,
                label=label,
                group_id=infer_group_id_from_stem(pair.stem),
            )
        )

    if missing_files:
        logger.error(
            "Some source files were not found. The process can continue with the valid files, "
            "but results may be skewed. Please check the warnings above."
        )
    if not records:
        raise ValueError("No valid image/mask pairs found after checking paths. Halting execution.")
    return records


def select_best_contamination_threshold(
    contamination_rates: Sequence[float],
    true_labels: Sequence[str],
    *,
    thresholds: NDArray[np.float64] = THRESHOLDS,
) -> float:
    """Select the threshold that maximizes F1 for the rejected class."""

    best_f1 = -1.0
    best_tau = 0.0
    for tau in thresholds:
        predictions = [
            REJECTED_LABEL if rate > tau else APPROVED_LABEL for rate in contamination_rates
        ]
        f1 = f1_score(
            true_labels,
            predictions,
            pos_label=REJECTED_LABEL,
            zero_division=ZERO_DIVISION,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)
    return best_tau


def evaluate_on_test_set(
    *,
    train_records: Sequence[LabeledSourceRecord],
    test_records: Sequence[LabeledSourceRecord],
    best_params: GraphContaminationParameters,
    logger: logging.Logger,
    scorer: Scorer,
    progress_factory: RecordProgressFactory | None = None,
    preloaded_sources: dict[str, PreloadedGraphSources] | None = None,
) -> float:
    """Evaluate the tuned parameters on the held-out test set and return final tau."""

    logger.info("\n--- Evaluating final model on the held-out test set ---")
    logger.info("Step 1: Finding final contamination threshold using all training data...")
    train_rates, train_labels = _score_records(
        train_records,
        best_params,
        scorer,
        preloaded_sources=preloaded_sources,
    )
    final_tau = select_best_contamination_threshold(train_rates, train_labels)
    logger.info(" > Final optimal contamination threshold (tau) found: %.2f", final_tau)
    logger.info("Step 2: Evaluating performance on test set...")
    test_iterable: Iterable[LabeledSourceRecord] = test_records
    if progress_factory is not None:
        test_iterable = progress_factory(test_records)
    test_rates, test_labels = _score_records(
        test_iterable,
        best_params,
        scorer,
        preloaded_sources=preloaded_sources,
    )
    predictions = [REJECTED_LABEL if rate > final_tau else APPROVED_LABEL for rate in test_rates]
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(
        test_labels,
        predictions,
        pos_label=REJECTED_LABEL,
        zero_division=ZERO_DIVISION,
    )
    logger.info("\n--- Final Test Set Performance ---")
    logger.info("Accuracy: %.3f", accuracy)
    logger.info("F1-Score (Rejected): %.3f", f1)
    return final_tau


def run_graph_tuning_pipeline(config: GraphTuningPipelineConfig) -> GraphTuningSummary:
    """Run Stage 4.2 graph tuning and return the best recommended parameters."""

    started_at = time.time()
    config.logger.info("--- Hyperparameter Tuning with Bayesian Optimization ---")
    labels_by_stem = collect_review_labels(config.review_base_dir)
    config.logger.info("Performing pre-flight check on all source file paths...")
    records = resolve_labeled_source_records(
        labels_by_stem=labels_by_stem,
        source_hdf5_path=config.source_hdf5_path,
        logger=config.logger,
    )
    preloaded_sources = _preload_record_sources(
        records,
        scorer=config.scorer,
        logger=config.logger,
    )

    train_records, test_records = _split_records(
        records,
        test_set_size=config.test_set_size,
        random_state=config.random_state,
    )
    config.logger.info(
        "Data split: %s training, %s test (after validation).",
        len(train_records),
        len(test_records),
    )
    config.logger.info(
        "Starting Bayesian Optimization with a budget of %s evaluations.",
        config.n_bayesian_calls,
    )

    search_space = build_search_space(
        bg_intensity_range=config.bg_intensity_range,
        k_range=config.k_range,
        min_size_range=config.min_size_range,
        erosion_range=config.erosion_range,
    )

    objective = _build_objective(
        train_records=train_records,
        n_splits_inner_cv=config.n_splits_inner_cv,
        random_state=config.random_state,
        scorer=config.scorer,
        preloaded_sources=preloaded_sources,
    )
    optimization_result = (
        config.optimizer(objective=objective, search_space=search_space)
        if config.optimizer is not None
        else _run_gp_minimize(
            objective=objective,
            search_space=search_space,
            n_bayesian_calls=config.n_bayesian_calls,
            n_initial_points=config.n_initial_points,
            random_state=config.random_state,
        )
    )

    config.logger.info("\n\n--- Bayesian Optimization Complete ---")
    config.logger.info(
        "Best cross-validated F1-Score (Rejected): %.3f",
        optimization_result.best_score,
    )
    config.logger.info("Best graph parameters found: %s", optimization_result.best_params)

    if test_records:
        final_tau = evaluate_on_test_set(
            train_records=train_records,
            test_records=test_records,
            best_params=optimization_result.best_params,
            logger=config.logger,
            scorer=config.scorer,
            progress_factory=config.progress_factory,
            preloaded_sources=preloaded_sources,
        )
    else:
        config.logger.warning("Test set is empty. Skipping final evaluation.")
        train_rates, train_labels = _score_records(
            train_records,
            optimization_result.best_params,
            config.scorer,
            preloaded_sources=preloaded_sources,
        )
        final_tau = select_best_contamination_threshold(train_rates, train_labels)

    config.logger.info("\nTotal execution time: %.2f minutes.", (time.time() - started_at) / 60)
    config.logger.info("Final recommended tau: %.2f", final_tau)
    return GraphTuningSummary(
        total_labeled_pairs=len(records),
        training_pairs=len(train_records),
        test_pairs=len(test_records),
        best_cross_validated_f1=optimization_result.best_score,
        best_params=optimization_result.best_params,
        final_tau=final_tau,
    )


def build_recommendation_message(summary: GraphTuningSummary) -> str:
    """Build the final recommendation message for the operator."""

    return (
        "\n\n---------------------------------------------------------\n"
        "--- Recommended Default Parameters for Pipeline ---\n"
        "---------------------------------------------------------\n"
        f"Graph Method `k`:                      {summary.best_params.k:g}\n"
        f"Graph Method `min_size`:               {summary.best_params.min_size}\n"
        "Graph Method `bg_intensity_thresh`:    "
        f"{summary.best_params.bg_intensity_thresh}\n"
        "ROI Erosion (Safety Margin):           "
        f"{summary.best_params.erosion_px} pixels\n"
        f"Contamination Rate Threshold (tau):    {summary.final_tau:.2f}\n"
        "---------------------------------------------------------"
    )


def build_graph_cleaning_parameter_artifact(
    summary: GraphTuningSummary,
    *,
    random_state: int,
) -> GraphCleaningParameterArtifact:
    """Build the Stage 4 handoff artifact used by `4_3_cleaner_script.py`."""

    return GraphCleaningParameterArtifact(
        graph_params=summary.best_params,
        tau=summary.final_tau,
        best_cross_validated_f1=summary.best_cross_validated_f1,
        total_labeled_pairs=summary.total_labeled_pairs,
        training_pairs=summary.training_pairs,
        test_pairs=summary.test_pairs,
        random_state=random_state,
        generated_by="4_2_tune_graph_method.py",
    )


def build_search_space(
    *,
    bg_intensity_range: tuple[int, int],
    k_range: tuple[int, int],
    min_size_range: tuple[int, int],
    erosion_range: tuple[int, int],
) -> list[Integer]:
    """Build the Bayesian search space using the configured integer ranges."""

    return [
        Integer(*bg_intensity_range, name="bg_intensity_thresh"),
        Integer(*k_range, name="k"),
        Integer(*min_size_range, name="min_size"),
        Integer(*erosion_range, name="erosion_px"),
    ]


def infer_group_id_from_stem(stem: str) -> str:
    match = re.search(r"PATIENT_([^_]+)", stem)
    if match is not None:
        candidate = match.group(1).strip("-_")
        if candidate:
            return candidate
    return stem


def _build_objective(
    *,
    train_records: Sequence[LabeledSourceRecord],
    n_splits_inner_cv: int,
    random_state: int,
    scorer: Scorer,
    preloaded_sources: dict[str, PreloadedGraphSources] | None = None,
) -> Callable[[list[int]], float]:
    labels = np.array([record.label for record in train_records])
    record_array = np.array(train_records, dtype=object)
    groups = np.array([record.group_id for record in train_records], dtype=object)
    cv_splits = _build_grouped_cv_splits(
        labels=labels,
        groups=groups,
        n_splits_inner_cv=n_splits_inner_cv,
        random_state=random_state,
    )

    def objective(values: list[int]) -> float:
        params = GraphContaminationParameters(
            bg_intensity_thresh=int(values[0]),
            k=float(values[1]),
            min_size=int(values[2]),
            erosion_px=int(values[3]),
        )
        fold_f1_scores: list[float] = []
        scored_rates = _score_record_rates(
            record_array,
            params,
            scorer,
            preloaded_sources=preloaded_sources,
        )
        for train_indices, validation_indices in cv_splits:
            train_rates, train_labels = _collect_scored_fold(
                record_array,
                train_indices,
                scored_rates,
            )
            optimal_tau = select_best_contamination_threshold(train_rates, train_labels)
            validation_rates, validation_labels = _collect_scored_fold(
                record_array,
                validation_indices,
                scored_rates,
            )
            predictions = [
                REJECTED_LABEL if rate > optimal_tau else APPROVED_LABEL
                for rate in validation_rates
            ]
            f1 = f1_score(
                validation_labels,
                predictions,
                pos_label=REJECTED_LABEL,
                zero_division=ZERO_DIVISION,
            )
            fold_f1_scores.append(float(f1))
        return -float(np.mean(fold_f1_scores) if fold_f1_scores else 0.0)

    return objective


def _run_gp_minimize(
    *,
    objective: Callable[[list[int]], float],
    search_space: list[Integer],
    n_bayesian_calls: int,
    n_initial_points: int,
    random_state: int,
) -> GraphTuningResult:
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_bayesian_calls,
        n_initial_points=n_initial_points,
        acq_func="EI",
        random_state=random_state,
        verbose=True,
    )
    result_payload = result
    assert result_payload is not None
    best_values = result_payload.x
    return GraphTuningResult(
        best_score=-float(result_payload.fun if result_payload.fun is not None else 0.0),
        best_params=GraphContaminationParameters(
            bg_intensity_thresh=int(best_values[0]),
            k=float(best_values[1]),
            min_size=int(best_values[2]),
            erosion_px=int(best_values[3]),
        ),
    )


def _list_review_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Could not read review folder: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in REVIEW_EXTENSIONS
    )


def _score_records(
    records: Iterable[LabeledSourceRecord],
    params: GraphContaminationParameters,
    scorer: Scorer,
    *,
    preloaded_sources: dict[str, PreloadedGraphSources] | None = None,
) -> tuple[list[float], list[str]]:
    rates: list[float] = []
    labels: list[str] = []
    for record in records:
        rate = _score_record(
            record,
            params,
            scorer,
            preloaded_sources=preloaded_sources,
        )
        if rate is None or np.isnan(rate):
            continue
        rates.append(float(rate))
        labels.append(record.label)
    return rates, labels


def _score_record_rates(
    records: NDArray[np.object_],
    params: GraphContaminationParameters,
    scorer: Scorer,
    *,
    preloaded_sources: dict[str, PreloadedGraphSources] | None = None,
) -> list[float | None]:
    return [
        _score_record(
            cast(LabeledSourceRecord, record),
            params,
            scorer,
            preloaded_sources=preloaded_sources,
        )
        for record in records.tolist()
    ]


def _collect_scored_fold(
    records: NDArray[np.object_],
    indices: NDArray[np.int64],
    scored_rates: Sequence[float | None],
) -> tuple[list[float], list[str]]:
    rates: list[float] = []
    labels: list[str] = []
    for index in indices.tolist():
        rate = scored_rates[index]
        if rate is None or np.isnan(rate):
            continue
        record = cast(LabeledSourceRecord, records[index])
        rates.append(float(rate))
        labels.append(record.label)
    return rates, labels


def _score_record(
    record: LabeledSourceRecord,
    params: GraphContaminationParameters,
    scorer: Scorer,
    *,
    preloaded_sources: dict[str, PreloadedGraphSources] | None = None,
) -> float | None:
    preloaded = preloaded_sources.get(record.pair.stem) if preloaded_sources is not None else None
    if preloaded is not None and scorer is calculate_roi_contamination:
        return calculate_roi_contamination_from_arrays(
            preloaded.image,
            preloaded.mask,
            params,
            base_name=record.pair.output_name,
        )
    return scorer(record.pair.image_path, record.pair.mask_path, params)


def _preload_record_sources(
    records: Sequence[LabeledSourceRecord],
    *,
    scorer: Scorer,
    logger: logging.Logger,
) -> dict[str, PreloadedGraphSources] | None:
    if scorer is not calculate_roi_contamination:
        return None

    preloaded_sources: dict[str, PreloadedGraphSources] = {}
    for record in records:
        image, mask = load_graph_sources(record.pair.image_path, record.pair.mask_path)
        if image is None or mask is None:
            logger.warning("Failed to preload graph tuning source: %s", record.pair.output_name)
            continue
        preloaded_sources[record.pair.stem] = PreloadedGraphSources(image=image, mask=mask)
    return preloaded_sources


def _build_grouped_cv_splits(
    *,
    labels: NDArray[np.str_],
    groups: NDArray[np.object_],
    n_splits_inner_cv: int,
    random_state: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    unique_groups = np.unique(groups)
    max_splits = min(len(unique_groups), n_splits_inner_cv)
    for n_splits in range(max_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        try:
            return [
                (train_idx.astype(np.int64), validation_idx.astype(np.int64))
                for train_idx, validation_idx in splitter.split(
                    np.zeros(len(labels), dtype=np.uint8),
                    labels,
                    groups,
                )
            ]
        except ValueError:
            continue
    raise ValueError(
        "Grouped cross-validation requires at least two feasible patient/slide groups per fold."
    )


def _split_records(
    records: Sequence[LabeledSourceRecord],
    *,
    test_set_size: float,
    random_state: int,
) -> tuple[list[LabeledSourceRecord], list[LabeledSourceRecord]]:
    labels = np.array([record.label for record in records])
    groups = np.array([record.group_id for record in records], dtype=object)
    unique_groups = np.unique(groups)
    requested_splits = max(2, int(round(1.0 / test_set_size)))
    max_splits = min(len(unique_groups), requested_splits)
    best_split: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None
    best_ratio_delta: float | None = None

    for n_splits in range(max_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        try:
            for train_idx, test_idx in splitter.split(
                np.zeros(len(labels), dtype=np.uint8),
                labels,
                groups,
            ):
                ratio_delta = abs((len(test_idx) / max(1, len(records))) - test_set_size)
                if best_split is None or best_ratio_delta is None or ratio_delta < best_ratio_delta:
                    best_split = (train_idx.astype(np.int64), test_idx.astype(np.int64))
                    best_ratio_delta = ratio_delta
        except ValueError:
            continue

    if best_split is None:
        raise ValueError(
            "Unable to create a patient-grouped train/test split. "
            "Check class balance and group counts."
        )

    train_indices, test_indices = best_split
    record_list = list(records)
    train_records = [record_list[index] for index in train_indices.tolist()]
    test_records = [record_list[index] for index in test_indices.tolist()]
    return train_records, test_records
