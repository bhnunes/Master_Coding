from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import gp_minimize
from skopt.space import Integer

from helpers.graph_contamination import GraphContaminationParameters, calculate_roi_contamination
from helpers.optimization_sampling_sampling import ImageMaskPair, discover_image_mask_pairs

APPROVED_LABEL = "Approved"
REJECTED_LABEL = "Rejected"
REVIEW_EXTENSIONS = {".png", ".jpg", ".tif"}
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

ProgressItem = TypeVar("ProgressItem")
ProgressFactory = Callable[[Iterable[ProgressItem]], Iterable[ProgressItem]]
Scorer = Callable[[Path, Path, GraphContaminationParameters], float | None]
RecordProgressFactory = Callable[[Iterable["LabeledSourceRecord"]], Iterable["LabeledSourceRecord"]]


@dataclass(frozen=True)
class LabeledSourceRecord:
    """A review label matched back to a source image/mask pair."""

    pair: ImageMaskPair
    label: str


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
    source_image_folder: Path,
    source_mask_folder: Path,
    logger: logging.Logger,
) -> list[LabeledSourceRecord]:
    """Resolve review labels to existing source image/mask pairs."""

    source_pairs = discover_image_mask_pairs(source_image_folder, source_mask_folder)
    records: list[LabeledSourceRecord] = []
    missing_files = False
    for stem, label in labels_by_stem.items():
        pair = source_pairs.get(stem)
        if pair is None:
            logger.warning("Source image/mask pair not found, skipping review item: %s", stem)
            missing_files = True
            continue
        records.append(LabeledSourceRecord(pair=pair, label=label))

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
        f1 = f1_score(true_labels, predictions, pos_label=REJECTED_LABEL, zero_division=0)
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
) -> float:
    """Evaluate the tuned parameters on the held-out test set and return final tau."""

    logger.info("\n--- Evaluating final model on the held-out test set ---")
    logger.info("Step 1: Finding final contamination threshold using all training data...")
    train_rates, train_labels = _score_records(train_records, best_params, scorer)
    final_tau = select_best_contamination_threshold(train_rates, train_labels)
    logger.info(" > Final optimal contamination threshold (tau) found: %.2f", final_tau)
    logger.info("Step 2: Evaluating performance on test set...")
    test_iterable: Iterable[LabeledSourceRecord] = test_records
    if progress_factory is not None:
        test_iterable = progress_factory(test_records)
    test_rates, test_labels = _score_records(test_iterable, best_params, scorer)
    predictions = [REJECTED_LABEL if rate > final_tau else APPROVED_LABEL for rate in test_rates]
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, pos_label=REJECTED_LABEL, zero_division=0)
    logger.info("\n--- Final Test Set Performance ---")
    logger.info("Accuracy: %.3f", accuracy)
    logger.info("F1-Score (Rejected): %.3f", f1)
    return final_tau


def run_graph_tuning_pipeline(
    *,
    source_image_folder: Path,
    source_mask_folder: Path,
    review_base_dir: Path,
    test_set_size: float,
    n_splits_inner_cv: int,
    n_bayesian_calls: int,
    n_initial_points: int,
    random_state: int,
    bg_intensity_range: tuple[int, int],
    k_range: tuple[int, int],
    min_size_range: tuple[int, int],
    erosion_range: tuple[int, int],
    logger: logging.Logger,
    scorer: Scorer = calculate_roi_contamination,
    optimizer: Callable[..., GraphTuningResult] | None = None,
    progress_factory: RecordProgressFactory | None = None,
) -> GraphTuningSummary:
    """Run Stage 4.2 graph tuning and return the best recommended parameters."""

    started_at = time.time()
    logger.info("--- Hyperparameter Tuning with Bayesian Optimization ---")
    labels_by_stem = collect_review_labels(review_base_dir)
    logger.info("Performing pre-flight check on all source file paths...")
    records = resolve_labeled_source_records(
        labels_by_stem=labels_by_stem,
        source_image_folder=source_image_folder,
        source_mask_folder=source_mask_folder,
        logger=logger,
    )

    train_records, test_records = _split_records(
        records, test_set_size=test_set_size, random_state=random_state
    )
    logger.info(
        "Data split: %s training, %s test (after validation).",
        len(train_records),
        len(test_records),
    )
    logger.info(
        "Starting Bayesian Optimization with a budget of %s evaluations.",
        n_bayesian_calls,
    )

    search_space = build_search_space(
        bg_intensity_range=bg_intensity_range,
        k_range=k_range,
        min_size_range=min_size_range,
        erosion_range=erosion_range,
    )

    objective = _build_objective(
        train_records=train_records,
        n_splits_inner_cv=n_splits_inner_cv,
        random_state=random_state,
        scorer=scorer,
    )
    optimization_result = (
        optimizer(objective=objective, search_space=search_space)
        if optimizer is not None
        else _run_gp_minimize(
            objective=objective,
            search_space=search_space,
            n_bayesian_calls=n_bayesian_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
        )
    )

    logger.info("\n\n--- Bayesian Optimization Complete ---")
    logger.info(
        "Best cross-validated F1-Score (Rejected): %.3f",
        optimization_result.best_score,
    )
    logger.info("Best graph parameters found: %s", optimization_result.best_params)

    if test_records:
        final_tau = evaluate_on_test_set(
            train_records=train_records,
            test_records=test_records,
            best_params=optimization_result.best_params,
            logger=logger,
            scorer=scorer,
            progress_factory=progress_factory,
        )
    else:
        logger.warning("Test set is empty. Skipping final evaluation.")
        train_rates, train_labels = _score_records(
            train_records, optimization_result.best_params, scorer
        )
        final_tau = select_best_contamination_threshold(train_rates, train_labels)

    logger.info("\nTotal execution time: %.2f minutes.", (time.time() - started_at) / 60)
    logger.info("Final recommended tau: %.2f", final_tau)
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


def _build_objective(
    *,
    train_records: Sequence[LabeledSourceRecord],
    n_splits_inner_cv: int,
    random_state: int,
    scorer: Scorer,
) -> Callable[[list[int]], float]:
    labels = np.array([record.label for record in train_records])
    record_array = np.array(train_records, dtype=object)
    splitter = StratifiedKFold(n_splits=n_splits_inner_cv, shuffle=True, random_state=random_state)

    def objective(values: list[int]) -> float:
        params = GraphContaminationParameters(
            bg_intensity_thresh=int(values[0]),
            k=float(values[1]),
            min_size=int(values[2]),
            erosion_px=int(values[3]),
        )
        fold_f1_scores: list[float] = []
        for train_indices, validation_indices in splitter.split(record_array, labels):
            train_fold = list(record_array[train_indices])
            validation_fold = list(record_array[validation_indices])
            train_rates, train_labels = _score_records(train_fold, params, scorer)
            optimal_tau = select_best_contamination_threshold(train_rates, train_labels)
            validation_rates, validation_labels = _score_records(validation_fold, params, scorer)
            predictions = [
                REJECTED_LABEL if rate > optimal_tau else APPROVED_LABEL
                for rate in validation_rates
            ]
            f1 = f1_score(
                validation_labels,
                predictions,
                pos_label=REJECTED_LABEL,
                zero_division=0,
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
    best_values = result.x
    return GraphTuningResult(
        best_score=-float(result.fun if result.fun is not None else 0.0),
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
) -> tuple[list[float], list[str]]:
    rates: list[float] = []
    labels: list[str] = []
    for record in records:
        rate = scorer(record.pair.image_path, record.pair.mask_path, params)
        if rate is None or np.isnan(rate):
            continue
        rates.append(float(rate))
        labels.append(record.label)
    return rates, labels


def _split_records(
    records: Sequence[LabeledSourceRecord],
    *,
    test_set_size: float,
    random_state: int,
) -> tuple[list[LabeledSourceRecord], list[LabeledSourceRecord]]:
    labels = [record.label for record in records]
    train_records, test_records = train_test_split(
        list(records),
        test_size=test_set_size,
        random_state=random_state,
        stratify=labels,
    )
    return list(train_records), list(test_records)
