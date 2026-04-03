from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build one patient-level row for split stratification and auditing."""

    return (
        df.groupby("patient_id")
        .agg(patient_label=("label", "max"), n_images=("label", "size"))
        .reset_index()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def _adaptive_min_patients(N: int, constraints: SplitConstraints) -> tuple[int, int, int]:
    min_train = constraints.min_train_patients
    min_val = constraints.min_val_patients
    min_test = constraints.min_test_patients
    if not constraints.adaptive:
        return min_train, min_val, min_test
    if N < (min_train + min_val + min_test):
        desired_test = max(1, int(round(constraints.test_ratio * N)))
        desired_val = max(1, int(round(constraints.val_ratio * N)))
        desired_train = max(1, N - desired_test - desired_val)
        if desired_train <= 0:
            desired_train = 1
        remaining = N - desired_train
        desired_test = min(desired_test, remaining)
        remaining -= desired_test
        desired_val = min(desired_val, remaining)
        min_train = min(min_train, desired_train)
        min_test = min(min_test, desired_test)
        min_val = min(min_val, desired_val)
        logging.warning(
            "Adaptive sizing engaged due to limited N_patients. "
            "Relaxed mins -> train>=%s, val>=%s, test>=%s (N=%s).",
            min_train,
            min_val,
            min_test,
            N,
        )
    if N >= 3:
        min_train = max(1, min_train)
        min_val = max(1, min_val)
        min_test = max(1, min_test)
    return min_train, min_val, min_test


def decide_split_sizes(
    patient_df: pd.DataFrame,
    constraints: SplitConstraints,
) -> tuple[int, int, int, dict[str, Any]]:
    """Resolve train/validation/test patient counts under configured constraints."""

    total_patients = len(patient_df)
    min_train, min_val, min_test = _adaptive_min_patients(total_patients, constraints)
    n_test = max(min_test, int(round(constraints.test_ratio * total_patients)))
    n_val = max(min_val, int(round(constraints.val_ratio * total_patients)))
    n_train = total_patients - n_test - n_val
    if n_train < min_train:
        n_val = max(min_val, total_patients - n_test - min_train)
        n_train = total_patients - n_test - n_val
    if constraints.adaptive and n_train < min_train:
        n_test = max(min_test, total_patients - n_val - min_train)
        n_train = total_patients - n_test - n_val
    if n_train < min_train or n_val < min_val or n_test < min_test:
        raise ValueError(
            f"Split impossible after sizing: train={n_train}, val={n_val}, test={n_test}. "
            f"mins: train>={min_train}, val>={min_val}, test>={min_test}. "
            "Need more patients or relax constraints."
        )
    meta = {
        "N_patients": total_patients,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "min_train": min_train,
        "min_val": min_val,
        "min_test": min_test,
        "test_ratio": constraints.test_ratio,
        "val_ratio": constraints.val_ratio,
        "adaptive": constraints.adaptive,
    }
    return n_train, n_val, n_test, meta


def _build_split_return(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    train_patients: set[int],
    val_patients: set[int],
    test_patients: set[int],
    seed: int,
    attempt: int,
    constraints: SplitConstraints,
    sizing_meta: dict[str, Any],
    score: float | None,
    score_split: str,
) -> dict[str, Any]:
    split_by_patient = {patient_id: "TRAIN" for patient_id in train_patients}
    split_by_patient.update({patient_id: "VALIDATION" for patient_id in val_patients})
    split_by_patient.update({patient_id: "TEST" for patient_id in test_patients})
    row_split = df["patient_id"].map(lambda patient_id: split_by_patient.get(int(patient_id)))
    train_df = df[row_split == "TRAIN"].reset_index(drop=True)
    val_df = df[row_split == "VALIDATION"].reset_index(drop=True)
    test_df = df[row_split == "TEST"].reset_index(drop=True)
    logging.info(
        "Split OK (attempt %s/%s, seed=%s): patients train/val/test=%s/%s/%s | "
        "images train/val/test=%s/%s/%s%s",
        attempt,
        constraints.max_tries,
        seed,
        len(train_patients),
        len(val_patients),
        len(test_patients),
        int(patient_df[patient_df["patient_id"].isin(sorted(train_patients))]["n_images"].sum()),
        int(patient_df[patient_df["patient_id"].isin(sorted(val_patients))]["n_images"].sum()),
        int(patient_df[patient_df["patient_id"].isin(sorted(test_patients))]["n_images"].sum()),
        f" | objective({score_split})={score:.6f}" if score is not None else "",
    )
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "split_seed": seed,
        "split_attempt": attempt,
        "objective_score": score,
        "objective_score_split": score_split,
        "constraints": {**asdict(constraints), **sizing_meta, "random_state": None},
    }


def _split_has_both_classes(patient_subset: set[int], patient_labels: dict[int, int]) -> bool:
    split_labels = {patient_labels[patient_id] for patient_id in patient_subset}
    return 0 in split_labels and 1 in split_labels


def _validation_supports_stage11(
    val_patients: set[int],
    *,
    constraints: SplitConstraints,
    dataset_has_both_classes: bool,
    patient_labels: dict[int, int],
) -> bool:
    if not constraints.enforce_stage11_validation_sizing:
        return True
    if len(val_patients) < constraints.min_validation_patients_for_ensemble:
        return False
    if not dataset_has_both_classes:
        return True

    positive_count = sum(1 for patient_id in val_patients if patient_labels[patient_id] == 1)
    negative_count = len(val_patients) - positive_count
    return (
        positive_count >= constraints.min_validation_positive_patients_for_ensemble
        and negative_count >= constraints.min_validation_negative_patients_for_ensemble
    )


def _select_test_patients_neutral(
    *,
    patient_ids: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    patient_labels: dict[int, int],
    n_test: int,
    n_val: int,
    constraints: SplitConstraints,
    rng: np.random.Generator,
    dataset_has_both_classes: bool,
) -> tuple[set[int], npt.NDArray[np.int64], npt.NDArray[np.int64], int, int]:
    for attempt in range(constraints.max_tries):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
            trainval_index, test_index = next(test_splitter.split(patient_ids, labels))
        except ValueError as error:
            raise ValueError(
                "Stratified split impossible at patient level for TEST "
                f"(test_size={n_test}). Details: {error}"
            ) from error

        trainval_ids = patient_ids[trainval_index]
        trainval_labels = labels[trainval_index]
        test_patients = {int(patient_id) for patient_id in patient_ids[test_index]}
        if constraints.require_both_classes_if_possible and dataset_has_both_classes:
            if not _split_has_both_classes(test_patients, patient_labels):
                continue
        try:
            validation_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_val,
                random_state=seed + 1,
            )
            next(validation_splitter.split(trainval_ids, trainval_labels))
        except ValueError:
            continue
        return test_patients, trainval_ids, trainval_labels, seed, attempt + 1

    raise ValueError(
        "Split search failed while selecting a neutral TEST split. "
        "Action: add patients, relax constraints, or disable class-coverage."
    )


def _select_train_val_patients(
    *,
    trainval_ids: npt.NDArray[np.int64],
    trainval_labels: npt.NDArray[np.int64],
    n_val: int,
    objective_enabled: bool,
    objective: ObjectiveConfig,
    patient_entropy_lookup: dict[int, float],
    patient_image_counts: dict[int, int],
    patient_labels: dict[int, int],
    test_patients: set[int],
    constraints: SplitConstraints,
    dataset_has_both_classes: bool,
    rng: np.random.Generator,
) -> tuple[set[int], set[int], int, int, float | None]:
    failure_counts = {
        "fail_val_strat": 0,
        "fail_overlap": 0,
        "fail_min_patients": 0,
        "fail_stage11_validation_sizing": 0,
        "fail_train_dominance": 0,
        "fail_class_coverage": 0,
    }
    best: tuple[set[int], set[int], int, int] | None = None
    best_score: float | None = None
    feasible_found = 0

    def image_count(patient_ids_subset: set[int]) -> int:
        return int(sum(patient_image_counts[patient_id] for patient_id in patient_ids_subset))

    test_images = image_count(test_patients)

    for attempt in range(constraints.max_tries):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            validation_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_val,
                random_state=seed,
            )
            train_index, val_index = next(validation_splitter.split(trainval_ids, trainval_labels))
        except ValueError:
            failure_counts["fail_val_strat"] += 1
            continue

        train_patients = {int(patient_id) for patient_id in trainval_ids[train_index]}
        val_patients = {int(patient_id) for patient_id in trainval_ids[val_index]}
        if train_patients & val_patients:
            failure_counts["fail_overlap"] += 1
            continue
        if (
            len(train_patients) < constraints.min_train_patients
            or len(val_patients) < constraints.min_val_patients
        ):
            failure_counts["fail_min_patients"] += 1
            continue
        if constraints.require_both_classes_if_possible and dataset_has_both_classes:
            if not (
                _split_has_both_classes(train_patients, patient_labels)
                and _split_has_both_classes(val_patients, patient_labels)
            ):
                failure_counts["fail_class_coverage"] += 1
                continue
        if not _validation_supports_stage11(
            val_patients,
            constraints=constraints,
            dataset_has_both_classes=dataset_has_both_classes,
            patient_labels=patient_labels,
        ):
            failure_counts["fail_stage11_validation_sizing"] += 1
            continue
        train_images = image_count(train_patients)
        val_images = image_count(val_patients)
        if constraints.require_train_image_dominance and not (
            train_images > val_images and train_images > test_images
        ):
            failure_counts["fail_train_dominance"] += 1
            continue

        feasible_found += 1
        if not objective_enabled:
            return train_patients, val_patients, seed, attempt + 1, None
        score_split = objective.score_split.upper()
        if score_split == "TRAIN":
            score_ids = sorted(train_patients)
        elif score_split == "VALIDATION":
            score_ids = sorted(val_patients)
        else:
            raise ValueError(f"Invalid objective.score_split: {objective.score_split}")
        score_values = [patient_entropy_lookup[patient_id] for patient_id in score_ids]
        score = (
            float(np.median(np.asarray(score_values, dtype=np.float64)))
            if score_values
            else float("-inf")
        )
        if best is None or best_score is None:
            best = (train_patients, val_patients, seed, attempt + 1)
            best_score = score
        elif objective.maximize and score > best_score:
            best = (train_patients, val_patients, seed, attempt + 1)
            best_score = score
        elif (not objective.maximize) and score < best_score:
            best = (train_patients, val_patients, seed, attempt + 1)
            best_score = score

    if best is not None:
        train_patients, val_patients, seed, attempt_no = best
        logging.info(
            "Best feasible TRAIN/VALIDATION split selected among %s feasible candidates "
            "(searched %s attempts). Best_score=%s",
            feasible_found,
            constraints.max_tries,
            f"{best_score:.6f}" if best_score is not None else "n/a",
        )
        return train_patients, val_patients, seed, attempt_no, best_score

    raise ValueError(
        "Split search failed while selecting TRAIN/VALIDATION under configured constraints.\n"
        f"Tried {constraints.max_tries} randomized stratified attempts.\n"
        "Failure breakdown:\n"
        f"  - VAL stratification failed: {failure_counts['fail_val_strat']}\n"
        f"  - Patient leakage overlap: {failure_counts['fail_overlap']}\n"
        f"  - Minimum patient counts failed: {failure_counts['fail_min_patients']}\n"
        "  - Stage 11 validation sizing failed: "
        f"{failure_counts['fail_stage11_validation_sizing']}\n"
        f"  - Class coverage failed: {failure_counts['fail_class_coverage']}\n"
        f"  - Train image dominance failed: {failure_counts['fail_train_dominance']}\n"
        "Action: add patients (especially minority class), relax constraints, "
        "disable Stage 11 validation sizing enforcement, or disable class-coverage.\n"
    )


def create_train_val_test_split_best(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
    optimize_training_set: bool = False,
    patient_entropy_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Search for a feasible patient-level split and optionally optimize it by entropy."""

    patient_df = build_patient_table(df)
    _, n_val, n_test, sizing_meta = decide_split_sizes(patient_df, constraints)
    patient_ids = patient_df["patient_id"].to_numpy(dtype=np.int64)
    labels = patient_df["patient_label"].astype(int).to_numpy(dtype=np.int64)
    dataset_has_both_classes = patient_df["patient_label"].nunique() >= 2
    rng = np.random.default_rng(random_state)
    patient_image_counts = dict(
        zip(
            patient_df["patient_id"].tolist(),
            patient_df["n_images"].astype(int).tolist(),
            strict=True,
        )
    )
    patient_labels = dict(
        zip(
            patient_df["patient_id"].tolist(),
            patient_df["patient_label"].astype(int).tolist(),
            strict=True,
        )
    )
    patient_entropy_lookup: dict[int, float] = {}
    if patient_entropy_df is not None:
        patient_entropy_lookup = dict(
            zip(
                patient_entropy_df["patient_id"].astype(int).tolist(),
                patient_entropy_df["patient_entropy_median"].astype(float).tolist(),
                strict=True,
            )
        )
    objective_enabled = optimize_training_set and objective.enable_objective
    if objective_enabled and patient_entropy_df is None:
        raise ValueError("Training-set optimization enabled but patient_entropy_df is None.")

    test_patients, trainval_ids, trainval_labels, test_seed, test_attempt = (
        _select_test_patients_neutral(
            patient_ids=patient_ids,
            labels=labels,
            patient_labels=patient_labels,
            n_test=n_test,
            n_val=n_val,
            constraints=constraints,
            rng=rng,
            dataset_has_both_classes=dataset_has_both_classes,
        )
    )
    train_patients, val_patients, trainval_seed, trainval_attempt, objective_score = (
        _select_train_val_patients(
            trainval_ids=trainval_ids,
            trainval_labels=trainval_labels,
            n_val=n_val,
            objective_enabled=objective_enabled,
            objective=objective,
            patient_entropy_lookup=patient_entropy_lookup,
            patient_image_counts=patient_image_counts,
            patient_labels=patient_labels,
            test_patients=test_patients,
            constraints=constraints,
            dataset_has_both_classes=dataset_has_both_classes,
            rng=rng,
        )
    )
    logging.info(
        "Neutral TEST split frozen before TRAIN/VALIDATION selection: "
        "test_attempt=%s | test_seed=%s | trainval_attempt=%s | trainval_seed=%s | "
        "optimize_training_set=%s",
        test_attempt,
        test_seed,
        trainval_attempt,
        trainval_seed,
        optimize_training_set,
    )
    return _build_split_return(
        df,
        patient_df,
        train_patients,
        val_patients,
        test_patients,
        trainval_seed,
        trainval_attempt,
        constraints,
        sizing_meta,
        objective_score,
        objective.score_split,
    )
