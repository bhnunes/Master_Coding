from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints
from helpers.crossfold.entropy import score_split_by_patient_entropy_median


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
    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)
    logging.info(
        "Split OK (attempt %s/%s, seed=%s): patients train/val/test=%s/%s/%s | "
        "images train/val/test=%s/%s/%s%s",
        attempt,
        constraints.max_tries,
        seed,
        len(train_patients),
        len(val_patients),
        len(test_patients),
        int(patient_df[patient_df["patient_id"].isin(train_patients)]["n_images"].sum()),
        int(patient_df[patient_df["patient_id"].isin(val_patients)]["n_images"].sum()),
        int(patient_df[patient_df["patient_id"].isin(test_patients)]["n_images"].sum()),
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


def create_train_val_test_split_best(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
    patient_entropy_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Search for a feasible patient-level split and optionally optimize it by entropy."""

    patient_df = build_patient_table(df)
    _, _, n_test, sizing_meta = decide_split_sizes(patient_df, constraints)
    patient_ids = patient_df["patient_id"].to_numpy()
    labels = patient_df["patient_label"].astype(int).to_numpy()
    dataset_has_both_classes = patient_df["patient_label"].nunique() >= 2
    rng = np.random.default_rng(random_state)

    def image_count(patient_ids_subset: set[int]) -> int:
        return int(patient_df[patient_df["patient_id"].isin(patient_ids_subset)]["n_images"].sum())

    failure_counts = {
        "fail_val_strat": 0,
        "fail_overlap": 0,
        "fail_min_patients": 0,
        "fail_train_dominance": 0,
        "fail_class_coverage": 0,
    }
    best: tuple[set[int], set[int], set[int], int, int] | None = None
    best_score: float | None = None
    feasible_found = 0

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
        try:
            validation_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=sizing_meta["n_val"],
                random_state=seed + 1,
            )
            train_index, val_index = next(validation_splitter.split(trainval_ids, trainval_labels))
        except ValueError:
            failure_counts["fail_val_strat"] += 1
            continue

        train_patients = set(trainval_ids[train_index])
        val_patients = set(trainval_ids[val_index])
        test_patients = set(patient_ids[test_index])
        if (
            (train_patients & val_patients)
            or (train_patients & test_patients)
            or (val_patients & test_patients)
        ):
            failure_counts["fail_overlap"] += 1
            continue
        if (
            len(train_patients) < sizing_meta["min_train"]
            or len(val_patients) < sizing_meta["min_val"]
            or len(test_patients) < sizing_meta["min_test"]
        ):
            failure_counts["fail_min_patients"] += 1
            continue
        if constraints.require_both_classes_if_possible and dataset_has_both_classes:

            def split_has_both(patient_subset: set[int]) -> bool:
                split_labels = set(
                    patient_df[patient_df["patient_id"].isin(patient_subset)][
                        "patient_label"
                    ].tolist()
                )
                return 0 in split_labels and 1 in split_labels

            if not (
                split_has_both(train_patients)
                and split_has_both(val_patients)
                and split_has_both(test_patients)
            ):
                failure_counts["fail_class_coverage"] += 1
                continue
        train_images = image_count(train_patients)
        val_images = image_count(val_patients)
        test_images = image_count(test_patients)
        if constraints.require_train_image_dominance and not (
            train_images > val_images and train_images > test_images
        ):
            failure_counts["fail_train_dominance"] += 1
            continue

        feasible_found += 1
        if not objective.enable_objective:
            return _build_split_return(
                df,
                patient_df,
                train_patients,
                val_patients,
                test_patients,
                seed,
                attempt + 1,
                constraints,
                sizing_meta,
                None,
                objective.score_split,
            )
        if patient_entropy_df is None:
            raise ValueError("Objective enabled but patient_entropy_df is None.")
        score_split = objective.score_split.upper()
        if score_split == "TRAIN":
            score_ids = sorted(train_patients)
        elif score_split == "VALIDATION":
            score_ids = sorted(val_patients)
        elif score_split == "TEST":
            score_ids = sorted(test_patients)
        else:
            raise ValueError(f"Invalid objective.score_split: {objective.score_split}")
        score = score_split_by_patient_entropy_median(patient_entropy_df, score_ids)
        if best is None or best_score is None:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
        elif objective.maximize and score > best_score:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
        elif (not objective.maximize) and score < best_score:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score

    if best is not None and best_score is not None:
        train_patients, val_patients, test_patients, seed, attempt_no = best
        logging.info(
            "Best feasible split selected among %s feasible candidates "
            "(searched %s attempts). Best_score=%.6f",
            feasible_found,
            constraints.max_tries,
            best_score,
        )
        return _build_split_return(
            df,
            patient_df,
            train_patients,
            val_patients,
            test_patients,
            seed,
            attempt_no,
            constraints,
            sizing_meta,
            float(best_score),
            objective.score_split,
        )

    raise ValueError(
        "Split search failed under configured constraints.\n"
        f"Tried {constraints.max_tries} randomized stratified attempts "
        f"(random_state={random_state}).\n"
        "Failure breakdown:\n"
        f"  - VAL stratification failed: {failure_counts['fail_val_strat']}\n"
        f"  - Patient leakage overlap: {failure_counts['fail_overlap']}\n"
        f"  - Minimum patient counts failed: {failure_counts['fail_min_patients']}\n"
        f"  - Class coverage failed: {failure_counts['fail_class_coverage']}\n"
        f"  - Train image dominance failed: {failure_counts['fail_train_dominance']}\n"
        "Action: add patients (especially minority class), relax constraints, "
        "or disable class-coverage.\n"
    )
