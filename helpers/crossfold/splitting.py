from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints

MIN_HELD_OUT_PATIENTS_PER_SPLIT = 20


class TrialWeightSuggester(Protocol):
    def suggest_float(self, name: str, low: float, high: float) -> float: ...


class SplitState(TypedDict):
    patient_ids: list[int]
    patient_count: int
    total_samples: int
    cancer_samples: int
    cancer_ratio: float
    non_cancer_samples: int


@dataclass(frozen=True)
class SplitReturnConfig:
    df: pd.DataFrame
    patient_df: pd.DataFrame
    train_patients: set[int]
    val_patients: set[int]
    test_patients: set[int]
    seed: int
    sizing_meta: dict[str, Any]
    loss: float


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under logger redirects."""

    return sys.__stderr__


class _StudyProgressCallback:
    """Advance the Stage 5 Optuna progress bar once per finished trial."""

    def __init__(self, progress_bar: tqdm[Any]) -> None:
        self._progress_bar = progress_bar
        self._completed_trials = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._progress_bar.update(1)
        trial_state = getattr(trial, "state", optuna.trial.TrialState.COMPLETE)
        if trial_state == optuna.trial.TrialState.COMPLETE:
            self._completed_trials += 1
        postfix: dict[str, str] = {"done": str(self._completed_trials)}
        if self._completed_trials > 0:
            best_value = float(study.best_value)
            if np.isfinite(best_value):
                postfix["best"] = f"{best_value:.4f}"
        self._progress_bar.set_postfix(postfix)


def _optuna_callbacks(progress_bar: tqdm[Any]) -> list[_StudyProgressCallback]:
    return [_StudyProgressCallback(progress_bar)]


def _set_optuna_warning_verbosity() -> int:
    previous_verbosity = int(optuna.logging.get_verbosity())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return previous_verbosity


def _restore_optuna_verbosity(previous_verbosity: int) -> None:
    optuna.logging.set_verbosity(previous_verbosity)


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build one patient-level row for split stratification and auditing."""

    return (
        df.groupby("patient_id")
        .agg(
            total_samples=("label", "size"),
            cancer_samples=("label", "sum"),
        )
        .assign(
            non_cancer_samples=lambda table: table["total_samples"] - table["cancer_samples"],
            patient_label=lambda table: (table["cancer_samples"] > 0).astype(int),
            n_images=lambda table: table["total_samples"],
        )
        .loc[
            :,
            [
                "total_samples",
                "cancer_samples",
                "non_cancer_samples",
                "patient_label",
                "n_images",
            ],
        ]
        .reset_index()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def compute_global_cancer_ratio(df: pd.DataFrame) -> float:
    """Compute the dataset-level cancer sample ratio used by split optimization."""

    total_samples = len(df)
    if total_samples <= 0:
        raise ValueError("Cannot compute a global cancer ratio from an empty dataset.")
    cancer_samples = int(df["label"].astype(int).sum())
    return cancer_samples / total_samples


def _ratio_after_assignment(
    *,
    current_total_samples: int,
    current_cancer_samples: int,
    added_total_samples: int,
    added_cancer_samples: int,
) -> float:
    new_total_samples = current_total_samples + added_total_samples
    if new_total_samples <= 0:
        return 0.0
    return (current_cancer_samples + added_cancer_samples) / new_total_samples


def allocate_patients_greedily(
    patient_df: pd.DataFrame,
    *,
    global_cancer_ratio: float,
    test_patient_count: int,
    validation_patient_count: int,
) -> dict[str, Any]:
    """Assign patients greedily using ratio deltas with TEST, VALIDATION, TRAIN tie-breaks."""

    split_state: dict[str, SplitState] = {
        "TEST": {
            "patient_ids": [],
            "patient_count": 0,
            "total_samples": 0,
            "cancer_samples": 0,
            "cancer_ratio": 0.0,
            "non_cancer_samples": 0,
        },
        "VALIDATION": {
            "patient_ids": [],
            "patient_count": 0,
            "total_samples": 0,
            "cancer_samples": 0,
            "cancer_ratio": 0.0,
            "non_cancer_samples": 0,
        },
        "TRAIN": {
            "patient_ids": [],
            "patient_count": 0,
            "total_samples": 0,
            "cancer_samples": 0,
            "cancer_ratio": 0.0,
            "non_cancer_samples": 0,
        },
    }
    constrained_splits = {
        "TEST": test_patient_count,
        "VALIDATION": validation_patient_count,
    }
    tie_break_order = ("TEST", "VALIDATION")

    for patient in patient_df.itertuples(index=False):
        patient_id = int(patient.patient_id)
        patient_total_samples = int(patient.total_samples)
        patient_cancer_samples = int(patient.cancer_samples)

        candidate_splits = [
            split_name
            for split_name in tie_break_order
            if split_state[split_name]["patient_count"] < constrained_splits[split_name]
        ]
        if not candidate_splits:
            candidate_splits = ["TRAIN"]

        best_split_name: str | None = None
        best_delta: float | None = None
        for split_name in candidate_splits:
            current_state = split_state[split_name]
            ratio = _ratio_after_assignment(
                current_total_samples=current_state["total_samples"],
                current_cancer_samples=current_state["cancer_samples"],
                added_total_samples=patient_total_samples,
                added_cancer_samples=patient_cancer_samples,
            )
            delta = abs(ratio - global_cancer_ratio)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_split_name = split_name

        if best_split_name is None:
            raise ValueError("Greedy allocation failed because no split had remaining capacity.")

        selected_state = split_state[best_split_name]
        selected_state["patient_ids"].append(patient_id)
        selected_state["patient_count"] += 1
        selected_state["total_samples"] += patient_total_samples
        selected_state["cancer_samples"] += patient_cancer_samples

    for _split_name, split_details in split_state.items():
        total_samples = split_details["total_samples"]
        cancer_samples = split_details["cancer_samples"]
        split_details["cancer_ratio"] = cancer_samples / total_samples if total_samples > 0 else 0.0
        split_details["non_cancer_samples"] = total_samples - cancer_samples

    return {
        "train_patients": [int(value) for value in split_state["TRAIN"]["patient_ids"]],
        "val_patients": [int(value) for value in split_state["VALIDATION"]["patient_ids"]],
        "test_patients": [int(value) for value in split_state["TEST"]["patient_ids"]],
        "split_stats": {
            "TRAIN": split_state["TRAIN"],
            "VALIDATION": split_state["VALIDATION"],
            "TEST": split_state["TEST"],
        },
    }


def order_patients_by_trial_weights(
    patient_df: pd.DataFrame,
    trial: TrialWeightSuggester,
) -> pd.DataFrame:
    """Return a deterministic patient ordering using one Optuna weight per patient."""

    weighted = patient_df.copy()
    weighted["trial_weight"] = [
        float(trial.suggest_float(f"weight_{int(patient_id)}", 0.0, 1.0))
        for patient_id in weighted["patient_id"].tolist()
    ]
    return weighted.sort_values(
        by=["trial_weight", "patient_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_split_ratio_loss(
    split_stats: dict[str, dict[str, Any]],
    *,
    global_cancer_ratio: float,
) -> float:
    """Compute the algorithm loss from final per-split cancer ratios."""

    return float(
        abs(float(split_stats["TRAIN"]["cancer_ratio"]) - global_cancer_ratio)
        + abs(float(split_stats["VALIDATION"]["cancer_ratio"]) - global_cancer_ratio)
        + abs(float(split_stats["TEST"]["cancer_ratio"]) - global_cancer_ratio)
    )


def validate_split_patient_counts(
    patient_df: pd.DataFrame,
    *,
    test_patient_count: int,
    validation_patient_count: int,
) -> None:
    """Validate the requested split capacities before optimization starts."""

    total_patients = len(patient_df)
    if test_patient_count < MIN_HELD_OUT_PATIENTS_PER_SPLIT:
        raise ValueError("TEST patient count must be at least 20.")
    if validation_patient_count < MIN_HELD_OUT_PATIENTS_PER_SPLIT:
        raise ValueError("VALIDATION patient count must be at least 20.")
    if test_patient_count + validation_patient_count >= total_patients:
        raise ValueError(
            "TEST and VALIDATION patient counts must leave at least one patient for TRAIN."
        )


def validate_split_class_guardrails(split_stats: dict[str, dict[str, Any]]) -> None:
    """Fail fast when TEST or VALIDATION ends up with only one class."""

    for split_name in ("TEST", "VALIDATION"):
        split_details = split_stats[split_name]
        cancer_samples = int(split_details["cancer_samples"])
        non_cancer_samples = int(split_details["non_cancer_samples"])
        if cancer_samples == 0 or non_cancer_samples == 0:
            raise ValueError(f"{split_name} split must contain both classes.")


def validate_final_split_patient_counts(
    split_stats: dict[str, dict[str, Any]],
    *,
    test_patient_count: int,
    validation_patient_count: int,
) -> None:
    """Fail fast when final TEST or VALIDATION patient counts miss requested quotas."""

    actual_test_count = int(split_stats["TEST"]["patient_count"])
    actual_validation_count = int(split_stats["VALIDATION"]["patient_count"])
    if actual_test_count != test_patient_count:
        raise ValueError(
            "TEST split patient count did not match the requested quota: "
            f"{actual_test_count} != {test_patient_count}."
        )
    if actual_validation_count != validation_patient_count:
        raise ValueError(
            "VALIDATION split patient count did not match the requested quota: "
            f"{actual_validation_count} != {validation_patient_count}."
        )


def compute_expected_class_counts(
    *,
    total_samples: int,
    global_cancer_ratio: float,
) -> dict[str, float]:
    """Compute expected cancer and non-cancer counts from the global ratio."""

    expected_cancer_samples = total_samples * global_cancer_ratio
    return {
        "cancer_samples": float(expected_cancer_samples),
        "non_cancer_samples": float(total_samples - expected_cancer_samples),
    }


def compute_split_chi_square_statistic(
    *,
    observed_cancer_samples: int,
    observed_non_cancer_samples: int,
    expected_cancer_samples: float,
    expected_non_cancer_samples: float,
) -> float:
    """Compute the raw chi-square statistic for one split using NumPy only."""

    observed = np.asarray(
        [observed_cancer_samples, observed_non_cancer_samples],
        dtype=np.float64,
    )
    expected = np.asarray(
        [expected_cancer_samples, expected_non_cancer_samples],
        dtype=np.float64,
    )
    if np.any(expected <= 0.0):
        raise ValueError("Expected class counts must be positive for chi-square verification.")
    return float(np.sum(((observed - expected) ** 2) / expected))


def compute_split_verification_metrics(
    split_stats: dict[str, dict[str, Any]],
    *,
    global_cancer_ratio: float,
) -> dict[str, dict[str, float]]:
    """Compute expected counts and raw chi-square verification for each split."""

    verification: dict[str, dict[str, float]] = {}
    for split_name in ("TRAIN", "VALIDATION", "TEST"):
        split_details = split_stats[split_name]
        total_samples = int(split_details["total_samples"])
        expected = compute_expected_class_counts(
            total_samples=total_samples,
            global_cancer_ratio=global_cancer_ratio,
        )
        verification[split_name] = {
            "expected_cancer_samples": expected["cancer_samples"],
            "expected_non_cancer_samples": expected["non_cancer_samples"],
            "chi2_stat": compute_split_chi_square_statistic(
                observed_cancer_samples=int(split_details["cancer_samples"]),
                observed_non_cancer_samples=int(split_details["non_cancer_samples"]),
                expected_cancer_samples=expected["cancer_samples"],
                expected_non_cancer_samples=expected["non_cancer_samples"],
            ),
        }
    return verification


def optimize_patient_split_with_optuna(
    patient_df: pd.DataFrame,
    *,
    global_cancer_ratio: float,
    test_patient_count: int,
    validation_patient_count: int,
    random_state: int,
    optuna_trials: int,
) -> dict[str, Any]:
    """Optimize patient ordering with Optuna, then allocate greedily once with best weights."""

    validate_split_patient_counts(
        patient_df,
        test_patient_count=test_patient_count,
        validation_patient_count=validation_patient_count,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        ordered_patients = order_patients_by_trial_weights(patient_df, trial)
        allocation = allocate_patients_greedily(
            ordered_patients,
            global_cancer_ratio=global_cancer_ratio,
            test_patient_count=test_patient_count,
            validation_patient_count=validation_patient_count,
        )
        loss = compute_split_ratio_loss(
            allocation["split_stats"],
            global_cancer_ratio=global_cancer_ratio,
        )
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    logging.info(
        "Running Stage 5 Optuna split search: patients=%s | trials=%s",
        len(patient_df),
        optuna_trials,
    )
    previous_verbosity = _set_optuna_warning_verbosity()
    try:
        with tqdm(
            total=optuna_trials,
            desc="Stage 5 Optuna",
            unit="trial",
            file=_progress_file(),
            leave=False,
        ) as progress_bar:
            study.optimize(
                objective,
                n_trials=optuna_trials,
                callbacks=_optuna_callbacks(progress_bar),
            )
    finally:
        _restore_optuna_verbosity(previous_verbosity)

    best_trial = study.best_trial
    ordered_patients = patient_df.copy()
    ordered_patients["trial_weight"] = [
        float(best_trial.params.get(f"weight_{int(patient_id)}", 0.0))
        for patient_id in ordered_patients["patient_id"].tolist()
    ]
    ordered_patients = ordered_patients.sort_values(
        by=["trial_weight", "patient_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    final_allocation = allocate_patients_greedily(
        ordered_patients,
        global_cancer_ratio=global_cancer_ratio,
        test_patient_count=test_patient_count,
        validation_patient_count=validation_patient_count,
    )
    validate_final_split_patient_counts(
        final_allocation["split_stats"],
        test_patient_count=test_patient_count,
        validation_patient_count=validation_patient_count,
    )
    validate_split_class_guardrails(final_allocation["split_stats"])
    final_loss = compute_split_ratio_loss(
        final_allocation["split_stats"],
        global_cancer_ratio=global_cancer_ratio,
    )
    verification = compute_split_verification_metrics(
        final_allocation["split_stats"],
        global_cancer_ratio=global_cancer_ratio,
    )
    return {
        **final_allocation,
        "loss": final_loss,
        "verification": verification,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "optuna_trials": optuna_trials,
        "random_state": random_state,
        "ordered_patient_ids": ordered_patients["patient_id"].astype(int).tolist(),
    }


def _build_split_return(config: SplitReturnConfig) -> dict[str, Any]:
    split_by_patient = {patient_id: "TRAIN" for patient_id in config.train_patients}
    split_by_patient.update({patient_id: "VALIDATION" for patient_id in config.val_patients})
    split_by_patient.update({patient_id: "TEST" for patient_id in config.test_patients})
    row_split = config.df["patient_id"].map(
        lambda patient_id: split_by_patient.get(int(patient_id))
    )
    train_df = config.df[row_split == "TRAIN"].reset_index(drop=True)
    val_df = config.df[row_split == "VALIDATION"].reset_index(drop=True)
    test_df = config.df[row_split == "TEST"].reset_index(drop=True)
    logging.info(
        "Split OK (seed=%s): patients train/val/test=%s/%s/%s | "
        "images train/val/test=%s/%s/%s | loss=%.6f",
        config.seed,
        len(config.train_patients),
        len(config.val_patients),
        len(config.test_patients),
        int(
            config.patient_df[config.patient_df["patient_id"].isin(sorted(config.train_patients))][
                "n_images"
            ].sum()
        ),
        int(
            config.patient_df[config.patient_df["patient_id"].isin(sorted(config.val_patients))][
                "n_images"
            ].sum()
        ),
        int(
            config.patient_df[config.patient_df["patient_id"].isin(sorted(config.test_patients))][
                "n_images"
            ].sum()
        ),
        config.loss,
    )
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_patients": sorted(config.train_patients),
        "val_patients": sorted(config.val_patients),
        "test_patients": sorted(config.test_patients),
        "split_seed": config.seed,
        "split_attempt": 1,
        "objective_score": config.loss,
        "constraints": {**config.sizing_meta, "random_state": None},
    }


def create_train_val_test_split_best(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
) -> dict[str, Any]:
    """Build the Stage 5 split using the Optuna-guided greedy patient allocator."""

    patient_df = build_patient_table(df)
    global_cancer_ratio = compute_global_cancer_ratio(df)
    optimization_result = optimize_patient_split_with_optuna(
        patient_df,
        global_cancer_ratio=global_cancer_ratio,
        test_patient_count=constraints.test_patient_count,
        validation_patient_count=constraints.validation_patient_count,
        random_state=random_state,
        optuna_trials=objective.optuna_trials,
    )
    train_patients = set(optimization_result["train_patients"])
    val_patients = set(optimization_result["val_patients"])
    test_patients = set(optimization_result["test_patients"])
    sizing_meta = {
        "N_patients": len(patient_df),
        "n_train": len(train_patients),
        "n_val": len(val_patients),
        "n_test": len(test_patients),
        "test_patient_count": constraints.test_patient_count,
        "validation_patient_count": constraints.validation_patient_count,
        "global_cancer_ratio": global_cancer_ratio,
        "optuna_trials": objective.optuna_trials,
    }
    logging.info(
        "Optuna greedy split selected: train/val/test patients=%s/%s/%s | loss=%.6f | trials=%s",
        len(train_patients),
        len(val_patients),
        len(test_patients),
        float(optimization_result["loss"]),
        objective.optuna_trials,
    )
    split_result = _build_split_return(
        SplitReturnConfig(
            df=df,
            patient_df=patient_df,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            seed=random_state,
            sizing_meta=sizing_meta,
            loss=float(optimization_result["loss"]),
        )
    )
    split_result["verification"] = optimization_result["verification"]
    split_result["best_value"] = optimization_result["best_value"]
    split_result["best_params"] = optimization_result["best_params"]
    split_result["ordered_patient_ids"] = optimization_result["ordered_patient_ids"]
    return split_result
