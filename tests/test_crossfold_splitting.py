import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints
from helpers.crossfold.splitting import (
    allocate_patients_greedily,
    build_patient_table,
    compute_expected_class_counts,
    compute_global_cancer_ratio,
    compute_split_chi_square_statistic,
    compute_split_ratio_loss,
    compute_split_verification_metrics,
    create_train_val_test_split_best,
    optimize_patient_split_with_optuna,
    order_patients_by_trial_weights,
    validate_final_split_patient_counts,
    validate_split_class_guardrails,
    validate_split_patient_counts,
)


def test_build_patient_table_profiles_patient_sample_counts() -> None:
    dataset = pd.DataFrame(
        [
            {"patient_id": 2, "label": 0},
            {"patient_id": 1, "label": 0},
            {"patient_id": 1, "label": 1},
            {"patient_id": 2, "label": 0},
        ]
    )

    patient_table = build_patient_table(dataset)

    assert patient_table.to_dict("records") == [
        {
            "patient_id": 1,
            "total_samples": 2,
            "cancer_samples": 1,
            "non_cancer_samples": 1,
            "patient_label": 1,
            "n_images": 2,
        },
        {
            "patient_id": 2,
            "total_samples": 2,
            "cancer_samples": 0,
            "non_cancer_samples": 2,
            "patient_label": 0,
            "n_images": 2,
        },
    ]


def test_compute_global_cancer_ratio_uses_sample_level_counts() -> None:
    dataset = pd.DataFrame(
        [
            {"patient_id": 1, "label": 1},
            {"patient_id": 1, "label": 1},
            {"patient_id": 2, "label": 0},
            {"patient_id": 3, "label": 0},
            {"patient_id": 3, "label": 1},
        ]
    )

    assert compute_global_cancer_ratio(dataset) == pytest.approx(3 / 5)


def test_compute_global_cancer_ratio_rejects_empty_dataset() -> None:
    dataset = pd.DataFrame(columns=["patient_id", "label"])

    with pytest.raises(ValueError, match="empty dataset"):
        compute_global_cancer_ratio(dataset)


def test_allocate_patients_greedily_respects_capacities_and_assigns_remainder_to_train() -> None:
    patient_table = pd.DataFrame(
        [
            {"patient_id": 1, "total_samples": 4, "cancer_samples": 4},
            {"patient_id": 2, "total_samples": 4, "cancer_samples": 0},
            {"patient_id": 3, "total_samples": 4, "cancer_samples": 2},
            {"patient_id": 4, "total_samples": 4, "cancer_samples": 2},
            {"patient_id": 5, "total_samples": 4, "cancer_samples": 2},
        ]
    ).assign(non_cancer_samples=lambda table: table["total_samples"] - table["cancer_samples"])

    allocation = allocate_patients_greedily(
        patient_table,
        global_cancer_ratio=0.5,
        test_patient_count=1,
        validation_patient_count=1,
    )

    assert allocation["test_patients"] == [1]
    assert allocation["val_patients"] == [2]
    assert allocation["train_patients"] == [3, 4, 5]
    assert allocation["split_stats"]["TEST"]["patient_count"] == 1
    assert allocation["split_stats"]["VALIDATION"]["patient_count"] == 1
    assert allocation["split_stats"]["TRAIN"]["patient_count"] == 3


def test_allocate_patients_greedily_uses_test_then_validation_then_train_tie_break() -> None:
    patient_table = pd.DataFrame(
        [
            {"patient_id": 11, "total_samples": 2, "cancer_samples": 1},
            {"patient_id": 12, "total_samples": 2, "cancer_samples": 1},
            {"patient_id": 13, "total_samples": 2, "cancer_samples": 1},
        ]
    ).assign(non_cancer_samples=lambda table: table["total_samples"] - table["cancer_samples"])

    allocation = allocate_patients_greedily(
        patient_table,
        global_cancer_ratio=0.5,
        test_patient_count=1,
        validation_patient_count=1,
    )

    assert allocation["test_patients"] == [11]
    assert allocation["val_patients"] == [12]
    assert allocation["train_patients"] == [13]


def test_allocate_patients_greedily_fills_validation_quota_before_train() -> None:
    patient_table = pd.DataFrame(
        [
            {"patient_id": 1, "total_samples": 10, "cancer_samples": 9},
            {"patient_id": 2, "total_samples": 10, "cancer_samples": 9},
            {"patient_id": 3, "total_samples": 10, "cancer_samples": 5},
            {"patient_id": 4, "total_samples": 10, "cancer_samples": 5},
        ]
    ).assign(non_cancer_samples=lambda table: table["total_samples"] - table["cancer_samples"])

    allocation = allocate_patients_greedily(
        patient_table,
        global_cancer_ratio=0.5,
        test_patient_count=1,
        validation_patient_count=2,
    )

    assert allocation["test_patients"] == [1]
    assert allocation["val_patients"] == [2, 3]
    assert allocation["train_patients"] == [4]
    assert allocation["split_stats"]["VALIDATION"]["patient_count"] == 2


def test_order_patients_by_trial_weights_uses_trial_suggestions_and_patient_id_tie_break() -> None:
    patient_table = pd.DataFrame(
        [
            {"patient_id": 3, "total_samples": 2, "cancer_samples": 1, "non_cancer_samples": 1},
            {"patient_id": 1, "total_samples": 2, "cancer_samples": 1, "non_cancer_samples": 1},
            {"patient_id": 2, "total_samples": 2, "cancer_samples": 1, "non_cancer_samples": 1},
        ]
    )

    class TrialStub:
        def __init__(self) -> None:
            self.params = {"weight_3": 0.4, "weight_1": 0.2, "weight_2": 0.2}

        def suggest_float(self, name: str, low: float, high: float) -> float:
            assert (low, high) == (0.0, 1.0)
            return float(self.params[name])

    ordered = order_patients_by_trial_weights(patient_table, TrialStub())

    assert ordered["patient_id"].tolist() == [1, 2, 3]


def test_compute_split_ratio_loss_sums_absolute_ratio_deltas() -> None:
    split_stats = {
        "TRAIN": {"cancer_ratio": 0.40},
        "VALIDATION": {"cancer_ratio": 0.55},
        "TEST": {"cancer_ratio": 0.75},
    }

    assert compute_split_ratio_loss(split_stats, global_cancer_ratio=0.50) == pytest.approx(0.40)


def test_optimize_patient_split_with_optuna_uses_best_trial_weights(
    monkeypatch: MonkeyPatch,
) -> None:
    patient_table = pd.DataFrame(
        [
            {"patient_id": 1, "total_samples": 4, "cancer_samples": 4, "non_cancer_samples": 0},
            {"patient_id": 2, "total_samples": 4, "cancer_samples": 0, "non_cancer_samples": 4},
            {"patient_id": 3, "total_samples": 4, "cancer_samples": 2, "non_cancer_samples": 2},
            {"patient_id": 4, "total_samples": 4, "cancer_samples": 2, "non_cancer_samples": 2},
            {"patient_id": 5, "total_samples": 4, "cancer_samples": 2, "non_cancer_samples": 2},
        ]
    )
    trial_params = [
        {"weight_1": 0.1, "weight_2": 0.2, "weight_3": 0.3, "weight_4": 0.4, "weight_5": 0.5},
        {"weight_1": 0.9, "weight_2": 0.8, "weight_3": 0.1, "weight_4": 0.2, "weight_5": 0.3},
    ]

    class TrialStub:
        def __init__(self, params: dict[str, float]) -> None:
            self.params = params

        def suggest_float(self, name: str, low: float, high: float) -> float:
            assert (low, high) == (0.0, 1.0)
            return float(self.params[name])

    class StudyStub:
        def __init__(self, params_per_trial: list[dict[str, float]]) -> None:
            self._trials = [TrialStub(params) for params in params_per_trial]
            self.best_trial: TrialStub | None = None
            self.best_value: float | None = None

        def optimize(self, objective, n_trials: int) -> None:  # type: ignore[no-untyped-def]
            assert n_trials == 2
            for trial in self._trials:
                value = float(objective(trial))
                if self.best_value is None or value < self.best_value:
                    self.best_value = value
                    self.best_trial = trial

    study = StudyStub(trial_params)
    monkeypatch.setattr(
        "helpers.crossfold.splitting.optuna.create_study",
        lambda direction, sampler: study,
    )
    monkeypatch.setattr(
        "helpers.crossfold.splitting.validate_split_patient_counts",
        lambda patient_df, *, test_patient_count, validation_patient_count: None,
    )

    result = optimize_patient_split_with_optuna(
        patient_table,
        global_cancer_ratio=0.5,
        test_patient_count=1,
        validation_patient_count=1,
        random_state=42,
        optuna_trials=2,
    )

    assert result["best_value"] == pytest.approx(result["loss"])
    assert result["best_params"] == trial_params[1]
    assert result["ordered_patient_ids"] == [3, 4, 5, 2, 1]
    assert result["test_patients"] == [3]
    assert result["val_patients"] == [4]
    assert result["train_patients"] == [5, 2, 1]


def test_validate_split_patient_counts_requires_minimum_test_size() -> None:
    patient_table = pd.DataFrame({"patient_id": list(range(1, 50))})

    with pytest.raises(ValueError, match="TEST patient count must be at least 20"):
        validate_split_patient_counts(
            patient_table,
            test_patient_count=19,
            validation_patient_count=20,
        )


def test_validate_split_patient_counts_requires_minimum_validation_size() -> None:
    patient_table = pd.DataFrame({"patient_id": list(range(1, 50))})

    with pytest.raises(ValueError, match="VALIDATION patient count must be at least 20"):
        validate_split_patient_counts(
            patient_table,
            test_patient_count=20,
            validation_patient_count=19,
        )


def test_validate_split_patient_counts_requires_non_empty_train() -> None:
    patient_table = pd.DataFrame({"patient_id": list(range(1, 41))})

    with pytest.raises(ValueError, match="leave at least one patient for TRAIN"):
        validate_split_patient_counts(
            patient_table,
            test_patient_count=20,
            validation_patient_count=20,
        )


def test_validate_split_class_guardrails_rejects_one_class_test_split() -> None:
    split_stats = {
        "TRAIN": {"cancer_samples": 10, "non_cancer_samples": 10},
        "VALIDATION": {"cancer_samples": 3, "non_cancer_samples": 2},
        "TEST": {"cancer_samples": 5, "non_cancer_samples": 0},
    }

    with pytest.raises(ValueError, match="TEST split must contain both classes"):
        validate_split_class_guardrails(split_stats)


def test_validate_final_split_patient_counts_rejects_mismatched_quota() -> None:
    split_stats = {
        "TRAIN": {"patient_count": 10},
        "VALIDATION": {"patient_count": 14},
        "TEST": {"patient_count": 22},
    }

    with pytest.raises(ValueError, match="VALIDATION split patient count did not match"):
        validate_final_split_patient_counts(
            split_stats,
            test_patient_count=22,
            validation_patient_count=30,
        )


def test_validate_split_class_guardrails_rejects_one_class_validation_split() -> None:
    split_stats = {
        "TRAIN": {"cancer_samples": 10, "non_cancer_samples": 10},
        "VALIDATION": {"cancer_samples": 0, "non_cancer_samples": 5},
        "TEST": {"cancer_samples": 3, "non_cancer_samples": 2},
    }

    with pytest.raises(ValueError, match="VALIDATION split must contain both classes"):
        validate_split_class_guardrails(split_stats)


def test_compute_expected_class_counts_uses_global_ratio() -> None:
    expected = compute_expected_class_counts(total_samples=20, global_cancer_ratio=0.35)

    assert expected == {
        "cancer_samples": pytest.approx(7.0),
        "non_cancer_samples": pytest.approx(13.0),
    }


def test_compute_split_chi_square_statistic_matches_numpy_formula() -> None:
    chi2_stat = compute_split_chi_square_statistic(
        observed_cancer_samples=6,
        observed_non_cancer_samples=4,
        expected_cancer_samples=5.0,
        expected_non_cancer_samples=5.0,
    )

    assert chi2_stat == pytest.approx(0.4)


def test_compute_split_chi_square_statistic_rejects_non_positive_expected_counts() -> None:
    with pytest.raises(ValueError, match="Expected class counts must be positive"):
        compute_split_chi_square_statistic(
            observed_cancer_samples=0,
            observed_non_cancer_samples=10,
            expected_cancer_samples=0.0,
            expected_non_cancer_samples=10.0,
        )


def test_compute_split_verification_metrics_reports_expected_counts_and_chi_square() -> None:
    split_stats = {
        "TRAIN": {"total_samples": 10, "cancer_samples": 6, "non_cancer_samples": 4},
        "VALIDATION": {"total_samples": 8, "cancer_samples": 4, "non_cancer_samples": 4},
        "TEST": {"total_samples": 6, "cancer_samples": 2, "non_cancer_samples": 4},
    }

    verification = compute_split_verification_metrics(split_stats, global_cancer_ratio=0.5)

    assert verification["TRAIN"]["expected_cancer_samples"] == pytest.approx(5.0)
    assert verification["TRAIN"]["expected_non_cancer_samples"] == pytest.approx(5.0)
    assert verification["TRAIN"]["chi2_stat"] == pytest.approx(0.4)
    assert verification["VALIDATION"]["chi2_stat"] == pytest.approx(0.0)
    assert verification["TEST"]["expected_cancer_samples"] == pytest.approx(3.0)
    assert verification["TEST"]["expected_non_cancer_samples"] == pytest.approx(3.0)
    assert verification["TEST"]["chi2_stat"] == pytest.approx(2 / 3)


def _balanced_dataset_with_patient_count(patient_count: int) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for patient_id in range(1, patient_count + 1):
        rows.extend(
            [
                {
                    "patient_id": patient_id,
                    "label": 1,
                    "image_path": f"/images/{patient_id}_pos.png",
                    "mask_path": f"/masks/{patient_id}_pos.png",
                    "filename": f"PATIENT_{patient_id}_PATCH_001.png",
                },
                {
                    "patient_id": patient_id,
                    "label": 0,
                    "image_path": f"/images/{patient_id}_neg.png",
                    "mask_path": f"/masks/{patient_id}_neg.png",
                    "filename": f"PATIENT_{patient_id}_PATCH_002.png",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_create_train_val_test_split_best_respects_exact_patient_capacities() -> None:
    dataset = _balanced_dataset_with_patient_count(43)

    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=42,
        constraints=SplitConstraints(test_patient_count=20, validation_patient_count=20),
        objective=ObjectiveConfig(optuna_trials=3),
    )

    assert len(split_data["test_patients"]) == 20
    assert len(split_data["val_patients"]) == 20
    assert len(split_data["train_patients"]) == 3
    assert split_data["constraints"]["n_test"] == 20
    assert split_data["constraints"]["n_val"] == 20
    assert split_data["constraints"]["n_train"] == 3
    assert split_data["train_df"]["patient_id"].nunique() == 3
    assert split_data["val_df"]["patient_id"].nunique() == 20
    assert split_data["test_df"]["patient_id"].nunique() == 20
    assert set(split_data["train_patients"]).isdisjoint(split_data["val_patients"])
    assert set(split_data["train_patients"]).isdisjoint(split_data["test_patients"])
    assert set(split_data["val_patients"]).isdisjoint(split_data["test_patients"])
    assert split_data["verification"].keys() == {"TRAIN", "VALIDATION", "TEST"}


def test_create_train_val_test_split_best_is_deterministic_for_same_seed() -> None:
    dataset = _balanced_dataset_with_patient_count(43)
    constraints = SplitConstraints(test_patient_count=20, validation_patient_count=20)
    objective = ObjectiveConfig(optuna_trials=3)

    first = create_train_val_test_split_best(
        df=dataset,
        random_state=17,
        constraints=constraints,
        objective=objective,
    )
    second = create_train_val_test_split_best(
        df=dataset,
        random_state=17,
        constraints=constraints,
        objective=objective,
    )

    assert first["ordered_patient_ids"] == second["ordered_patient_ids"]
    assert first["train_patients"] == second["train_patients"]
    assert first["val_patients"] == second["val_patients"]
    assert first["test_patients"] == second["test_patients"]
    assert first["objective_score"] == pytest.approx(second["objective_score"])
    assert first["verification"] == second["verification"]
