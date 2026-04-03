import pandas as pd
import pytest

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints
from helpers.crossfold.splitting import create_train_val_test_split_best


def _dataset() -> pd.DataFrame:
    rows = []
    for patient_id, label in (
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
    ):
        rows.append(
            {
                "patient_id": patient_id,
                "image_path": f"/images/{patient_id}.png",
                "mask_path": f"/masks/{patient_id}.png",
                "label": label,
                "filename": f"PATIENT_{patient_id}_PATCH_001.png",
            }
        )
    return pd.DataFrame(rows)


def test_create_train_val_test_split_best_uses_entropy_objective() -> None:
    dataset = _dataset()
    patient_entropy_df = pd.DataFrame(
        [
            {"patient_id": 1, "patient_entropy_median": 10.0},
            {"patient_id": 2, "patient_entropy_median": 9.0},
            {"patient_id": 3, "patient_entropy_median": 1.0},
            {"patient_id": 4, "patient_entropy_median": 0.5},
            {"patient_id": 5, "patient_entropy_median": 8.0},
            {"patient_id": 6, "patient_entropy_median": 7.0},
            {"patient_id": 7, "patient_entropy_median": 0.4},
            {"patient_id": 8, "patient_entropy_median": 0.3},
        ]
    )

    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=0,
        constraints=SplitConstraints(
            min_test_patients=2,
            min_val_patients=2,
            min_train_patients=2,
            enforce_stage11_validation_sizing=False,
            test_ratio=0.25,
            val_ratio=0.25,
            require_train_image_dominance=False,
            require_both_classes_if_possible=False,
            max_tries=20,
            adaptive=True,
        ),
        objective=ObjectiveConfig(enable_objective=True, score_split="TRAIN", maximize=True),
        optimize_training_set=True,
        patient_entropy_df=patient_entropy_df,
    )

    assert split_data["objective_score"] is not None
    assert split_data["objective_score"] == pytest.approx(7.5)


def test_create_train_val_test_split_best_reports_impossible_stratification() -> None:
    dataset = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": "a",
                "mask_path": "a",
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
            },
            {
                "patient_id": 2,
                "image_path": "b",
                "mask_path": "b",
                "label": 0,
                "filename": "PATIENT_2_PATCH_001.png",
            },
            {
                "patient_id": 3,
                "image_path": "c",
                "mask_path": "c",
                "label": 0,
                "filename": "PATIENT_3_PATCH_001.png",
            },
        ]
    )

    try:
        create_train_val_test_split_best(
            df=dataset,
            random_state=0,
            constraints=SplitConstraints(
                min_test_patients=1,
                min_val_patients=1,
                min_train_patients=1,
                enforce_stage11_validation_sizing=False,
                test_ratio=0.34,
                val_ratio=0.33,
                require_train_image_dominance=False,
                require_both_classes_if_possible=False,
                max_tries=5,
                adaptive=False,
            ),
            objective=ObjectiveConfig(enable_objective=False),
            optimize_training_set=False,
            patient_entropy_df=None,
        )
    except ValueError as error:
        assert "Stratified split impossible" in str(error)
    else:
        raise AssertionError("Expected an impossible stratification error")


def test_training_set_optimization_keeps_test_patients_neutral() -> None:
    dataset = _dataset()
    patient_entropy_df = pd.DataFrame(
        [
            {"patient_id": 1, "patient_entropy_median": 10.0},
            {"patient_id": 2, "patient_entropy_median": 9.0},
            {"patient_id": 3, "patient_entropy_median": 1.0},
            {"patient_id": 4, "patient_entropy_median": 0.5},
            {"patient_id": 5, "patient_entropy_median": 8.0},
            {"patient_id": 6, "patient_entropy_median": 7.0},
            {"patient_id": 7, "patient_entropy_median": 0.4},
            {"patient_id": 8, "patient_entropy_median": 0.3},
        ]
    )
    constraints = SplitConstraints(
        min_test_patients=2,
        min_val_patients=2,
        min_train_patients=2,
        enforce_stage11_validation_sizing=False,
        test_ratio=0.25,
        val_ratio=0.25,
        require_train_image_dominance=False,
        require_both_classes_if_possible=False,
        max_tries=20,
        adaptive=True,
    )

    neutral = create_train_val_test_split_best(
        df=dataset,
        random_state=0,
        constraints=constraints,
        objective=ObjectiveConfig(enable_objective=True, score_split="TRAIN", maximize=True),
        optimize_training_set=False,
        patient_entropy_df=None,
    )
    optimized = create_train_val_test_split_best(
        df=dataset,
        random_state=0,
        constraints=constraints,
        objective=ObjectiveConfig(enable_objective=True, score_split="TRAIN", maximize=True),
        optimize_training_set=True,
        patient_entropy_df=patient_entropy_df,
    )

    assert neutral["test_patients"] == optimized["test_patients"]
    assert set(optimized["train_patients"]).isdisjoint(optimized["val_patients"])
    assert set(optimized["train_patients"]).isdisjoint(optimized["test_patients"])
    assert set(optimized["val_patients"]).isdisjoint(optimized["test_patients"])
    assert optimized["objective_score"] is not None
