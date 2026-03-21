import pandas as pd

from helpers.crossfold.config import ObjectiveConfig, SplitConstraints
from helpers.crossfold.splitting import (
    build_patient_table,
    create_train_val_test_split_best,
    decide_split_sizes,
)


def _build_dataset() -> pd.DataFrame:
    rows = []
    for patient_id, label, count in (
        (1, 1, 5),
        (2, 1, 4),
        (3, 1, 3),
        (4, 1, 2),
        (5, 0, 6),
        (6, 0, 4),
        (7, 0, 3),
        (8, 0, 2),
    ):
        for index in range(count):
            rows.append(
                {
                    "patient_id": patient_id,
                    "image_path": f"/images/p{patient_id}_{index}.png",
                    "mask_path": f"/masks/p{patient_id}_{index}.png",
                    "label": label,
                    "filename": f"PATIENT_{patient_id}_PATCH_{index:03d}.png",
                }
            )
    return pd.DataFrame(rows)


def test_build_patient_table_uses_max_patch_label() -> None:
    dataset = pd.DataFrame(
        [
            {"patient_id": 1, "label": 0},
            {"patient_id": 1, "label": 1},
            {"patient_id": 2, "label": 0},
        ]
    )

    patient_table = build_patient_table(dataset)

    assert patient_table.to_dict("records") == [
        {"patient_id": 1, "patient_label": 1, "n_images": 2},
        {"patient_id": 2, "patient_label": 0, "n_images": 1},
    ]


def test_decide_split_sizes_returns_expected_counts() -> None:
    patient_table = build_patient_table(_build_dataset())

    n_train, n_val, n_test, meta = decide_split_sizes(
        patient_table,
        SplitConstraints(
            min_test_patients=1,
            min_val_patients=1,
            min_train_patients=1,
            enforce_stage11_validation_sizing=False,
            test_ratio=0.17,
            val_ratio=0.17,
            adaptive=True,
        ),
    )

    assert (n_train, n_val, n_test) == (6, 1, 1)
    assert meta["N_patients"] == 8


def test_create_train_val_test_split_best_keeps_patients_disjoint() -> None:
    dataset = _build_dataset()

    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=42,
        constraints=SplitConstraints(
            min_test_patients=1,
            min_val_patients=1,
            min_train_patients=1,
            enforce_stage11_validation_sizing=False,
            test_ratio=0.25,
            val_ratio=0.25,
            require_train_image_dominance=True,
            require_both_classes_if_possible=False,
            max_tries=50,
            adaptive=True,
        ),
        objective=ObjectiveConfig(enable_objective=False),
        patient_entropy_df=None,
    )

    train_patients = set(split_data["train_patients"])
    val_patients = set(split_data["val_patients"])
    test_patients = set(split_data["test_patients"])

    assert train_patients
    assert val_patients
    assert test_patients
    assert not (train_patients & val_patients)
    assert not (train_patients & test_patients)
    assert not (val_patients & test_patients)
    assert len(split_data["train_df"]) > len(split_data["val_df"])
    assert len(split_data["train_df"]) > len(split_data["test_df"])


def test_create_train_val_test_split_best_enforces_stage11_validation_minimums() -> None:
    dataset = _build_dataset()

    try:
        create_train_val_test_split_best(
            df=dataset,
            random_state=42,
            constraints=SplitConstraints(
                min_test_patients=1,
                min_val_patients=1,
                min_train_patients=1,
                enforce_stage11_validation_sizing=True,
                min_validation_patients_for_ensemble=6,
                min_validation_positive_patients_for_ensemble=3,
                min_validation_negative_patients_for_ensemble=3,
                test_ratio=0.25,
                val_ratio=0.25,
                require_train_image_dominance=False,
                require_both_classes_if_possible=False,
                max_tries=50,
                adaptive=True,
            ),
            objective=ObjectiveConfig(enable_objective=False),
            patient_entropy_df=None,
        )
    except ValueError as error:
        message = str(error)
        assert "Stage 11 validation sizing failed" in message
    else:
        raise AssertionError("Expected Stage 11 validation sizing enforcement to fail")
