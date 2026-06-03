from helpers.ensemble_optimizer.splitting import build_holdout_split, build_indices_and_local_map

CALIBRATION_PATIENT_COUNT = 1
EMPTY_HOLDOUT_COUNT = 0
OPTIMIZATION_PATIENT_COUNT = 2
FIRST_LOCAL_STOP = 2
SECOND_LOCAL_START = 2
SECOND_LOCAL_STOP = 5


def test_build_holdout_split_uses_interleaved_strategy_for_small_dataset() -> None:
    patient_ids = ["pos1", "pos2", "neg1"]
    positive_patients = {"pos1", "pos2"}

    split = build_holdout_split(
        patient_ids,
        positive_patients,
        calibration_frac=0.25,
        holdout_frac=0.2,
        seed=24,
    )

    assert len(split.calibration_patients) == CALIBRATION_PATIENT_COUNT
    assert len(split.holdout_patients) == EMPTY_HOLDOUT_COUNT
    assert len(split.optimization_patients) == OPTIMIZATION_PATIENT_COUNT
    assert split.calibration_patients.isdisjoint(split.optimization_patients)
    assert split.holdout_patients.isdisjoint(split.optimization_patients)


def test_build_holdout_split_creates_three_way_split_for_larger_dataset() -> None:
    patient_ids = [f"pos{i}" for i in range(1, 7)] + [f"neg{i}" for i in range(1, 7)]
    positive_patients = {patient_id for patient_id in patient_ids if patient_id.startswith("pos")}

    split = build_holdout_split(
        patient_ids,
        positive_patients,
        calibration_frac=0.25,
        holdout_frac=0.25,
        seed=24,
    )

    assert split.optimization_patients
    assert split.calibration_patients
    assert split.holdout_patients
    assert split.optimization_patients.isdisjoint(split.calibration_patients)
    assert split.optimization_patients.isdisjoint(split.holdout_patients)
    assert split.calibration_patients.isdisjoint(split.holdout_patients)


def test_build_indices_and_local_map_preserves_patient_slices() -> None:
    patient_map = {"p1": [3, 4], "p2": [1], "p3": [7, 8, 9]}

    indices, local_map, ordered_patients = build_indices_and_local_map({"p3", "p1"}, patient_map)

    assert ordered_patients == ["p1", "p3"]
    assert indices.tolist() == [3, 4, 7, 8, 9]
    assert local_map["p1"].start == 0
    assert local_map["p1"].stop == FIRST_LOCAL_STOP
    assert local_map["p3"].start == SECOND_LOCAL_START
    assert local_map["p3"].stop == SECOND_LOCAL_STOP
