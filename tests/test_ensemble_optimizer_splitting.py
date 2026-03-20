from helpers.ensemble_optimizer.splitting import build_holdout_split, build_indices_and_local_map


def test_build_holdout_split_uses_interleaved_strategy_for_small_dataset() -> None:
    patient_ids = ["pos1", "pos2", "neg1"]
    positive_patients = {"pos1", "pos2"}

    split = build_holdout_split(
        patient_ids,
        positive_patients,
        holdout_frac=0.2,
        seed=24,
    )

    assert len(split.holdout_patients) == 1
    assert len(split.optimization_patients) == 2
    assert split.holdout_patients.isdisjoint(split.optimization_patients)


def test_build_indices_and_local_map_preserves_patient_slices() -> None:
    patient_map = {"p1": [3, 4], "p2": [1], "p3": [7, 8, 9]}

    indices, local_map, ordered_patients = build_indices_and_local_map({"p3", "p1"}, patient_map)

    assert ordered_patients == ["p1", "p3"]
    assert indices.tolist() == [3, 4, 7, 8, 9]
    assert local_map["p1"].start == 0
    assert local_map["p1"].stop == 2
    assert local_map["p3"].start == 2
    assert local_map["p3"].stop == 5
