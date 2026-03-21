from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HoldoutSplit:
    optimization_patients: set[str]
    calibration_patients: set[str]
    holdout_patients: set[str]
    positive_patients: set[str]
    negative_patients: set[str]


def pick_n(total: int, frac: float) -> int:
    if total <= 1:
        return 0
    return min(max(1, int(round(total * frac))), total - 1)


def _allocate_subset_counts(
    total: int,
    *,
    calibration_frac: float,
    holdout_frac: float,
) -> tuple[int, int]:
    if total <= 1:
        return 0, 0
    if total == 2:
        return 1, 0

    calibration_count = pick_n(total, calibration_frac)
    remaining_after_calibration = total - calibration_count
    if remaining_after_calibration <= 1:
        return calibration_count, 0

    holdout_count = min(int(round(total * holdout_frac)), total - calibration_count - 1)
    if holdout_frac > 0 and total >= 4:
        holdout_count = max(1, holdout_count)
    holdout_count = max(0, holdout_count)
    return calibration_count, holdout_count


def build_holdout_split(
    patient_ids: Sequence[str],
    positive_patients: set[str],
    *,
    calibration_frac: float,
    holdout_frac: float,
    seed: int,
) -> HoldoutSplit:
    all_patients = list(dict.fromkeys(patient_ids))
    negative_patients = set(all_patients) - positive_patients
    positive = list(positive_patients)
    negative = list(negative_patients)
    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    if len(all_patients) <= 10:
        mixed: list[str] = []
        max_len = max(len(positive), len(negative))
        for index in range(max_len):
            if index < len(positive):
                mixed.append(positive[index])
            if index < len(negative):
                mixed.append(negative[index])
        calibration_patients = {mixed[-1]} if mixed else set()
        holdout_patients = {mixed[-2]} if len(mixed) >= 4 else set()
        optimization_patients = set(mixed) - calibration_patients - holdout_patients
    else:
        n_calibration_positive, n_holdout_positive = _allocate_subset_counts(
            len(positive),
            calibration_frac=calibration_frac,
            holdout_frac=holdout_frac,
        )
        n_calibration_negative, n_holdout_negative = _allocate_subset_counts(
            len(negative),
            calibration_frac=calibration_frac,
            holdout_frac=holdout_frac,
        )
        calibration_patients = set(
            positive[:n_calibration_positive] + negative[:n_calibration_negative]
        )
        holdout_patients = set(
            positive[n_calibration_positive : n_calibration_positive + n_holdout_positive]
            + negative[n_calibration_negative : n_calibration_negative + n_holdout_negative]
        )
        optimization_patients = set(positive[n_calibration_positive + n_holdout_positive :]) | set(
            negative[n_calibration_negative + n_holdout_negative :]
        )

    if len(all_patients) >= 3 and not calibration_patients:
        raise RuntimeError(
            "Validation split produced 0 calibration patients. Increase "
            "ENSEMBLE_OPT_VAL_CALIBRATION_FRAC or use a different split policy."
        )
    if len(all_patients) >= 4 and not optimization_patients:
        raise RuntimeError("Validation split produced 0 optimization patients.")

    return HoldoutSplit(
        optimization_patients=optimization_patients,
        calibration_patients=calibration_patients,
        holdout_patients=holdout_patients,
        positive_patients=positive_patients,
        negative_patients=negative_patients,
    )


def build_indices_and_local_map(
    patients: set[str],
    patient_map: dict[str, list[int]],
) -> tuple[npt.NDArray[np.int64], dict[str, slice], list[str]]:
    ordered_patients = sorted(patients)
    indices: list[int] = []
    local_map: dict[str, slice] = {}
    start = 0
    for patient in ordered_patients:
        frames = patient_map[patient]
        indices.extend(frames)
        stop = start + len(frames)
        local_map[patient] = slice(start, stop)
        start = stop
    return np.asarray(indices, dtype=np.int64), local_map, ordered_patients
