from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HoldoutSplit:
    optimization_patients: set[str]
    holdout_patients: set[str]
    positive_patients: set[str]
    negative_patients: set[str]


def pick_n(total: int, frac: float) -> int:
    if total <= 1:
        return 0
    return min(max(1, int(round(total * frac))), total - 1)


def build_holdout_split(
    patient_ids: Sequence[str],
    positive_patients: set[str],
    *,
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
        holdout_patients = {mixed[-1]} if mixed else set()
        optimization_patients = set(mixed[:-1])
    else:
        n_holdout_positive = pick_n(len(positive), holdout_frac) if positive else 0
        n_holdout_negative = pick_n(len(negative), holdout_frac) if negative else 0
        holdout_patients = set(positive[:n_holdout_positive] + negative[:n_holdout_negative])
        optimization_patients = set(positive[n_holdout_positive:] + negative[n_holdout_negative:])

    if len(all_patients) >= 2 and not holdout_patients:
        raise RuntimeError(
            "Holdout split produced 0 patients. Increase ENSEMBLE_OPT_VAL_HOLDOUT_FRAC or "
            "use a different split policy for small-N validation."
        )

    return HoldoutSplit(
        optimization_patients=optimization_patients,
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
