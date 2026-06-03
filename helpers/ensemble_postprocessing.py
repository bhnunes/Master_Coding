from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from skimage.measure import label as connected_component_label

BINARY_MASK_NDIM = 2
BATCH_MASK_NDIM = 3


@dataclass(frozen=True)
class PostprocessingConfig:
    """Frozen hard-prediction filtering stored in the ensemble recipe."""

    min_component_area_px: int
    min_patient_positive_patches: int
    min_patient_positive_area_fraction: float
    min_component_area_fraction_patch: float

    def __post_init__(self) -> None:
        if self.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be greater than or equal to 0.")
        if self.min_patient_positive_patches < 1:
            raise ValueError("min_patient_positive_patches must be greater than or equal to 1.")
        if not 0.0 <= self.min_patient_positive_area_fraction <= 1.0:
            raise ValueError("min_patient_positive_area_fraction must be within [0, 1].")
        if not 0.0 <= self.min_component_area_fraction_patch <= 1.0:
            raise ValueError("min_component_area_fraction_patch must be within [0, 1].")


def postprocessing_config_to_payload(config: PostprocessingConfig) -> dict[str, float | int | str]:
    return {
        "method": "threshold_components_patient_suppression",
        "min_component_area_px": int(config.min_component_area_px),
        "min_patient_positive_patches": int(config.min_patient_positive_patches),
        "min_patient_positive_area_fraction": float(config.min_patient_positive_area_fraction),
        "min_component_area_fraction_patch": float(config.min_component_area_fraction_patch),
    }


def postprocessing_config_from_payload(payload: Mapping[str, Any]) -> PostprocessingConfig:
    method = str(payload.get("method", "")).strip()
    if method != "threshold_components_patient_suppression":
        raise ValueError(
            "Recipe postprocessing_config.method must be "
            "'threshold_components_patient_suppression'."
        )
    return PostprocessingConfig(
        min_component_area_px=int(payload["min_component_area_px"]),
        min_patient_positive_patches=int(payload["min_patient_positive_patches"]),
        min_patient_positive_area_fraction=float(payload["min_patient_positive_area_fraction"]),
        min_component_area_fraction_patch=float(payload["min_component_area_fraction_patch"]),
    )


def threshold_and_filter_components(
    probabilities: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    *,
    decision_threshold: float,
    min_component_area_px: int,
    min_component_area_fraction_patch: float = 0.0,
) -> npt.NDArray[np.uint8]:
    """Threshold probabilities and remove small connected components per patch."""

    binary = cast(npt.NDArray[np.uint8], (probabilities > decision_threshold).astype(np.uint8))
    effective_min_component_area_px = _effective_min_component_area_px(
        binary,
        min_component_area_px=min_component_area_px,
        min_component_area_fraction_patch=min_component_area_fraction_patch,
    )
    if effective_min_component_area_px <= 0 or binary.size == 0:
        return binary
    if binary.ndim == BINARY_MASK_NDIM:
        return _filter_components_2d(
            binary,
            min_component_area_px=effective_min_component_area_px,
        )
    if binary.ndim == BATCH_MASK_NDIM:
        filtered = np.zeros_like(binary, dtype=np.uint8)
        for index in range(binary.shape[0]):
            filtered[index] = _filter_components_2d(
                binary[index],
                min_component_area_px=effective_min_component_area_px,
            )
        return filtered
    raise ValueError(f"Expected 2D or 3D probability array, got shape {binary.shape}.")


def _effective_min_component_area_px(
    binary: npt.NDArray[np.uint8],
    *,
    min_component_area_px: int,
    min_component_area_fraction_patch: float,
) -> int:
    if min_component_area_px < 0:
        raise ValueError("min_component_area_px must be greater than or equal to 0.")
    if not 0.0 <= min_component_area_fraction_patch <= 1.0:
        raise ValueError("min_component_area_fraction_patch must be within [0, 1].")
    if min_component_area_fraction_patch <= 0.0:
        return int(min_component_area_px)
    if binary.ndim == BINARY_MASK_NDIM:
        patch_area = binary.shape[0] * binary.shape[1]
    elif binary.ndim == BATCH_MASK_NDIM:
        patch_area = binary.shape[1] * binary.shape[2]
    else:
        raise ValueError(f"Expected 2D or 3D probability array, got shape {binary.shape}.")
    fraction_area_px = int(math.ceil(patch_area * min_component_area_fraction_patch))
    return max(int(min_component_area_px), fraction_area_px)


def _filter_components_2d(
    binary_mask: npt.NDArray[np.uint8],
    *,
    min_component_area_px: int,
) -> npt.NDArray[np.uint8]:
    labels = cast(
        npt.NDArray[np.int32],
        connected_component_label(binary_mask.astype(bool), connectivity=1),  # type: ignore[no-untyped-call]
    )
    if labels.max() <= 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    component_areas = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(component_areas >= min_component_area_px)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    return cast(npt.NDArray[np.uint8], np.isin(labels, keep_labels).astype(np.uint8))


def confusion_counts_from_binary_masks(
    prediction: npt.NDArray[np.uint8],
    truth: npt.NDArray[np.uint8],
) -> dict[str, int]:
    pred_bool = prediction.astype(bool)
    truth_bool = truth.astype(bool)
    return {
        "tp": int(np.sum(pred_bool & truth_bool)),
        "fp": int(np.sum(pred_bool & ~truth_bool)),
        "fn": int(np.sum(~pred_bool & truth_bool)),
        "tn": int(np.sum(~pred_bool & ~truth_bool)),
    }


def truth_pixel_counts(truth: npt.NDArray[np.uint8]) -> dict[str, int]:
    truth_bool = truth.astype(bool)
    positive = int(np.sum(truth_bool))
    return {"positive": positive, "negative": int(truth_bool.size - positive)}


def count_positive_prediction_patches(predictions: npt.NDArray[np.uint8]) -> int:
    if predictions.ndim == BINARY_MASK_NDIM:
        return int(np.any(predictions))
    if predictions.ndim == BATCH_MASK_NDIM:
        return int(np.sum(np.any(predictions.astype(bool), axis=(1, 2))))
    raise ValueError(f"Expected 2D or 3D prediction array, got shape {predictions.shape}.")


def largest_component_area(predictions: npt.NDArray[np.uint8]) -> int:
    """Return the largest connected positive component across one patch or a patch batch."""

    if predictions.ndim == BINARY_MASK_NDIM:
        return _largest_component_area_2d(predictions)
    if predictions.ndim == BATCH_MASK_NDIM:
        return max(
            (
                _largest_component_area_2d(predictions[index])
                for index in range(predictions.shape[0])
            ),
            default=0,
        )
    raise ValueError(f"Expected 2D or 3D prediction array, got shape {predictions.shape}.")


def _largest_component_area_2d(binary_mask: npt.NDArray[np.uint8]) -> int:
    labels = cast(
        npt.NDArray[np.int32],
        connected_component_label(binary_mask.astype(bool), connectivity=1),  # type: ignore[no-untyped-call]
    )
    if labels.max() <= 0:
        return 0
    component_areas = np.bincount(labels.ravel())
    return int(component_areas[1:].max(initial=0))


def patient_positive_area_fractions_from_stats(
    stats_by_patient: Mapping[str, list[dict[str, int]]],
) -> dict[str, float]:
    """Compute per-patient predicted positive area fractions from pixel-count stats."""

    fractions: dict[str, float] = {}
    for patient_id, stats in stats_by_patient.items():
        pred_positive = sum(item["tp"] + item["fp"] for item in stats)
        total_pixels = sum(item["tp"] + item["fp"] + item["fn"] + item["tn"] for item in stats)
        fractions[patient_id] = float(pred_positive / total_pixels) if total_pixels > 0 else 0.0
    return fractions


def build_suppressed_patient_set(
    positive_patch_counts: Mapping[str, int],
    *,
    min_patient_positive_patches: int,
    positive_area_fractions: Mapping[str, float] | None = None,
    min_patient_positive_area_fraction: float = 0.0,
) -> set[str]:
    if min_patient_positive_patches < 1:
        raise ValueError("min_patient_positive_patches must be greater than or equal to 1.")
    if not 0.0 <= min_patient_positive_area_fraction <= 1.0:
        raise ValueError("min_patient_positive_area_fraction must be within [0, 1].")
    fractions = positive_area_fractions or {}
    patient_ids = set(positive_patch_counts) | set(fractions)
    return {
        patient_id
        for patient_id in patient_ids
        if positive_patch_counts.get(patient_id, 0) < min_patient_positive_patches
        or fractions.get(patient_id, 0.0) < min_patient_positive_area_fraction
    }


def apply_patient_positive_patch_suppression(
    stats_by_patient: Mapping[str, list[dict[str, int]]],
    truth_counts_by_patient: Mapping[str, dict[str, int]],
    positive_patch_counts: Mapping[str, int],
    *,
    min_patient_positive_patches: int,
    positive_area_fractions: Mapping[str, float] | None = None,
    min_patient_positive_area_fraction: float = 0.0,
) -> dict[str, list[dict[str, int]]]:
    suppressed_patients = build_suppressed_patient_set(
        positive_patch_counts,
        min_patient_positive_patches=min_patient_positive_patches,
        positive_area_fractions=positive_area_fractions,
        min_patient_positive_area_fraction=min_patient_positive_area_fraction,
    )
    output: dict[str, list[dict[str, int]]] = {}
    for patient_id, patient_stats in stats_by_patient.items():
        if patient_id not in suppressed_patients:
            output[patient_id] = list(patient_stats)
            continue
        truth_counts = truth_counts_by_patient[patient_id]
        output[patient_id] = [
            {
                "tp": 0,
                "fp": 0,
                "fn": int(truth_counts["positive"]),
                "tn": int(truth_counts["negative"]),
            }
        ]
    return output
