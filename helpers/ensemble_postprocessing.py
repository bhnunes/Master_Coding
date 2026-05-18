from __future__ import annotations

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

    def __post_init__(self) -> None:
        if self.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be greater than or equal to 0.")
        if self.min_patient_positive_patches < 1:
            raise ValueError("min_patient_positive_patches must be greater than or equal to 1.")


def postprocessing_config_to_payload(config: PostprocessingConfig) -> dict[str, int | str]:
    return {
        "method": "threshold_components_patient_suppression",
        "min_component_area_px": int(config.min_component_area_px),
        "min_patient_positive_patches": int(config.min_patient_positive_patches),
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
    )


def threshold_and_filter_components(
    probabilities: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    *,
    decision_threshold: float,
    min_component_area_px: int,
) -> npt.NDArray[np.uint8]:
    """Threshold probabilities and remove small connected components per patch."""

    binary = cast(npt.NDArray[np.uint8], (probabilities > decision_threshold).astype(np.uint8))
    if min_component_area_px <= 0 or binary.size == 0:
        return binary
    if binary.ndim == BINARY_MASK_NDIM:
        return _filter_components_2d(binary, min_component_area_px=min_component_area_px)
    if binary.ndim == BATCH_MASK_NDIM:
        filtered = np.zeros_like(binary, dtype=np.uint8)
        for index in range(binary.shape[0]):
            filtered[index] = _filter_components_2d(
                binary[index],
                min_component_area_px=min_component_area_px,
            )
        return filtered
    raise ValueError(f"Expected 2D or 3D probability array, got shape {binary.shape}.")


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


def build_suppressed_patient_set(
    positive_patch_counts: Mapping[str, int],
    *,
    min_patient_positive_patches: int,
) -> set[str]:
    return {
        patient_id
        for patient_id, positive_patch_count in positive_patch_counts.items()
        if positive_patch_count < min_patient_positive_patches
    }


def apply_patient_positive_patch_suppression(
    stats_by_patient: Mapping[str, list[dict[str, int]]],
    truth_counts_by_patient: Mapping[str, dict[str, int]],
    positive_patch_counts: Mapping[str, int],
    *,
    min_patient_positive_patches: int,
) -> dict[str, list[dict[str, int]]]:
    suppressed_patients = build_suppressed_patient_set(
        positive_patch_counts,
        min_patient_positive_patches=min_patient_positive_patches,
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
