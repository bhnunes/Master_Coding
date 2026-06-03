from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import auc as sklearn_auc

MASK_BATCH_NDIM = 4
MASK_IMAGE_NDIM = 3
BINARY_MASK_NDIM = 2
BATCH_BINARY_MASK_NDIM = 3
BINARY_CLASS_COUNT = 2
BOOTSTRAP_MIN_PATIENTS = 20
CONFIDENCE_INTERVAL_PERCENTILES = [2.5, 97.5]
PATCH_LEVEL_METRIC_KEYS = ("accuracy", "avacc", "sensitivity", "specificity")


@dataclass(frozen=True)
class _PerPatientMetricSummary:
    tp: np.ndarray[Any, Any]
    fp: np.ndarray[Any, Any]
    fn: np.ndarray[Any, Any]
    tn: np.ndarray[Any, Any]
    per_patient_scores: dict[str, np.ndarray[Any, Any]]
    dice_pos_only: np.ndarray[Any, Any]
    neg_clean: np.ndarray[Any, Any]
    n_pos_patients: int
    n_neg_patients: int


def mask_to_binary_indices(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == MASK_BATCH_NDIM and masks.shape[1] == BINARY_CLASS_COUNT:
        return torch.argmax(masks, dim=1).to(torch.uint8)
    if masks.ndim == MASK_IMAGE_NDIM:
        return masks.to(torch.uint8)
    raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")


def calculate_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)

    denom_tpr = tp + fn
    denom_tnr = tn + fp
    denom_prec = tp + fp
    denom_acc = tp + tn + fp + fn

    gt_empty = (tp + fn) == 0.0
    pred_empty = (tp + fp) == 0.0

    if gt_empty:
        if pred_empty:
            dice = 1.0
            iou = 1.0
        else:
            dice = 0.0
            iou = 0.0
    else:
        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn
        dice = (2 * tp) / denom_dice if denom_dice > 0 else 0.0
        iou = tp / denom_iou if denom_iou > 0 else 0.0

    return {
        "dice": dice,
        "iou": iou,
        "accuracy": (tp + tn) / denom_acc if denom_acc > 0 else np.nan,
        "tpr": tp / denom_tpr if denom_tpr > 0 else np.nan,
        "tnr": tn / denom_tnr if denom_tnr > 0 else np.nan,
        "precision": tp / denom_prec if denom_prec > 0 else np.nan,
        "fpr": fp / denom_tnr if denom_tnr > 0 else np.nan,
        "fnr": fn / denom_tpr if denom_tpr > 0 else np.nan,
    }


def calculate_patch_classification_metrics(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
) -> dict[str, float]:
    metrics = calculate_metrics(tp, fp, fn, tn)
    sensitivity = metrics["tpr"]
    specificity = metrics["tnr"]
    avacc = (
        (sensitivity + specificity) / 2.0
        if not (np.isnan(sensitivity) or np.isnan(specificity))
        else np.nan
    )
    return {
        "accuracy": metrics["accuracy"],
        "avacc": avacc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def patch_labels_from_binary_masks(
    masks: np.ndarray[Any, Any],
    *,
    positive_area_fraction_threshold: float,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if not 0.0 <= positive_area_fraction_threshold <= 1.0:
        raise ValueError("positive_area_fraction_threshold must be between 0.0 and 1.0.")

    mask_array = np.asarray(masks).astype(bool)
    if mask_array.ndim == BINARY_MASK_NDIM:
        mask_array = mask_array[np.newaxis, :, :]
    if mask_array.ndim != BATCH_BINARY_MASK_NDIM:
        raise ValueError(f"Expected 2D or 3D binary masks, got shape {mask_array.shape}.")

    positive_pixels = np.count_nonzero(mask_array, axis=(1, 2))
    if positive_area_fraction_threshold <= 0.0:
        return cast(np.ndarray[Any, np.dtype[np.bool_]], positive_pixels > 0)

    patch_area = mask_array.shape[1] * mask_array.shape[2]
    positive_fraction = positive_pixels / float(patch_area)
    return cast(
        np.ndarray[Any, np.dtype[np.bool_]],
        positive_fraction >= positive_area_fraction_threshold,
    )


def confusion_counts_from_patch_labels(
    prediction_labels: np.ndarray[Any, Any],
    truth_labels: np.ndarray[Any, Any],
) -> dict[str, int]:
    pred_bool = np.asarray(prediction_labels).astype(bool).ravel()
    truth_bool = np.asarray(truth_labels).astype(bool).ravel()
    if pred_bool.shape != truth_bool.shape:
        raise ValueError(
            "Prediction and truth patch labels must have the same shape: "
            f"{pred_bool.shape} != {truth_bool.shape}."
        )
    return {
        "tp": int(np.sum(pred_bool & truth_bool)),
        "fp": int(np.sum(pred_bool & ~truth_bool)),
        "fn": int(np.sum(~pred_bool & truth_bool)),
        "tn": int(np.sum(~pred_bool & ~truth_bool)),
    }


def summarize_patch_classification_metrics(
    stats_by_patient: dict[str, list[dict[str, int]]],
) -> dict[str, Any]:
    all_patch_stats = [
        stat for patient_stats in stats_by_patient.values() for stat in patient_stats
    ]
    tp = int(sum(item["tp"] for item in all_patch_stats))
    fp = int(sum(item["fp"] for item in all_patch_stats))
    fn = int(sum(item["fn"] for item in all_patch_stats))
    tn = int(sum(item["tn"] for item in all_patch_stats))
    positive_patches = tp + fn
    negative_patches = tn + fp
    return {
        "point_estimate": calculate_patch_classification_metrics(tp, fp, fn, tn),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "support": {
            "total_patches": tp + fp + fn + tn,
            "positive_patches": positive_patches,
            "negative_patches": negative_patches,
        },
    }


def compute_auc_from_histograms(
    auc_pos_hist: torch.Tensor,
    auc_neg_hist: torch.Tensor,
) -> float:
    auc_pos_np = auc_pos_hist.cpu().numpy()
    auc_neg_np = auc_neg_hist.cpu().numpy()
    total_pos = int(auc_pos_np.sum())
    total_neg = int(auc_neg_np.sum())
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")

    tp_curve = np.cumsum(auc_pos_np[::-1]).astype(np.float64)
    fp_curve = np.cumsum(auc_neg_np[::-1]).astype(np.float64)
    tpr = np.concatenate(([0.0], tp_curve / total_pos))
    fpr = np.concatenate(([0.0], fp_curve / total_neg))
    return float(sklearn_auc(fpr, tpr))


def _summarize_per_patient_metrics(
    stats_by_patient: dict[str, list[dict[str, int]]],
    unique_patient_ids: list[str],
    metric_keys: list[str],
) -> _PerPatientMetricSummary:
    n_patients = len(unique_patient_ids)
    tp = np.zeros(n_patients, dtype=np.float64)
    fp = np.zeros(n_patients, dtype=np.float64)
    fn = np.zeros(n_patients, dtype=np.float64)
    tn = np.zeros(n_patients, dtype=np.float64)
    per_patient_scores = {key: np.zeros(n_patients, dtype=np.float64) for key in metric_keys}
    dice_pos_only = np.full(n_patients, np.nan, dtype=np.float64)
    neg_clean = np.full(n_patients, np.nan, dtype=np.float64)
    n_pos_patients = 0
    n_neg_patients = 0

    for index, patient_id in enumerate(unique_patient_ids):
        patient_stats = stats_by_patient[patient_id]
        tp_patient = float(sum(item["tp"] for item in patient_stats))
        fp_patient = float(sum(item["fp"] for item in patient_stats))
        fn_patient = float(sum(item["fn"] for item in patient_stats))
        tn_patient = float(sum(item["tn"] for item in patient_stats))
        tp[index] = tp_patient
        fp[index] = fp_patient
        fn[index] = fn_patient
        tn[index] = tn_patient
        patient_metrics = calculate_metrics(tp_patient, fp_patient, fn_patient, tn_patient)
        for key in metric_keys:
            per_patient_scores[key][index] = patient_metrics[key]
        if (tp_patient + fn_patient) > 0:
            dice_pos_only[index] = patient_metrics["dice"]
            n_pos_patients += 1
        else:
            neg_clean[index] = 1.0 if fp_patient == 0 else 0.0
            n_neg_patients += 1

    return _PerPatientMetricSummary(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        per_patient_scores=per_patient_scores,
        dice_pos_only=dice_pos_only,
        neg_clean=neg_clean,
        n_pos_patients=n_pos_patients,
        n_neg_patients=n_neg_patients,
    )


def _bootstrap_metric_intervals(
    *,
    metric_keys: list[str],
    per_patient_summary: _PerPatientMetricSummary,
    n_bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[float], list[float]]:
    n_patients = len(per_patient_summary.tp)
    rng = np.random.default_rng(seed)
    boot_micro = {key: np.zeros(n_bootstrap_samples, dtype=np.float64) for key in metric_keys}
    boot_macro = {key: np.zeros(n_bootstrap_samples, dtype=np.float64) for key in metric_keys}
    boot_dice_pos_only = np.zeros(n_bootstrap_samples, dtype=np.float64)
    boot_neg_clean = np.zeros(n_bootstrap_samples, dtype=np.float64)

    for index in range(n_bootstrap_samples):
        sampled_indices = rng.choice(n_patients, size=n_patients, replace=True)
        metrics = calculate_metrics(
            tp=float(np.sum(per_patient_summary.tp[sampled_indices])),
            fp=float(np.sum(per_patient_summary.fp[sampled_indices])),
            fn=float(np.sum(per_patient_summary.fn[sampled_indices])),
            tn=float(np.sum(per_patient_summary.tn[sampled_indices])),
        )
        for key in metric_keys:
            boot_micro[key][index] = metrics[key]

        for key in metric_keys:
            boot_macro[key][index] = np.nanmean(
                per_patient_summary.per_patient_scores[key][sampled_indices]
            )
        boot_dice_pos_only[index] = np.nanmean(per_patient_summary.dice_pos_only[sampled_indices])
        boot_neg_clean[index] = np.nanmean(per_patient_summary.neg_clean[sampled_indices])

    final_micro_ci = {
        key: np.percentile(boot_micro[key], CONFIDENCE_INTERVAL_PERCENTILES).tolist()
        for key in metric_keys
    }
    final_macro_ci = {
        key: np.percentile(boot_macro[key], CONFIDENCE_INTERVAL_PERCENTILES).tolist()
        for key in metric_keys
    }
    macro_dice_pos_only_ci = np.percentile(
        boot_dice_pos_only,
        CONFIDENCE_INTERVAL_PERCENTILES,
    ).tolist()
    macro_neg_clean_rate_ci = np.percentile(
        boot_neg_clean,
        CONFIDENCE_INTERVAL_PERCENTILES,
    ).tolist()
    return final_micro_ci, final_macro_ci, macro_dice_pos_only_ci, macro_neg_clean_rate_ci


def summarize_patient_metrics(
    stats_by_patient: dict[str, list[dict[str, int]]],
    *,
    seed: int,
    n_bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    if not stats_by_patient:
        raise ValueError("No patients were processed during inference.")

    unique_patient_ids = list(stats_by_patient.keys())
    n_patients = len(unique_patient_ids)
    run_bootstrap = n_patients >= BOOTSTRAP_MIN_PATIENTS
    metric_keys = ["dice", "iou", "accuracy", "tpr", "tnr", "precision", "fpr", "fnr"]

    final_micro_metrics: dict[str, Any] = {
        "point_estimate": {},
        "ci": {key: [np.nan, np.nan] for key in metric_keys},
    }
    final_macro_metrics: dict[str, Any] = {
        "point_estimate": {},
        "ci": {key: [np.nan, np.nan] for key in metric_keys},
    }

    all_patch_stats = [
        stat for patient_stats in stats_by_patient.values() for stat in patient_stats
    ]
    tp_micro = float(sum(item["tp"] for item in all_patch_stats))
    fp_micro = float(sum(item["fp"] for item in all_patch_stats))
    fn_micro = float(sum(item["fn"] for item in all_patch_stats))
    tn_micro = float(sum(item["tn"] for item in all_patch_stats))
    final_micro_metrics["point_estimate"] = calculate_metrics(
        tp_micro, fp_micro, fn_micro, tn_micro
    )

    per_patient_summary = _summarize_per_patient_metrics(
        stats_by_patient,
        unique_patient_ids,
        metric_keys,
    )

    for key in metric_keys:
        final_macro_metrics["point_estimate"][key] = float(
            np.nanmean(per_patient_summary.per_patient_scores[key])
        )

    macro_dice_pos_only_pe = (
        float(np.nanmean(per_patient_summary.dice_pos_only))
        if per_patient_summary.n_pos_patients > 0
        else float("nan")
    )
    macro_neg_clean_rate_pe = (
        float(np.nanmean(per_patient_summary.neg_clean))
        if per_patient_summary.n_neg_patients > 0
        else float("nan")
    )

    macro_dice_pos_only_ci = [np.nan, np.nan]
    macro_neg_clean_rate_ci = [np.nan, np.nan]
    if run_bootstrap:
        (
            final_micro_metrics["ci"],
            final_macro_metrics["ci"],
            boot_dice_ci,
            boot_neg_ci,
        ) = _bootstrap_metric_intervals(
            metric_keys=metric_keys,
            per_patient_summary=per_patient_summary,
            n_bootstrap_samples=n_bootstrap_samples,
            seed=seed,
        )
        if per_patient_summary.n_pos_patients > 0:
            macro_dice_pos_only_ci = boot_dice_ci
        if per_patient_summary.n_neg_patients > 0:
            macro_neg_clean_rate_ci = boot_neg_ci

    return {
        "micro_averaged_metrics": final_micro_metrics,
        "macro_averaged_metrics": final_macro_metrics,
        "bootstrap": {
            "ran": bool(run_bootstrap),
            "n_patients": int(n_patients),
            "n_bootstrap_samples": int(n_bootstrap_samples) if run_bootstrap else 0,
            "seed": int(seed),
        },
        "confusion_matrix": {
            "tp": int(tp_micro),
            "fp": int(fp_micro),
            "fn": int(fn_micro),
            "tn": int(tn_micro),
        },
        "macro_dice_rule6_split": {
            "dice_pos_only": {
                "point_estimate": float(macro_dice_pos_only_pe),
                "ci": [float(macro_dice_pos_only_ci[0]), float(macro_dice_pos_only_ci[1])],
            },
            "neg_clean_rate": {
                "point_estimate": float(macro_neg_clean_rate_pe),
                "ci": [float(macro_neg_clean_rate_ci[0]), float(macro_neg_clean_rate_ci[1])],
            },
            "n_pos_patients": int(per_patient_summary.n_pos_patients),
            "n_neg_patients": int(per_patient_summary.n_neg_patients),
        },
    }
