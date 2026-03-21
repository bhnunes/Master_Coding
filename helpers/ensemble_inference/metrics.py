from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc as sklearn_auc


def mask_to_binary_indices(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 2:
        return torch.argmax(masks, dim=1).to(torch.uint8)
    if masks.ndim == 3:
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
    run_bootstrap = n_patients >= 20
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
        patient_metrics = calculate_metrics(tp_patient, fp_patient, fn_patient, tn_patient)
        for key in metric_keys:
            per_patient_scores[key][index] = patient_metrics[key]
        has_tumor = (tp_patient + fn_patient) > 0
        if has_tumor:
            dice_pos_only[index] = patient_metrics["dice"]
            n_pos_patients += 1
        else:
            neg_clean[index] = 1.0 if fp_patient == 0 else 0.0
            n_neg_patients += 1

    for key in metric_keys:
        final_macro_metrics["point_estimate"][key] = float(np.nanmean(per_patient_scores[key]))

    macro_dice_pos_only_pe = (
        float(np.nanmean(dice_pos_only)) if n_pos_patients > 0 else float("nan")
    )
    macro_neg_clean_rate_pe = float(np.nanmean(neg_clean)) if n_neg_patients > 0 else float("nan")

    macro_dice_pos_only_ci = [np.nan, np.nan]
    macro_neg_clean_rate_ci = [np.nan, np.nan]
    if run_bootstrap:
        rng = np.random.default_rng(seed)
        boot_micro = {key: np.zeros(n_bootstrap_samples, dtype=np.float64) for key in metric_keys}
        boot_macro = {key: np.zeros(n_bootstrap_samples, dtype=np.float64) for key in metric_keys}
        boot_dice_pos_only = np.zeros(n_bootstrap_samples, dtype=np.float64)
        boot_neg_clean = np.zeros(n_bootstrap_samples, dtype=np.float64)

        for index in range(n_bootstrap_samples):
            resampled_pids = rng.choice(unique_patient_ids, size=n_patients, replace=True)
            resampled_stats = [stat for pid in resampled_pids for stat in stats_by_patient[pid]]
            metrics = calculate_metrics(
                tp=sum(item["tp"] for item in resampled_stats),
                fp=sum(item["fp"] for item in resampled_stats),
                fn=sum(item["fn"] for item in resampled_stats),
                tn=sum(item["tn"] for item in resampled_stats),
            )
            for key in metric_keys:
                boot_micro[key][index] = metrics[key]

            sampled_indices = rng.choice(n_patients, size=n_patients, replace=True)
            for key in metric_keys:
                boot_macro[key][index] = np.nanmean(per_patient_scores[key][sampled_indices])
            boot_dice_pos_only[index] = np.nanmean(dice_pos_only[sampled_indices])
            boot_neg_clean[index] = np.nanmean(neg_clean[sampled_indices])

        for key in metric_keys:
            final_micro_metrics["ci"][key] = np.percentile(boot_micro[key], [2.5, 97.5]).tolist()
            final_macro_metrics["ci"][key] = np.percentile(boot_macro[key], [2.5, 97.5]).tolist()

        if n_pos_patients > 0:
            macro_dice_pos_only_ci = np.percentile(boot_dice_pos_only, [2.5, 97.5]).tolist()
        if n_neg_patients > 0:
            macro_neg_clean_rate_ci = np.percentile(boot_neg_clean, [2.5, 97.5]).tolist()

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
            "n_pos_patients": int(n_pos_patients),
            "n_neg_patients": int(n_neg_patients),
        },
    }
