from __future__ import annotations

import contextlib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from helpers.ensemble_inference.metrics import (
    compute_auc_from_histograms,
    mask_to_binary_indices,
    summarize_patient_metrics,
)


def _autocast_context(
    images: torch.Tensor, *, use_amp: bool
) -> contextlib.AbstractContextManager[Any]:
    if not use_amp or not images.is_cuda:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def predict_with_tta_batched(
    model: nn.Module, images: torch.Tensor, use_amp: bool = True
) -> torch.Tensor:
    with _autocast_context(images, use_amp=use_amp):
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        probs = (
            torch.sigmoid(output).squeeze(1)
            if output.shape[1] == 1
            else torch.softmax(output, dim=1)[:, 1]
        )

    images_h = torch.flip(images, dims=[3])
    with _autocast_context(images_h, use_amp=use_amp):
        output_h = model(images_h)
        if isinstance(output_h, (tuple, list)):
            output_h = output_h[0]
        probs_h = (
            torch.sigmoid(output_h).squeeze(1)
            if output_h.shape[1] == 1
            else torch.softmax(output_h, dim=1)[:, 1]
        )
    probs.add_(torch.flip(probs_h, dims=[2]))

    images_v = torch.flip(images, dims=[2])
    with _autocast_context(images_v, use_amp=use_amp):
        output_v = model(images_v)
        if isinstance(output_v, (tuple, list)):
            output_v = output_v[0]
        probs_v = (
            torch.sigmoid(output_v).squeeze(1)
            if output_v.shape[1] == 1
            else torch.softmax(output_v, dim=1)[:, 1]
        )
    probs.add_(torch.flip(probs_v, dims=[1]))
    probs.div_(3.0)
    return probs


def compute_two_stream_probabilities(
    models_list: list[nn.Module],
    constituent_models_info: list[dict[str, Any]],
    images: torch.Tensor,
    *,
    roi_threshold: float,
    roi_scale: int,
) -> torch.Tensor:
    sem_accum: torch.Tensor | None = None
    spa_accum: torch.Tensor | None = None
    for model, meta in zip(models_list, constituent_models_info, strict=False):
        weight = float(meta.get("weight", meta.get("ensemble_weight", 0.0)))
        role = str(meta.get("stream_role", "none"))
        if weight <= 1e-8:
            continue
        probs = predict_with_tta_batched(model, images, use_amp=True)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs.clamp_(0.0, 1.0)
        if role == "semantic":
            sem_accum = probs * weight if sem_accum is None else sem_accum.add(probs, alpha=weight)
        elif role == "spatial":
            spa_accum = probs * weight if spa_accum is None else spa_accum.add(probs, alpha=weight)

    if spa_accum is None or sem_accum is None:
        return torch.zeros(
            (images.shape[0], images.shape[2], images.shape[3]), device=images.device
        )

    _, _, height, width = images.shape
    small_h = max(1, height // roi_scale)
    small_w = max(1, width // roi_scale)
    sem_small = functional.interpolate(
        sem_accum.unsqueeze(1),
        size=(small_h, small_w),
        mode="bilinear",
        align_corners=False,
    )
    roi_small = (sem_small > roi_threshold).float()
    roi_mask = functional.interpolate(roi_small, size=(height, width), mode="nearest").squeeze(1)
    final_probs = cast(torch.Tensor, spa_accum * roi_mask)
    final_probs.clamp_(0.0, 1.0)
    return final_probs


@torch.inference_mode()
def analyze_ensemble_metrics(
    models_list: list[nn.Module],
    constituent_models_info: list[dict[str, Any]],
    test_loader: DataLoader[Any],
    *,
    device: torch.device,
    optimal_threshold: float,
    roi_scale: int,
    train_mean: list[float],
    train_std: list[float],
    gpu_normalizer: nn.Module,
    seed: int,
) -> dict[str, Any]:
    for model in models_list:
        model.eval()

    stats_by_patient: dict[str, list[dict[str, int]]] = defaultdict(list)
    auc_bins = 4096
    auc_pos_hist = torch.zeros(auc_bins, dtype=torch.int64, device=device)
    auc_neg_hist = torch.zeros(auc_bins, dtype=torch.int64, device=device)

    for batch_data in test_loader:
        if batch_data is None:
            continue
        images, masks, patient_ids = batch_data
        images = gpu_normalizer(images.to(device, non_blocking=True))
        true_gpu = mask_to_binary_indices(masks.to(device, non_blocking=True))
        final_probs = compute_two_stream_probabilities(
            models_list,
            constituent_models_info,
            images,
            roi_threshold=optimal_threshold,
            roi_scale=roi_scale,
        )

        bin_idx = (final_probs * (auc_bins - 1)).long().clamp_(0, auc_bins - 1)
        flat_bins = bin_idx.view(-1)
        flat_true = true_gpu.view(-1).bool()
        auc_pos_hist.add_(torch.bincount(flat_bins[flat_true], minlength=auc_bins))
        auc_neg_hist.add_(torch.bincount(flat_bins[~flat_true], minlength=auc_bins))

        pred_gpu = (final_probs > optimal_threshold).to(torch.uint8)
        conf_vec = pred_gpu.mul(2).add_(true_gpu).view(pred_gpu.size(0), -1)
        for index, patient_id in enumerate(patient_ids):
            counts = torch.bincount(conf_vec[index], minlength=4).cpu().tolist()
            stats_by_patient[patient_id].append(
                {
                    "tn": int(counts[0]),
                    "fn": int(counts[1]),
                    "fp": int(counts[2]),
                    "tp": int(counts[3]),
                }
            )

    summary = summarize_patient_metrics(stats_by_patient, seed=seed)
    summary["auc"] = compute_auc_from_histograms(auc_pos_hist, auc_neg_hist)
    summary["normalization"] = {
        "mean": [float(x) for x in train_mean],
        "std": [float(x) for x in train_std],
    }
    summary["ensemble"] = {
        "method": "two_stream_spatial_gating",
        "threshold": float(optimal_threshold),
        "weights": None,
    }
    return summary


@torch.inference_mode()
def export_visualizations(
    models_list: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    threshold: float,
    roi_scale: int,
    train_mean: list[float],
    train_std: list[float],
    constituent_models_info: list[dict[str, Any]],
    gpu_normalizer: nn.Module,
    output_dir: Path,
    num_samples: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_samples <= 0:
        return []
    try:
        batch_data = next(iter(dataloader))
    except StopIteration:
        return []
    if batch_data is None:
        return []

    images, masks, _patient_ids = batch_data
    actual = min(num_samples, images.shape[0])
    if actual <= 0:
        return []

    images_vis = images[:actual].to(device, non_blocking=True)
    images_norm = gpu_normalizer(images_vis)
    final_probs = compute_two_stream_probabilities(
        models_list,
        constituent_models_info,
        images_norm,
        roi_threshold=threshold,
        roi_scale=roi_scale,
    )
    pred_masks = (final_probs >= threshold).to(torch.uint8).cpu().numpy()
    true_masks = mask_to_binary_indices(masks[:actual]).cpu().numpy().astype(np.uint8)
    images_np = images_norm.cpu().numpy()
    mean = np.asarray(train_mean, dtype=np.float32)
    std = np.asarray(train_std, dtype=np.float32)

    output_paths: list[Path] = []
    for index in range(actual):
        image = images_np[index].transpose(1, 2, 0)
        image = np.clip(std * image + mean, 0, 1)
        pred_overlay = np.where(pred_masks[index] == 0, np.nan, pred_masks[index]).astype(
            np.float32
        )
        true_overlay = np.where(true_masks[index] == 0, np.nan, true_masks[index]).astype(
            np.float32
        )
        figure, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(image)
        axes[1].imshow(pred_overlay, cmap="jet", alpha=0.5)
        axes[1].set_title("Predicted Mask")
        axes[2].imshow(image)
        axes[2].imshow(true_overlay, cmap="jet", alpha=0.5)
        axes[2].set_title("True Mask")
        axes[3].imshow(final_probs[index].cpu().numpy(), vmin=0, vmax=1)
        axes[3].set_title("Probability")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        output_path = output_dir / f"sample_{index + 1}.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths
