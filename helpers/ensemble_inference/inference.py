from __future__ import annotations

import contextlib
import re
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
    calculate_metrics,
    compute_auc_from_histograms,
    mask_to_binary_indices,
    summarize_patient_metrics,
)

_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _compute_confusion_counts(
    pred_mask: torch.Tensor, true_mask: torch.Tensor
) -> tuple[int, int, int, int]:
    pred_uint8 = pred_mask.to(torch.uint8)
    true_uint8 = true_mask.to(torch.uint8)
    conf = pred_uint8.mul(2).add_(true_uint8)
    counts = torch.bincount(conf.view(-1), minlength=4).cpu().tolist()
    return int(counts[3]), int(counts[2]), int(counts[1]), int(counts[0])


def _sanitize_path_component(value: str) -> str:
    sanitized = _FILENAME_SANITIZE_PATTERN.sub("_", value.strip())
    return sanitized.strip("._") or "sample"


def _build_visualization_filename(
    *,
    rank: int,
    patient_id: str,
    filename: str | None,
) -> str:
    parts = [f"worst_dice_{rank:02d}", _sanitize_path_component(patient_id)]
    if filename is not None and filename.strip():
        parts.append(_sanitize_path_component(Path(filename).stem))
    return "__".join(parts) + ".png"


def _sample_sort_key(sample: dict[str, Any]) -> tuple[float, int]:
    return (float(sample["dice"]), int(sample["sample_index"]))


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
    roi_threshold: float,
    decision_threshold: float,
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
        images, masks, patient_ids, *_metadata = batch_data
        images = gpu_normalizer(images.to(device, non_blocking=True))
        true_gpu = mask_to_binary_indices(masks.to(device, non_blocking=True))
        final_probs = compute_two_stream_probabilities(
            models_list,
            constituent_models_info,
            images,
            roi_threshold=roi_threshold,
            roi_scale=roi_scale,
        )

        bin_idx = (final_probs * (auc_bins - 1)).long().clamp_(0, auc_bins - 1)
        flat_bins = bin_idx.view(-1)
        flat_true = true_gpu.view(-1).bool()
        auc_pos_hist.add_(torch.bincount(flat_bins[flat_true], minlength=auc_bins))
        auc_neg_hist.add_(torch.bincount(flat_bins[~flat_true], minlength=auc_bins))

        pred_gpu = (final_probs > decision_threshold).to(torch.uint8)
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
        "roi_threshold": float(roi_threshold),
        "decision_threshold": float(decision_threshold),
        "weights": None,
    }
    return summary


@torch.inference_mode()
def export_visualizations(
    models_list: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    roi_threshold: float,
    decision_threshold: float,
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
    ranked_samples: list[dict[str, Any]] = []
    sample_index = 0

    for batch_data in dataloader:
        if batch_data is None:
            continue

        if len(batch_data) == 3:
            images, masks, patient_ids = batch_data
            filenames = [None] * images.shape[0]
        else:
            images, masks, patient_ids, filenames = batch_data
        actual = images.shape[0]
        if actual <= 0:
            continue

        images_vis = images.to(device, non_blocking=True)
        images_norm = gpu_normalizer(images_vis)
        final_probs = compute_two_stream_probabilities(
            models_list,
            constituent_models_info,
            images_norm,
            roi_threshold=roi_threshold,
            roi_scale=roi_scale,
        )
        pred_masks = (final_probs > decision_threshold).to(torch.uint8)
        true_masks = mask_to_binary_indices(masks.to(device, non_blocking=True))
        images_np = images_norm.cpu().numpy()
        pred_masks_np = pred_masks.cpu().numpy().astype(np.uint8)
        true_masks_np = true_masks.cpu().numpy().astype(np.uint8)
        probs_np = final_probs.cpu().numpy()

        for index in range(actual):
            tp, fp, fn, tn = _compute_confusion_counts(pred_masks[index], true_masks[index])
            dice = float(calculate_metrics(tp, fp, fn, tn)["dice"])
            candidate = {
                "dice": dice,
                "sample_index": sample_index,
                "patient_id": patient_ids[index],
                "filename": filenames[index],
                "image": images_np[index],
                "pred_mask": pred_masks_np[index],
                "true_mask": true_masks_np[index],
                "probability": probs_np[index],
            }
            if len(ranked_samples) < num_samples:
                ranked_samples.append(candidate)
            else:
                current_best_index = max(
                    range(len(ranked_samples)),
                    key=lambda selected_index: _sample_sort_key(ranked_samples[selected_index]),
                )
                if _sample_sort_key(candidate) < _sample_sort_key(
                    ranked_samples[current_best_index]
                ):
                    ranked_samples[current_best_index] = candidate
            sample_index += 1

    if not ranked_samples:
        return []

    selected_samples = sorted(ranked_samples, key=_sample_sort_key)

    mean = np.asarray(train_mean, dtype=np.float32)
    std = np.asarray(train_std, dtype=np.float32)

    output_paths: list[Path] = []
    for rank, sample in enumerate(selected_samples, start=1):
        image = sample["image"].transpose(1, 2, 0)
        image = np.clip(std * image + mean, 0, 1)
        pred_overlay = np.where(sample["pred_mask"] == 0, np.nan, sample["pred_mask"]).astype(
            np.float32
        )
        true_overlay = np.where(sample["true_mask"] == 0, np.nan, sample["true_mask"]).astype(
            np.float32
        )
        figure, axes = plt.subplots(1, 4, figsize=(18, 5))
        filename = sample["filename"]
        title = f"Worst Dice #{rank}: {sample['dice']:.4f} | {sample['patient_id']}"
        if isinstance(filename, str) and filename.strip():
            title += f" | {filename}"
        figure.suptitle(title)
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(image)
        axes[1].imshow(pred_overlay, cmap="jet", alpha=0.5)
        axes[1].set_title("Predicted Mask")
        axes[2].imshow(image)
        axes[2].imshow(true_overlay, cmap="jet", alpha=0.5)
        axes[2].set_title("True Mask")
        axes[3].imshow(sample["probability"], vmin=0, vmax=1)
        axes[3].set_title("Probability")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        output_path = output_dir / _build_visualization_filename(
            rank=rank,
            patient_id=sample["patient_id"],
            filename=filename if isinstance(filename, str) else None,
        )
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths
