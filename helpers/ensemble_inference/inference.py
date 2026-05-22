from __future__ import annotations

import contextlib
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.ensemble_inference.metrics import (
    calculate_metrics,
    compute_auc_from_histograms,
    confusion_counts_from_patch_labels,
    mask_to_binary_indices,
    patch_labels_from_binary_masks,
    summarize_patch_classification_metrics,
    summarize_patient_metrics,
)
from helpers.ensemble_postprocessing import (
    PostprocessingConfig,
    apply_patient_positive_patch_suppression,
    build_suppressed_patient_set,
    confusion_counts_from_binary_masks,
    count_positive_prediction_patches,
    largest_component_area,
    patient_positive_area_fractions_from_stats,
    postprocessing_config_to_payload,
    threshold_and_filter_components,
    truth_pixel_counts,
)
from helpers.runtime_platform import load_headless_matplotlib_pyplot

_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_NEGLIGIBLE_MODEL_WEIGHT = 1e-8
_BATCH_WITHOUT_FILENAMES = 3
_PATCH_METADATA_LABEL_INDEX = 1
_AUC_BINS = 4096
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5
_PATCH_GT_RULE_MANIFEST = "stage2_manifest_label_after_extraction_overlap_rule"
_PATCH_GT_RULE_MASK_FALLBACK = "binary_mask_has_positive_pixel_fallback"


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under LoggerWriter."""

    return sys.__stderr__


def _progress_total(iterable: Iterable[Any]) -> int | None:
    if not isinstance(iterable, Sized):
        return None
    try:
        return len(iterable)
    except TypeError:
        return None


def _progress_iterable(
    iterable: Iterable[Any],
    *,
    desc: str,
    unit: str,
) -> tqdm[Any]:
    return tqdm(
        iterable,
        total=_progress_total(iterable),
        desc=desc,
        unit=unit,
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        position=0,
        file=_progress_file(),
    )


@dataclass(frozen=True)
class EnsembleAnalysisConfig:
    device: torch.device
    roi_threshold: float
    decision_threshold: float
    postprocessing_config: PostprocessingConfig
    roi_scale: int
    train_mean: list[float]
    train_std: list[float]
    gpu_normalizer: nn.Module
    seed: int
    patch_positive_area_fraction_threshold: float = 0.0


@dataclass(frozen=True)
class VisualizationExportConfig:
    device: torch.device
    roi_threshold: float
    decision_threshold: float
    postprocessing_config: PostprocessingConfig
    roi_scale: int
    train_mean: list[float]
    train_std: list[float]
    constituent_models_info: list[dict[str, Any]]
    gpu_normalizer: nn.Module
    output_dir: Path
    num_samples: int


@dataclass(frozen=True)
class VisualizationBatch:
    images: torch.Tensor
    masks: torch.Tensor
    patient_ids: list[str]
    filenames: list[str | None]


@dataclass(frozen=True)
class VisualizationSample:
    dice: float
    sample_index: int
    patient_id: str
    filename: str | None
    image: np.ndarray[Any, Any]
    pred_mask: np.ndarray[Any, Any]
    true_mask: np.ndarray[Any, Any]
    probability: np.ndarray[Any, Any]


@dataclass
class _MetricAccumulator:
    stats_by_patient: dict[str, list[dict[str, int]]]
    truth_counts_by_patient: dict[str, dict[str, int]]
    patch_stats_by_patient: dict[str, list[dict[str, int]]]
    patch_truth_counts_by_patient: dict[str, dict[str, int]]
    positive_patch_counts: dict[str, int]
    largest_component_areas: dict[str, int]
    patch_ground_truth_sources: set[str]
    auc_pos_hist: torch.Tensor
    auc_neg_hist: torch.Tensor
    processed_samples: int = 0
    skipped_batches: int = 0


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


def _visualization_sort_key(sample: VisualizationSample) -> tuple[float, int]:
    return (sample.dice, sample.sample_index)


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
        if weight <= _NEGLIGIBLE_MODEL_WEIGHT:
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
    final_probs = spa_accum * roi_mask
    final_probs.clamp_(0.0, 1.0)
    return final_probs


def _patch_truth_labels_from_batch(
    metadata: list[Any],
    true_masks: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], str]:
    if (
        len(metadata) > _PATCH_METADATA_LABEL_INDEX
        and metadata[_PATCH_METADATA_LABEL_INDEX] is not None
    ):
        labels = metadata[_PATCH_METADATA_LABEL_INDEX]
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        truth_labels = labels_np.astype(bool).ravel()
        source = _PATCH_GT_RULE_MANIFEST
    else:
        truth_labels = patch_labels_from_binary_masks(
            true_masks,
            positive_area_fraction_threshold=0.0,
        )
        source = _PATCH_GT_RULE_MASK_FALLBACK

    if truth_labels.shape[0] != true_masks.shape[0]:
        raise ValueError(
            "Patch-level truth labels do not match the current batch size: "
            f"{truth_labels.shape[0]} != {true_masks.shape[0]}."
        )
    return truth_labels, source


def _build_metric_accumulator(config: EnsembleAnalysisConfig) -> _MetricAccumulator:
    return _MetricAccumulator(
        stats_by_patient=defaultdict(list),
        truth_counts_by_patient=defaultdict(lambda: {"positive": 0, "negative": 0}),
        patch_stats_by_patient=defaultdict(list),
        patch_truth_counts_by_patient=defaultdict(lambda: {"positive": 0, "negative": 0}),
        positive_patch_counts=defaultdict(int),
        largest_component_areas=defaultdict(int),
        patch_ground_truth_sources=set(),
        auc_pos_hist=torch.zeros(_AUC_BINS, dtype=torch.int64, device=config.device),
        auc_neg_hist=torch.zeros(_AUC_BINS, dtype=torch.int64, device=config.device),
    )


def _record_auc_histograms(
    accumulator: _MetricAccumulator,
    final_probs: torch.Tensor,
    true_masks: torch.Tensor,
) -> None:
    bin_idx = (final_probs * (_AUC_BINS - 1)).long().clamp_(0, _AUC_BINS - 1)
    flat_bins = bin_idx.view(-1)
    flat_true = true_masks.view(-1).bool()
    accumulator.auc_pos_hist.add_(torch.bincount(flat_bins[flat_true], minlength=_AUC_BINS))
    accumulator.auc_neg_hist.add_(torch.bincount(flat_bins[~flat_true], minlength=_AUC_BINS))


def _record_pixel_level_counts(
    accumulator: _MetricAccumulator,
    *,
    patient_ids: list[Any],
    predictions: np.ndarray[Any, Any],
    truth_masks: np.ndarray[Any, Any],
) -> None:
    for index, patient_id in enumerate(patient_ids):
        patient_id_str = str(patient_id)
        accumulator.stats_by_patient[patient_id_str].append(
            confusion_counts_from_binary_masks(predictions[index], truth_masks[index])
        )
        truth_counts = truth_pixel_counts(truth_masks[index])
        accumulator.truth_counts_by_patient[patient_id_str]["positive"] += truth_counts["positive"]
        accumulator.truth_counts_by_patient[patient_id_str]["negative"] += truth_counts["negative"]
        accumulator.positive_patch_counts[patient_id_str] += count_positive_prediction_patches(
            predictions[index]
        )
        accumulator.largest_component_areas[patient_id_str] = max(
            accumulator.largest_component_areas[patient_id_str],
            largest_component_area(predictions[index]),
        )


def _record_patch_level_counts(
    accumulator: _MetricAccumulator,
    *,
    patient_ids: list[Any],
    predictions: np.ndarray[Any, Any],
    truth_masks: np.ndarray[Any, Any],
    metadata: list[Any],
    positive_area_fraction_threshold: float,
) -> None:
    truth_patch_labels, truth_label_source = _patch_truth_labels_from_batch(metadata, truth_masks)
    accumulator.patch_ground_truth_sources.add(truth_label_source)
    pred_patch_labels = patch_labels_from_binary_masks(
        predictions,
        positive_area_fraction_threshold=positive_area_fraction_threshold,
    )
    for index, patient_id in enumerate(patient_ids):
        patient_id_str = str(patient_id)
        accumulator.patch_stats_by_patient[patient_id_str].append(
            confusion_counts_from_patch_labels(
                np.asarray([pred_patch_labels[index]]),
                np.asarray([truth_patch_labels[index]]),
            )
        )
        if truth_patch_labels[index]:
            accumulator.patch_truth_counts_by_patient[patient_id_str]["positive"] += 1
        else:
            accumulator.patch_truth_counts_by_patient[patient_id_str]["negative"] += 1


def _record_batch_metrics(
    accumulator: _MetricAccumulator,
    *,
    models_list: list[nn.Module],
    constituent_models_info: list[dict[str, Any]],
    batch_data: Any,
    config: EnsembleAnalysisConfig,
) -> None:
    images, masks, patient_ids, *metadata = batch_data
    images = config.gpu_normalizer(images.to(config.device, non_blocking=True))
    true_gpu = mask_to_binary_indices(masks.to(config.device, non_blocking=True))
    final_probs = compute_two_stream_probabilities(
        models_list,
        constituent_models_info,
        images,
        roi_threshold=config.roi_threshold,
        roi_scale=config.roi_scale,
    )

    _record_auc_histograms(accumulator, final_probs, true_gpu)
    final_probs_np = final_probs.detach().cpu().numpy().astype(np.float32)
    true_np = true_gpu.cpu().numpy().astype(np.uint8)
    pred_np = threshold_and_filter_components(
        final_probs_np,
        decision_threshold=config.decision_threshold,
        min_component_area_px=config.postprocessing_config.min_component_area_px,
        min_component_area_fraction_patch=(
            config.postprocessing_config.min_component_area_fraction_patch
        ),
    )
    _record_pixel_level_counts(
        accumulator,
        patient_ids=patient_ids,
        predictions=pred_np,
        truth_masks=true_np,
    )
    _record_patch_level_counts(
        accumulator,
        patient_ids=patient_ids,
        predictions=pred_np,
        truth_masks=true_np,
        metadata=metadata,
        positive_area_fraction_threshold=config.patch_positive_area_fraction_threshold,
    )
    accumulator.processed_samples += len(patient_ids)


def _patch_level_payload(
    *,
    pre_patient_suppression_stats: dict[str, list[dict[str, int]]],
    post_patient_suppression_stats: dict[str, list[dict[str, int]]],
    positive_area_fraction_threshold: float,
    ground_truth_sources: set[str],
) -> dict[str, Any]:
    comparator = ">" if positive_area_fraction_threshold <= 0.0 else ">="
    return {
        "method": "diagset_patch_recognition_from_segmentation_masks",
        "primary_comparison": "pre_patient_suppression",
        "prediction_rule": {
            "mask_source": (
                "component_filtered_binary_segmentation_mask_before_patient_suppression"
            ),
            "positive_area_fraction_threshold": float(positive_area_fraction_threshold),
            "positive_comparator": comparator,
        },
        "ground_truth_rule": (
            "; ".join(sorted(ground_truth_sources))
            if ground_truth_sources
            else _PATCH_GT_RULE_MASK_FALLBACK
        ),
        "pre_patient_suppression": summarize_patch_classification_metrics(
            pre_patient_suppression_stats
        ),
        "post_patient_suppression": summarize_patch_classification_metrics(
            post_patient_suppression_stats
        ),
    }


def _aggregate_patient_counts(patient_stats: list[dict[str, int]]) -> dict[str, int]:
    return {
        "tp": int(sum(item["tp"] for item in patient_stats)),
        "fp": int(sum(item["fp"] for item in patient_stats)),
        "fn": int(sum(item["fn"] for item in patient_stats)),
        "tn": int(sum(item["tn"] for item in patient_stats)),
    }


def _patient_diagnostic_rows(
    *,
    pre_patient_suppression_stats: dict[str, list[dict[str, int]]],
    post_patient_suppression_stats: dict[str, list[dict[str, int]]],
    positive_patch_counts: dict[str, int],
    largest_component_areas: dict[str, int],
    suppressed_patients: set[str],
) -> list[dict[str, int | float | str | bool]]:
    rows: list[dict[str, int | float | str | bool]] = []
    for patient_id in sorted(post_patient_suppression_stats):
        counts = _aggregate_patient_counts(post_patient_suppression_stats[patient_id])
        pre_counts = _aggregate_patient_counts(pre_patient_suppression_stats[patient_id])
        metrics = calculate_metrics(
            float(counts["tp"]),
            float(counts["fp"]),
            float(counts["fn"]),
            float(counts["tn"]),
        )
        rows.append(
            {
                "patient_id": patient_id,
                "gt_positive_area": counts["tp"] + counts["fn"],
                "pred_positive_area": counts["tp"] + counts["fp"],
                "pre_suppression_pred_positive_area": pre_counts["tp"] + pre_counts["fp"],
                "tp_area": counts["tp"],
                "fp_area": counts["fp"],
                "fn_area": counts["fn"],
                "tn_area": counts["tn"],
                "dice": float(metrics["dice"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["tpr"]),
                "positive_patch_count": int(positive_patch_counts.get(patient_id, 0)),
                "largest_component_area": int(largest_component_areas.get(patient_id, 0)),
                "suppressed": patient_id in suppressed_patients,
            }
        )
    return rows


@torch.inference_mode()
def analyze_ensemble_metrics(
    models_list: list[nn.Module],
    constituent_models_info: list[dict[str, Any]],
    test_loader: DataLoader[Any],
    config: EnsembleAnalysisConfig,
) -> dict[str, Any]:
    for model in models_list:
        model.eval()

    accumulator = _build_metric_accumulator(config)
    progress = _progress_iterable(test_loader, desc="Inference TEST", unit="batch")
    try:
        for batch_data in progress:
            if batch_data is None:
                accumulator.skipped_batches += 1
                progress.set_postfix(skipped=accumulator.skipped_batches, refresh=False)
                continue
            _record_batch_metrics(
                accumulator,
                models_list=models_list,
                constituent_models_info=constituent_models_info,
                batch_data=batch_data,
                config=config,
            )
            progress.set_postfix(
                samples=accumulator.processed_samples,
                patients=len(accumulator.stats_by_patient),
                skipped=accumulator.skipped_batches,
                refresh=False,
            )
    finally:
        progress.close()

    positive_area_fractions = patient_positive_area_fractions_from_stats(
        accumulator.stats_by_patient
    )
    suppressed_patients = build_suppressed_patient_set(
        accumulator.positive_patch_counts,
        min_patient_positive_patches=(config.postprocessing_config.min_patient_positive_patches),
        positive_area_fractions=positive_area_fractions,
        min_patient_positive_area_fraction=(
            config.postprocessing_config.min_patient_positive_area_fraction
        ),
    )
    postprocessed_stats = apply_patient_positive_patch_suppression(
        accumulator.stats_by_patient,
        accumulator.truth_counts_by_patient,
        accumulator.positive_patch_counts,
        min_patient_positive_patches=config.postprocessing_config.min_patient_positive_patches,
        positive_area_fractions=positive_area_fractions,
        min_patient_positive_area_fraction=(
            config.postprocessing_config.min_patient_positive_area_fraction
        ),
    )
    postprocessed_patch_stats = apply_patient_positive_patch_suppression(
        accumulator.patch_stats_by_patient,
        accumulator.patch_truth_counts_by_patient,
        accumulator.positive_patch_counts,
        min_patient_positive_patches=config.postprocessing_config.min_patient_positive_patches,
        positive_area_fractions=positive_area_fractions,
        min_patient_positive_area_fraction=(
            config.postprocessing_config.min_patient_positive_area_fraction
        ),
    )
    summary = summarize_patient_metrics(postprocessed_stats, seed=config.seed)
    summary["patient_diagnostics"] = _patient_diagnostic_rows(
        pre_patient_suppression_stats=accumulator.stats_by_patient,
        post_patient_suppression_stats=postprocessed_stats,
        positive_patch_counts=accumulator.positive_patch_counts,
        largest_component_areas=accumulator.largest_component_areas,
        suppressed_patients=suppressed_patients,
    )
    summary["diagset_patch_level_metrics"] = _patch_level_payload(
        pre_patient_suppression_stats=accumulator.patch_stats_by_patient,
        post_patient_suppression_stats=postprocessed_patch_stats,
        positive_area_fraction_threshold=config.patch_positive_area_fraction_threshold,
        ground_truth_sources=accumulator.patch_ground_truth_sources,
    )
    summary["auc"] = compute_auc_from_histograms(
        accumulator.auc_pos_hist,
        accumulator.auc_neg_hist,
    )
    summary["auc_source"] = "raw_probabilities_before_hard_postprocessing"
    summary["normalization"] = {
        "mean": [float(x) for x in config.train_mean],
        "std": [float(x) for x in config.train_std],
    }
    summary["postprocessing"] = postprocessing_config_to_payload(config.postprocessing_config)
    summary["postprocessing"]["suppressed_patient_count"] = len(suppressed_patients)
    summary["ensemble"] = {
        "method": "two_stream_spatial_gating",
        "roi_threshold": float(config.roi_threshold),
        "decision_threshold": float(config.decision_threshold),
        "postprocessing": postprocessing_config_to_payload(config.postprocessing_config),
        "weights": None,
    }
    return summary


def _unpack_visualization_batch(batch_data: Any) -> VisualizationBatch | None:
    if batch_data is None:
        return None
    if len(batch_data) == _BATCH_WITHOUT_FILENAMES:
        images, masks, patient_ids = batch_data
        filenames = [None] * images.shape[0]
    else:
        images, masks, patient_ids, filenames, *_extra = batch_data
    return VisualizationBatch(
        images=images,
        masks=masks,
        patient_ids=list(patient_ids),
        filenames=list(filenames),
    )


def _collect_visualization_candidates(
    models_list: list[nn.Module],
    batch: VisualizationBatch,
    *,
    config: VisualizationExportConfig,
    sample_index_start: int,
    suppressed_patients: set[str],
) -> tuple[list[VisualizationSample], int]:
    actual = batch.images.shape[0]
    if actual <= 0:
        return [], sample_index_start

    images_vis = batch.images.to(config.device, non_blocking=True)
    images_norm = config.gpu_normalizer(images_vis)
    final_probs = compute_two_stream_probabilities(
        models_list,
        config.constituent_models_info,
        images_norm,
        roi_threshold=config.roi_threshold,
        roi_scale=config.roi_scale,
    )
    true_masks = mask_to_binary_indices(batch.masks.to(config.device, non_blocking=True))
    images_np = images_norm.cpu().numpy()
    true_masks_np = true_masks.cpu().numpy().astype(np.uint8)
    probs_np = final_probs.cpu().numpy().astype(np.float32)
    pred_masks_np = threshold_and_filter_components(
        probs_np,
        decision_threshold=config.decision_threshold,
        min_component_area_px=config.postprocessing_config.min_component_area_px,
        min_component_area_fraction_patch=(
            config.postprocessing_config.min_component_area_fraction_patch
        ),
    )

    samples: list[VisualizationSample] = []
    sample_index = sample_index_start
    for index in range(actual):
        patient_id = batch.patient_ids[index]
        if patient_id in suppressed_patients:
            pred_masks_np[index] = np.zeros_like(pred_masks_np[index], dtype=np.uint8)
        counts = confusion_counts_from_binary_masks(pred_masks_np[index], true_masks_np[index])
        dice = float(
            calculate_metrics(counts["tp"], counts["fp"], counts["fn"], counts["tn"])["dice"]
        )
        samples.append(
            VisualizationSample(
                dice=dice,
                sample_index=sample_index,
                patient_id=patient_id,
                filename=batch.filenames[index],
                image=images_np[index],
                pred_mask=pred_masks_np[index],
                true_mask=true_masks_np[index],
                probability=probs_np[index],
            )
        )
        sample_index += 1
    return samples, sample_index


def _collect_visualization_patient_evidence(
    models_list: list[nn.Module],
    dataloader: DataLoader[Any],
    config: VisualizationExportConfig,
) -> tuple[dict[str, int], dict[str, float]]:
    positive_patch_counts: dict[str, int] = defaultdict(int)
    predicted_positive_pixels: dict[str, int] = defaultdict(int)
    total_pixels: dict[str, int] = defaultdict(int)
    processed_samples = 0
    progress = _progress_iterable(dataloader, desc="Visualization counts", unit="batch")
    try:
        for batch_data in progress:
            batch = _unpack_visualization_batch(batch_data)
            if batch is None or batch.images.shape[0] <= 0:
                continue
            images_norm = config.gpu_normalizer(batch.images.to(config.device, non_blocking=True))
            final_probs = compute_two_stream_probabilities(
                models_list,
                config.constituent_models_info,
                images_norm,
                roi_threshold=config.roi_threshold,
                roi_scale=config.roi_scale,
            )
            predictions = threshold_and_filter_components(
                final_probs.cpu().numpy().astype(np.float32),
                decision_threshold=config.decision_threshold,
                min_component_area_px=config.postprocessing_config.min_component_area_px,
                min_component_area_fraction_patch=(
                    config.postprocessing_config.min_component_area_fraction_patch
                ),
            )
            for index, patient_id in enumerate(batch.patient_ids):
                patient_id_str = str(patient_id)
                positive_patch_counts[patient_id_str] += count_positive_prediction_patches(
                    predictions[index]
                )
                predicted_positive_pixels[patient_id_str] += int(
                    np.count_nonzero(predictions[index])
                )
                total_pixels[patient_id_str] += int(predictions[index].size)
            processed_samples += batch.images.shape[0]
            progress.set_postfix(
                samples=processed_samples,
                patients=len(positive_patch_counts),
                refresh=False,
            )
    finally:
        progress.close()
    positive_area_fractions = {
        patient_id: (
            float(predicted_positive_pixels[patient_id] / total_pixels[patient_id])
            if total_pixels[patient_id] > 0
            else 0.0
        )
        for patient_id in positive_patch_counts
    }
    return positive_patch_counts, positive_area_fractions


def _update_ranked_samples(
    ranked_samples: list[VisualizationSample],
    candidates: list[VisualizationSample],
    *,
    num_samples: int,
) -> None:
    for candidate in candidates:
        if len(ranked_samples) < num_samples:
            ranked_samples.append(candidate)
            continue
        current_best_index = max(
            range(len(ranked_samples)),
            key=lambda selected_index: _visualization_sort_key(ranked_samples[selected_index]),
        )
        if _visualization_sort_key(candidate) < _visualization_sort_key(
            ranked_samples[current_best_index]
        ):
            ranked_samples[current_best_index] = candidate


def _render_visualization_sample(
    sample: VisualizationSample,
    *,
    rank: int,
    config: VisualizationExportConfig,
) -> Path:
    plt = load_headless_matplotlib_pyplot()
    mean = np.asarray(config.train_mean, dtype=np.float32)
    std = np.asarray(config.train_std, dtype=np.float32)
    image = sample.image.transpose(1, 2, 0)
    image = np.clip(std * image + mean, 0, 1)
    pred_overlay = np.where(sample.pred_mask == 0, np.nan, sample.pred_mask).astype(np.float32)
    true_overlay = np.where(sample.true_mask == 0, np.nan, sample.true_mask).astype(np.float32)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    title = f"Worst Dice #{rank}: {sample.dice:.4f} | {sample.patient_id}"
    if isinstance(sample.filename, str) and sample.filename.strip():
        title += f" | {sample.filename}"
    figure.suptitle(title)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(image)
    axes[1].imshow(pred_overlay, cmap="jet", alpha=0.5)
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(image)
    axes[2].imshow(true_overlay, cmap="jet", alpha=0.5)
    axes[2].set_title("True Mask")
    axes[3].imshow(sample.probability, vmin=0, vmax=1)
    axes[3].set_title("Probability")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path = config.output_dir / _build_visualization_filename(
        rank=rank,
        patient_id=sample.patient_id,
        filename=sample.filename if isinstance(sample.filename, str) else None,
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


@torch.inference_mode()
def export_visualizations(
    models_list: list[nn.Module],
    dataloader: DataLoader[Any],
    config: VisualizationExportConfig,
) -> list[Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.num_samples <= 0:
        return []
    ranked_samples: list[VisualizationSample] = []
    sample_index = 0
    positive_patch_counts, positive_area_fractions = _collect_visualization_patient_evidence(
        models_list,
        dataloader,
        config,
    )
    suppressed_patients = build_suppressed_patient_set(
        positive_patch_counts,
        min_patient_positive_patches=config.postprocessing_config.min_patient_positive_patches,
        positive_area_fractions=positive_area_fractions,
        min_patient_positive_area_fraction=(
            config.postprocessing_config.min_patient_positive_area_fraction
        ),
    )

    ranking_progress = _progress_iterable(dataloader, desc="Visualization ranking", unit="batch")
    try:
        for batch_data in ranking_progress:
            batch = _unpack_visualization_batch(batch_data)
            if batch is None:
                continue
            candidates, sample_index = _collect_visualization_candidates(
                models_list,
                batch,
                config=config,
                sample_index_start=sample_index,
                suppressed_patients=suppressed_patients,
            )
            _update_ranked_samples(ranked_samples, candidates, num_samples=config.num_samples)
            ranking_progress.set_postfix(
                samples=sample_index,
                selected=len(ranked_samples),
                refresh=False,
            )
    finally:
        ranking_progress.close()

    if not ranked_samples:
        return []

    selected_samples = sorted(ranked_samples, key=_visualization_sort_key)

    output_paths: list[Path] = []
    render_progress = _progress_iterable(
        selected_samples,
        desc="Visualization render",
        unit="sample",
    )
    try:
        for rank, sample in enumerate(render_progress, start=1):
            output_paths.append(_render_visualization_sample(sample, rank=rank, config=config))
            render_progress.set_postfix(done=rank, refresh=False)
    finally:
        render_progress.close()
    return output_paths
