from __future__ import annotations

import gc
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import optuna
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.splitting import (
    HoldoutSplit,
    build_holdout_split,
    build_indices_and_local_map,
)
from helpers.training.gpu import GPUNormalizer
from helpers.training.runtime import autocast_ctx, setup_precision
from helpers.training.utils import clear_gpu


@dataclass(frozen=True)
class OptimizationResult:
    semantic_indices: list[int]
    spatial_indices: list[int]
    semantic_weights: list[float]
    spatial_weights: list[float]
    roi_threshold: float
    holdout_metrics: dict[str, float | int | str]


def get_stream_type(architecture: str, config: EnsembleOptimizerConfig) -> str:
    architecture_upper = architecture.upper()
    if architecture_upper in config.semantic_architectures:
        return "semantic"
    if architecture_upper in config.spatial_architectures:
        return "spatial"
    return "spatial"


def generate_roi_batch(
    probabilities: torch.Tensor,
    context_scale: int,
    threshold: float,
) -> torch.Tensor:
    _, height, width = probabilities.shape
    small = functional.interpolate(
        probabilities.unsqueeze(1),
        scale_factor=1.0 / context_scale,
        mode="bilinear",
        align_corners=False,
    )
    mask_small = (small > threshold).float()
    return cast(
        torch.Tensor,
        functional.interpolate(mask_small, size=(height, width), mode="nearest").squeeze(1),
    )


def compute_patient_auprc_in_roi(
    y_true: Any,
    y_pred: Any,
    roi_mask: Any,
) -> float:
    valid_indices = roi_mask > 0
    if not np.any(valid_indices):
        return 0.0
    y_true_roi = y_true[valid_indices]
    y_pred_roi = y_pred[valid_indices]
    if np.sum(y_true_roi) == 0:
        return 0.0
    try:
        return float(average_precision_score(y_true_roi, y_pred_roi))
    except Exception:
        return 0.0


@torch.inference_mode()
def predict_with_tta_batched(
    model: nn.Module, images: torch.Tensor, architecture: str
) -> torch.Tensor:
    amp_dtype, _, _ = setup_precision(architecture, amp_precision="auto")

    with autocast_ctx(images, amp_dtype):
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        probabilities = (
            torch.sigmoid(output).squeeze(1)
            if output.shape[1] == 1
            else torch.softmax(output, dim=1)[:, 1, :, :]
        )

    images_h = torch.flip(images, dims=[3])
    with autocast_ctx(images_h, amp_dtype):
        output_h = model(images_h)
        if isinstance(output_h, (tuple, list)):
            output_h = output_h[0]
        probabilities_h = (
            torch.sigmoid(output_h).squeeze(1)
            if output_h.shape[1] == 1
            else torch.softmax(output_h, dim=1)[:, 1, :, :]
        )
    probabilities.add_(torch.flip(probabilities_h, dims=[2]))
    del images_h, output_h, probabilities_h

    images_v = torch.flip(images, dims=[2])
    with autocast_ctx(images_v, amp_dtype):
        output_v = model(images_v)
        if isinstance(output_v, (tuple, list)):
            output_v = output_v[0]
        probabilities_v = (
            torch.sigmoid(output_v).squeeze(1)
            if output_v.shape[1] == 1
            else torch.softmax(output_v, dim=1)[:, 1, :, :]
        )
    probabilities.add_(torch.flip(probabilities_v, dims=[1]))
    probabilities.div_(3.0)
    return probabilities


def cache_predictions_sequential(
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    cache_dir: Path,
) -> tuple[list[Path], Path, Path, int, int, int, np.memmap[Any, Any]]:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(cast(Any, dataloader.dataset))
    batch_example = next(batch for batch in dataloader if batch is not None)
    images = batch_example[0]
    height = int(images.shape[2])
    width = int(images.shape[3])

    trues_path = cache_dir / "trues.dat"
    trues_memmap = np.memmap(
        trues_path, dtype="uint8", mode="w+", shape=(total_samples, height, width)
    )
    prediction_paths: list[Path] = []
    patient_ids: list[str] = []
    normalizer = GPUNormalizer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device,
    )

    first_pass = True
    last_written = 0
    for model_index, model in enumerate(models):
        architecture = str(getattr(model, "arch_name", "UNK"))
        prediction_path = cache_dir / f"pred_{model_index}.dat"
        prediction_memmap = np.memmap(
            prediction_path,
            dtype="uint16",
            mode="w+",
            shape=(total_samples, height, width),
        )
        write_position = 0
        for batch in tqdm(dataloader, desc=f"Infer {architecture}"):
            if batch is None:
                continue
            batch_images, batch_masks, batch_patient_ids = batch
            batch_size = int(batch_images.shape[0])
            if first_pass:
                trues_memmap[write_position : write_position + batch_size] = cast(
                    npt.NDArray[np.uint8],
                    batch_masks[:, 1, :, :].cpu().numpy().astype("uint8"),
                )
                patient_ids.extend(str(patient_id) for patient_id in batch_patient_ids)
            batch_images = normalizer(batch_images.to(device))
            predictions = predict_with_tta_batched(model, batch_images, architecture)
            predictions_np = predictions.detach().float().cpu().numpy()
            predictions_np = np.nan_to_num(predictions_np, nan=0.0, posinf=1.0, neginf=0.0)
            predictions_np = np.clip(predictions_np, 0.0, 1.0)
            prediction_memmap[write_position : write_position + batch_size] = (
                predictions_np * 65535
            ).astype(np.uint16)
            write_position += batch_size
        prediction_memmap.flush()
        model.cpu()
        clear_gpu()
        prediction_paths.append(prediction_path)
        last_written = write_position
        if first_pass:
            trues_memmap.flush()
            first_pass = False

    if last_written != total_samples:
        raise RuntimeError(
            "Cached "
            f"{last_written} samples but expected {total_samples}. "
            "Dataset produced invalid items."
        )

    pids_path = cache_dir / "pids.json"
    pids_path.write_text(json.dumps(patient_ids), encoding="utf-8")
    return prediction_paths, trues_path, pids_path, total_samples, height, width, trues_memmap


def _weighted_ensemble_from_u16_cache(
    u16_arrays: list[npt.NDArray[np.generic]], weights: list[float]
) -> npt.NDArray[np.float32]:
    accumulator: npt.NDArray[np.float32] | None = None
    scale = 1.0 / 65535.0
    for weight, values in zip(weights, u16_arrays, strict=False):
        if weight <= 1e-4:
            continue
        contribution = cast(npt.NDArray[np.float32], values.astype(np.float32) * (weight * scale))
        accumulator = contribution if accumulator is None else (accumulator + contribution)
    if accumulator is None:
        return cast(npt.NDArray[np.float32], np.zeros_like(u16_arrays[0], dtype=np.float32))
    return accumulator


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    total = float(sum(raw_weights)) + 1e-9
    return [float(weight / total) for weight in raw_weights]


def _compute_positive_patients(
    patient_map: dict[str, list[int]],
    truth_memmap: np.memmap[Any, Any],
) -> set[str]:
    positive_patients: set[str] = set()
    for patient_id, indices in patient_map.items():
        if any(np.any(truth_memmap[index]) for index in indices):
            positive_patients.add(patient_id)
    return positive_patients


def run_two_stream_optimization(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    predefined_split: HoldoutSplit | None = None,
) -> OptimizationResult:
    prediction_paths, _, pids_path, total_samples, height, width, truth_memmap = (
        cache_predictions_sequential(
            models,
            dataloader,
            device=device,
            cache_dir=config.pred_cache_dir,
        )
    )
    patient_ids = json.loads(pids_path.read_text(encoding="utf-8"))
    patient_map: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        patient_map[str(patient_id)].append(index)

    positive_patients = _compute_positive_patients(patient_map, truth_memmap)
    split = predefined_split or build_holdout_split(
        [str(patient_id) for patient_id in patient_ids],
        positive_patients,
        holdout_frac=config.val_holdout_frac,
        seed=config.seed + 123,
    )
    holdout_idx, holdout_local_map, holdout_patients = build_indices_and_local_map(
        split.holdout_patients,
        patient_map,
    )
    optimization_idx, optimization_local_map, optimization_patients = build_indices_and_local_map(
        split.optimization_patients,
        patient_map,
    )
    prediction_memmaps = [
        np.memmap(path, dtype="uint16", mode="r", shape=(total_samples, height, width))
        for path in prediction_paths
    ]

    semantic_indices = [
        index
        for index, model in enumerate(models)
        if get_stream_type(str(getattr(model, "arch_name", "")), config) == "semantic"
    ]
    spatial_indices = [
        index
        for index, model in enumerate(models)
        if get_stream_type(str(getattr(model, "arch_name", "")), config) == "spatial"
    ]
    if not semantic_indices:
        raise ValueError("No semantic models found based on configuration.")
    if not spatial_indices:
        spatial_indices = semantic_indices.copy()

    optimization_truth = truth_memmap[optimization_idx].astype(np.uint8)

    def objective_semantic(trial: optuna.Trial) -> float:
        weights = _normalize_weights(
            [trial.suggest_float(f"w_sem_{i}", 0.0, 1.0) for i in range(len(semantic_indices))]
        )
        roi_threshold = trial.suggest_float("roi_thresh", 0.15, 0.60)
        roi_area_fractions: list[float] = []
        roi_positive_recalls: list[float] = []
        empty_rois = 0
        for patient_id in optimization_patients:
            local_slice = optimization_local_map[patient_id]
            global_indices = optimization_idx[local_slice]
            patient_u16 = [prediction_memmaps[index][global_indices] for index in semantic_indices]
            patient_ensemble = _weighted_ensemble_from_u16_cache(patient_u16, weights)
            roi_mask = (
                generate_roi_batch(
                    torch.from_numpy(patient_ensemble),
                    config.roi_context_scale,
                    roi_threshold,
                )
                .numpy()
                .astype(np.uint8)
            )
            patient_truth = optimization_truth[local_slice]
            roi_area_fractions.append(float(np.mean(roi_mask)))
            if np.sum(roi_mask) == 0:
                empty_rois += 1
            if np.sum(patient_truth) > 0:
                intersection = np.sum((roi_mask == 1) & (patient_truth == 1))
                total_positive = np.sum(patient_truth)
                roi_positive_recalls.append(float(intersection / (total_positive + 1e-7)))
        median_area = float(np.median(roi_area_fractions)) if roi_area_fractions else 0.0
        empty_rate = float(empty_rois / max(1, len(optimization_patients)))
        mean_positive_recall = float(np.mean(roi_positive_recalls)) if roi_positive_recalls else 0.0
        if empty_rate > config.roi_empty_max:
            raise optuna.exceptions.TrialPruned(f"Trivial Empty: {empty_rate:.2f}")
        if mean_positive_recall < config.roi_min_pos_recall:
            raise optuna.exceptions.TrialPruned(f"Misses Positives: {mean_positive_recall:.2f}")
        gt_densities = [
            float(np.mean(optimization_truth[optimization_local_map[p]]))
            for p in optimization_patients
        ]
        mean_gt_density = float(np.mean(gt_densities)) if gt_densities else 0.0
        dynamic_max_median = max(config.roi_max_median, mean_gt_density + 0.15)
        if median_area > dynamic_max_median:
            raise optuna.exceptions.TrialPruned(
                f"Trivial Permissive: {median_area:.2f} > {dynamic_max_median:.2f}"
            )
        return mean_positive_recall

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    semantic_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    semantic_study.optimize(objective_semantic, n_trials=config.num_trials_semantic)
    best_semantic_weights = _normalize_weights(
        [semantic_study.best_params.get(f"w_sem_{i}", 0.0) for i in range(len(semantic_indices))]
    )
    best_roi_threshold = float(semantic_study.best_params["roi_thresh"])

    fixed_roi_mask = np.zeros((len(optimization_idx), height, width), dtype=np.uint8)
    chunk_size = 2048
    for start in range(0, len(optimization_idx), chunk_size):
        stop = min(start + chunk_size, len(optimization_idx))
        chunk_global_indices = optimization_idx[start:stop]
        chunk_u16 = [prediction_memmaps[index][chunk_global_indices] for index in semantic_indices]
        chunk_probs = _weighted_ensemble_from_u16_cache(chunk_u16, best_semantic_weights)
        chunk_roi = generate_roi_batch(
            torch.from_numpy(chunk_probs),
            config.roi_context_scale,
            best_roi_threshold,
        )
        fixed_roi_mask[start:stop] = chunk_roi.numpy().astype(np.uint8)

    def objective_spatial(trial: optuna.Trial) -> float:
        weights = _normalize_weights(
            [trial.suggest_float(f"w_spa_{i}", 0.0, 1.0) for i in range(len(spatial_indices))]
        )
        auprc_total = 0.0
        spill_total = 0.0
        evaluated_patients = 0
        for patient_id in optimization_patients:
            local_slice = optimization_local_map[patient_id]
            patient_truth = optimization_truth[local_slice]
            patient_roi = fixed_roi_mask[local_slice]
            is_positive_patient = bool(np.sum(patient_truth) > 0)
            if config.spatial_patient_policy == "positive_only" and not is_positive_patient:
                continue
            global_indices = optimization_idx[local_slice]
            patient_u16 = [prediction_memmaps[index][global_indices] for index in spatial_indices]
            patient_prediction = _weighted_ensemble_from_u16_cache(patient_u16, weights)
            auprc_total += compute_patient_auprc_in_roi(
                patient_truth.ravel(),
                patient_prediction.ravel(),
                patient_roi.ravel(),
            )
            mass_total = float(np.sum(patient_prediction) + 1e-7)
            mass_outside = float(np.sum(patient_prediction * (1 - patient_roi)))
            spill_total += mass_outside / mass_total
            evaluated_patients += 1
        if evaluated_patients == 0:
            return 0.0
        macro_auprc = auprc_total / evaluated_patients
        macro_spill = spill_total / evaluated_patients
        return float(macro_auprc - (config.spill_penalty_lambda * macro_spill))

    spatial_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    spatial_study.optimize(objective_spatial, n_trials=config.num_trials_spatial)
    best_spatial_weights = _normalize_weights(
        [spatial_study.best_params.get(f"w_spa_{i}", 0.0) for i in range(len(spatial_indices))]
    )

    def compute_ensemble_iterative(
        model_indices: list[int],
        weights: list[float],
        global_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        accumulator = cast(
            npt.NDArray[np.float32],
            np.zeros((len(global_indices), height, width), dtype=np.float32),
        )
        for model_index, weight in zip(model_indices, weights, strict=False):
            if weight <= 1e-5:
                continue
            chunk = prediction_memmaps[model_index][global_indices]
            accumulator += chunk.astype(np.float32) * (weight / 65535.0)
        return accumulator

    auprc_total = 0.0
    spill_total = 0.0
    evaluated_patients = 0
    positive_patient_count = 0
    negative_patient_count = 0
    for patient_id in tqdm(holdout_patients, desc="Eval Holdout"):
        gc.collect()
        local_slice = holdout_local_map[patient_id]
        global_indices = holdout_idx[local_slice]
        if len(global_indices) == 0:
            continue
        semantic_prediction = compute_ensemble_iterative(
            semantic_indices,
            best_semantic_weights,
            global_indices,
        )
        roi_mask = (
            generate_roi_batch(
                torch.from_numpy(semantic_prediction),
                config.roi_context_scale,
                best_roi_threshold,
            )
            .numpy()
            .astype(np.uint8)
        )
        spatial_prediction = compute_ensemble_iterative(
            spatial_indices, best_spatial_weights, global_indices
        )
        patient_truth = truth_memmap[global_indices].astype(np.uint8)
        is_positive_patient = bool(np.sum(patient_truth) > 0)
        if is_positive_patient:
            positive_patient_count += 1
        else:
            negative_patient_count += 1
        if config.spatial_patient_policy == "positive_only" and not is_positive_patient:
            continue
        auprc_total += compute_patient_auprc_in_roi(
            patient_truth.ravel(),
            spatial_prediction.ravel(),
            roi_mask.ravel(),
        )
        mass_total = float(np.sum(spatial_prediction) + 1e-7)
        mass_outside = float(np.sum(spatial_prediction * (1 - roi_mask)))
        spill_total += mass_outside / mass_total
        evaluated_patients += 1

    macro_auprc = float(auprc_total / evaluated_patients) if evaluated_patients > 0 else 0.0
    macro_spill = float(spill_total / evaluated_patients) if evaluated_patients > 0 else 0.0
    holdout_objective = float(macro_auprc - (config.spill_penalty_lambda * macro_spill))
    holdout_metrics: dict[str, float | int | str] = {
        "Macro_AUPRC_in_ROI": macro_auprc,
        "Macro_Spill": macro_spill,
        "Objective_AUPRC_minus_lambdaSpill": holdout_objective,
        "N_eval_patients": int(evaluated_patients),
        "N_pos_patients_total": int(positive_patient_count),
        "N_neg_patients_total": int(negative_patient_count),
        "Spatial_patient_policy": config.spatial_patient_policy,
        "Spill_lambda": float(config.spill_penalty_lambda),
    }
    return OptimizationResult(
        semantic_indices=semantic_indices,
        spatial_indices=spatial_indices,
        semantic_weights=best_semantic_weights,
        spatial_weights=best_spatial_weights,
        roi_threshold=best_roi_threshold,
        holdout_metrics=holdout_metrics,
    )
