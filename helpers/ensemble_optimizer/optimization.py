from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import optuna
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional
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

LOGGER = logging.getLogger(__name__)
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under LoggerWriter."""

    return sys.__stderr__


def _progress_disabled() -> bool:
    isatty = getattr(_progress_file(), "isatty", None)
    return not bool(isatty() if callable(isatty) else False)


class _StudyProgressCallback:
    """Advance a tqdm bar once per finished Optuna trial."""

    def __init__(self, progress_bar: tqdm[Any]) -> None:
        self._progress_bar = progress_bar

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial
        self._progress_bar.update(1)
        trials = getattr(study, "trials", [])
        completed = len(
            [
                finished_trial
                for finished_trial in trials
                if finished_trial.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        pruned = len(
            [
                finished_trial
                for finished_trial in trials
                if finished_trial.state == optuna.trial.TrialState.PRUNED
            ]
        )
        best_value = float(study.best_value) if completed > 0 else None
        postfix = {
            "done": str(completed),
            "pruned": str(pruned),
        }
        if best_value is not None and np.isfinite(best_value):
            postfix["best"] = f"{best_value:.4f}"
        self._progress_bar.set_postfix(postfix)


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in getattr(study, "trials", [])
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def _require_completed_trials(study: optuna.Study, *, stream_name: str) -> None:
    if _completed_trials(study):
        return
    raise RuntimeError(
        f"{stream_name} optimization produced no completed trials; all trials were pruned. "
        "Check ROI constraints and validation split."
    )


def _optuna_callbacks(progress_bar: tqdm[Any] | None) -> list[_StudyProgressCallback]:
    if progress_bar is None:
        return []
    return [_StudyProgressCallback(progress_bar)]


def _set_optuna_warning_verbosity() -> int:
    current_verbosity = int(optuna.logging.get_verbosity())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return current_verbosity


def _restore_optuna_verbosity(previous_verbosity: int) -> None:
    optuna.logging.set_verbosity(previous_verbosity)


@dataclass(frozen=True)
class OptimizationResult:
    semantic_indices: list[int]
    spatial_indices: list[int]
    semantic_weights: list[float]
    spatial_weights: list[float]
    roi_threshold: float
    decision_threshold: float
    calibration_metrics: dict[str, float | int | str]
    holdout_metrics: dict[str, float | int | str]


def get_stream_type(architecture: str, config: EnsembleOptimizerConfig) -> str:
    architecture_upper = architecture.upper()
    if architecture_upper in config.semantic_architectures:
        return "semantic"
    if architecture_upper in config.spatial_architectures:
        return "spatial"
    return "none"


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
    return functional.interpolate(mask_small, size=(height, width), mode="nearest").squeeze(1)


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
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    total_steps = len(models) * total_batches if total_batches is not None else None
    with tqdm(
        total=total_steps,
        desc="Cache preds",
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        file=_progress_file(),
        disable=_progress_disabled(),
    ) as progress_bar:
        for model_index, model in enumerate(models):
            architecture = str(getattr(model, "arch_name", "UNK"))
            progress_bar.set_postfix(
                {"model": f"{model_index + 1}/{len(models)}", "arch": architecture}
            )
            prediction_path = cache_dir / f"pred_{model_index}.dat"
            prediction_memmap = np.memmap(
                prediction_path,
                dtype="uint16",
                mode="w+",
                shape=(total_samples, height, width),
            )
            write_position = 0
            for batch in dataloader:
                if batch is None:
                    progress_bar.update(1)
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
                progress_bar.update(1)
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


def _compute_negative_false_positive_mass(
    patient_prediction: npt.NDArray[np.float32],
    patient_truth: npt.NDArray[np.uint8],
) -> float:
    if np.any(patient_truth):
        return 0.0
    return float(np.mean(patient_prediction))


def _compute_mcc(tp: float, fp: float, fn: float, tn: float) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / np.sqrt(denominator))


def _compute_patient_confusion_at_threshold(
    probabilities: npt.NDArray[np.float32],
    truth: npt.NDArray[np.uint8],
    roi_mask: npt.NDArray[np.uint8],
    threshold: float,
) -> tuple[int, int, int, int]:
    gated_probabilities = probabilities * roi_mask.astype(np.float32)
    predictions = gated_probabilities > threshold
    truth_bool = truth.astype(bool)
    tn = int(np.sum(~predictions & ~truth_bool))
    fn = int(np.sum(~predictions & truth_bool))
    fp = int(np.sum(predictions & ~truth_bool))
    tp = int(np.sum(predictions & truth_bool))
    return tp, fp, fn, tn


def _calibrate_decision_threshold(
    *,
    patient_ids: list[str],
    local_map: dict[str, slice],
    global_indices: npt.NDArray[np.int64],
    truth_memmap: np.memmap[Any, Any],
    prediction_memmaps: list[np.memmap[Any, Any]],
    semantic_indices: list[int],
    semantic_weights: list[float],
    spatial_indices: list[int],
    spatial_weights: list[float],
    roi_context_scale: int,
    roi_threshold: float,
) -> tuple[float, dict[str, float | int | str]]:
    if not patient_ids:
        return 0.5, {
            "Calibration_metric": "Patient_MCC",
            "Calibration_threshold": 0.5,
            "Calibration_best_mcc": 0.0,
            "Calibration_n_patients": 0,
            "Calibration_n_positive_patients": 0,
            "Calibration_n_negative_patients": 0,
        }

    patient_payloads: list[
        tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
    ] = []
    positive_patients = 0
    negative_patients = 0
    for patient_id in patient_ids:
        local_slice = local_map[patient_id]
        patient_global_indices = global_indices[local_slice]
        semantic_prediction = _weighted_ensemble_from_u16_cache(
            [prediction_memmaps[index][patient_global_indices] for index in semantic_indices],
            semantic_weights,
        )
        roi_mask = (
            generate_roi_batch(
                torch.from_numpy(semantic_prediction),
                roi_context_scale,
                roi_threshold,
            )
            .numpy()
            .astype(np.uint8)
        )
        spatial_prediction = _weighted_ensemble_from_u16_cache(
            [prediction_memmaps[index][patient_global_indices] for index in spatial_indices],
            spatial_weights,
        )
        patient_truth = truth_memmap[patient_global_indices].astype(np.uint8)
        if np.any(patient_truth):
            positive_patients += 1
        else:
            negative_patients += 1
        patient_payloads.append((spatial_prediction, patient_truth, roi_mask))

    best_threshold = 0.5
    best_mcc = float("-inf")
    for threshold in np.linspace(0.05, 0.95, 19):
        patient_scores: list[float] = []
        for spatial_prediction, patient_truth, roi_mask in patient_payloads:
            tp, fp, fn, tn = _compute_patient_confusion_at_threshold(
                spatial_prediction,
                patient_truth,
                roi_mask,
                float(threshold),
            )
            patient_scores.append(_compute_mcc(tp, fp, fn, tn))
        mean_mcc = float(np.mean(patient_scores)) if patient_scores else 0.0
        if mean_mcc > best_mcc:
            best_mcc = mean_mcc
            best_threshold = float(threshold)

    return best_threshold, {
        "Calibration_metric": "Patient_MCC",
        "Calibration_threshold": best_threshold,
        "Calibration_best_mcc": float(best_mcc if np.isfinite(best_mcc) else 0.0),
        "Calibration_n_patients": len(patient_ids),
        "Calibration_n_positive_patients": positive_patients,
        "Calibration_n_negative_patients": negative_patients,
    }


def run_two_stream_optimization(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    predefined_split: HoldoutSplit | None = None,
) -> OptimizationResult:
    LOGGER.info("Caching ensemble predictions for %s models.", len(models))
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
        calibration_frac=config.val_calibration_frac,
        holdout_frac=config.val_holdout_frac,
        seed=config.seed + 123,
    )
    holdout_idx, holdout_local_map, holdout_patients = build_indices_and_local_map(
        split.holdout_patients,
        patient_map,
    )
    calibration_idx, calibration_local_map, calibration_patients = build_indices_and_local_map(
        split.calibration_patients,
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

    LOGGER.info(
        "Prediction cache ready: %s samples across %s patients.",
        total_samples,
        len(patient_map),
    )
    LOGGER.info(
        "Stream allocation: semantic=%s, spatial=%s.",
        len(semantic_indices),
        len(spatial_indices),
    )

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

    previous_optuna_verbosity = _set_optuna_warning_verbosity()
    try:
        semantic_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )
        LOGGER.info(
            "Optimizing semantic stream over %s patients with %s trials.",
            len(optimization_patients),
            config.num_trials_semantic,
        )
        with tqdm(
            total=config.num_trials_semantic,
            desc="Semantic opt",
            leave=False,
            mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
            dynamic_ncols=True,
            file=_progress_file(),
            disable=_progress_disabled(),
        ) as semantic_progress:
            semantic_study.optimize(
                objective_semantic,
                n_trials=config.num_trials_semantic,
                callbacks=_optuna_callbacks(semantic_progress),
            )
        _require_completed_trials(semantic_study, stream_name="Semantic")
        LOGGER.info(
            "Semantic optimization complete: best_roi_threshold=%.4f best_objective=%.4f.",
            float(semantic_study.best_params["roi_thresh"]),
            float(semantic_study.best_value),
        )
        best_semantic_weights = _normalize_weights(
            [
                semantic_study.best_params.get(f"w_sem_{i}", 0.0)
                for i in range(len(semantic_indices))
            ]
        )
        best_roi_threshold = float(semantic_study.best_params["roi_thresh"])

        fixed_roi_mask = np.zeros((len(optimization_idx), height, width), dtype=np.uint8)
        chunk_size = 2048
        roi_chunk_count = max(1, int(np.ceil(len(optimization_idx) / chunk_size)))
        LOGGER.info("Building fixed ROI mask over %s chunk(s).", roi_chunk_count)
        with tqdm(
            total=roi_chunk_count,
            desc="Build ROI",
            leave=False,
            mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
            dynamic_ncols=True,
            file=_progress_file(),
            disable=_progress_disabled() or roi_chunk_count <= 1,
        ) as roi_progress:
            for start in range(0, len(optimization_idx), chunk_size):
                stop = min(start + chunk_size, len(optimization_idx))
                chunk_global_indices = optimization_idx[start:stop]
                chunk_u16 = [
                    prediction_memmaps[index][chunk_global_indices] for index in semantic_indices
                ]
                chunk_probs = _weighted_ensemble_from_u16_cache(chunk_u16, best_semantic_weights)
                chunk_roi = generate_roi_batch(
                    torch.from_numpy(chunk_probs),
                    config.roi_context_scale,
                    best_roi_threshold,
                )
                fixed_roi_mask[start:stop] = chunk_roi.numpy().astype(np.uint8)
                roi_progress.update(1)

        def objective_spatial(trial: optuna.Trial) -> float:
            weights = _normalize_weights(
                [trial.suggest_float(f"w_spa_{i}", 0.0, 1.0) for i in range(len(spatial_indices))]
            )
            positive_auprc_total = 0.0
            spill_total = 0.0
            evaluated_patients = 0
            evaluated_positive_patients = 0
            negative_fp_total = 0.0
            evaluated_negative_patients = 0
            for patient_id in optimization_patients:
                local_slice = optimization_local_map[patient_id]
                patient_truth = optimization_truth[local_slice]
                patient_roi = fixed_roi_mask[local_slice]
                is_positive_patient = bool(np.sum(patient_truth) > 0)
                if config.spatial_patient_policy == "positive_only" and not is_positive_patient:
                    continue
                global_indices = optimization_idx[local_slice]
                patient_u16 = [
                    prediction_memmaps[index][global_indices] for index in spatial_indices
                ]
                patient_prediction = _weighted_ensemble_from_u16_cache(patient_u16, weights)
                if is_positive_patient:
                    positive_auprc_total += compute_patient_auprc_in_roi(
                        patient_truth.ravel(),
                        patient_prediction.ravel(),
                        patient_roi.ravel(),
                    )
                    evaluated_positive_patients += 1
                else:
                    negative_fp_total += _compute_negative_false_positive_mass(
                        patient_prediction,
                        patient_truth,
                    )
                    evaluated_negative_patients += 1
                mass_total = float(np.sum(patient_prediction) + 1e-7)
                mass_outside = float(np.sum(patient_prediction * (1 - patient_roi)))
                spill_total += mass_outside / mass_total
                evaluated_patients += 1
            if evaluated_patients == 0:
                return 0.0
            macro_positive_auprc = (
                positive_auprc_total / evaluated_positive_patients
                if evaluated_positive_patients > 0
                else 0.0
            )
            macro_spill = spill_total / evaluated_patients
            macro_negative_fp = (
                negative_fp_total / evaluated_negative_patients
                if evaluated_negative_patients > 0
                else 0.0
            )
            return float(
                macro_positive_auprc
                - (config.spill_penalty_lambda * macro_spill)
                - (config.spill_penalty_lambda * macro_negative_fp)
            )

        spatial_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )
        LOGGER.info(
            "Optimizing spatial stream with policy=%s and %s trials.",
            config.spatial_patient_policy,
            config.num_trials_spatial,
        )
        with tqdm(
            total=config.num_trials_spatial,
            desc="Spatial opt",
            leave=False,
            mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
            dynamic_ncols=True,
            file=_progress_file(),
            disable=_progress_disabled(),
        ) as spatial_progress:
            spatial_study.optimize(
                objective_spatial,
                n_trials=config.num_trials_spatial,
                callbacks=_optuna_callbacks(spatial_progress),
            )
        _require_completed_trials(spatial_study, stream_name="Spatial")
        LOGGER.info(
            "Spatial optimization complete: best_objective=%.4f.",
            float(spatial_study.best_value),
        )
        best_spatial_weights = _normalize_weights(
            [spatial_study.best_params.get(f"w_spa_{i}", 0.0) for i in range(len(spatial_indices))]
        )
    finally:
        _restore_optuna_verbosity(previous_optuna_verbosity)
    LOGGER.info(
        "Calibrating decision threshold on %s patients.",
        len(calibration_patients),
    )
    decision_threshold, calibration_metrics = _calibrate_decision_threshold(
        patient_ids=calibration_patients,
        local_map=calibration_local_map,
        global_indices=calibration_idx,
        truth_memmap=truth_memmap,
        prediction_memmaps=prediction_memmaps,
        semantic_indices=semantic_indices,
        semantic_weights=best_semantic_weights,
        spatial_indices=spatial_indices,
        spatial_weights=best_spatial_weights,
        roi_context_scale=config.roi_context_scale,
        roi_threshold=best_roi_threshold,
    )
    LOGGER.info(
        "Calibration complete: threshold=%.4f best_mcc=%.4f.",
        decision_threshold,
        float(cast(float, calibration_metrics.get("Calibration_best_mcc", 0.0))),
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

    positive_auprc_total = 0.0
    spill_total = 0.0
    evaluated_patients = 0
    evaluated_positive_patients = 0
    negative_fp_total = 0.0
    evaluated_negative_patients = 0
    positive_patient_count = 0
    negative_patient_count = 0
    LOGGER.info("Evaluating holdout set across %s patients.", len(holdout_patients))
    with tqdm(
        holdout_patients,
        total=len(holdout_patients),
        desc="Eval holdout",
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        file=_progress_file(),
        disable=_progress_disabled(),
    ) as holdout_progress:
        for patient_id in holdout_progress:
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
            if is_positive_patient:
                positive_auprc_total += compute_patient_auprc_in_roi(
                    patient_truth.ravel(),
                    spatial_prediction.ravel(),
                    roi_mask.ravel(),
                )
                evaluated_positive_patients += 1
            else:
                negative_fp_total += _compute_negative_false_positive_mass(
                    spatial_prediction,
                    patient_truth,
                )
                evaluated_negative_patients += 1
            mass_total = float(np.sum(spatial_prediction) + 1e-7)
            mass_outside = float(np.sum(spatial_prediction * (1 - roi_mask)))
            spill_total += mass_outside / mass_total
            evaluated_patients += 1

    macro_positive_auprc = (
        float(positive_auprc_total / evaluated_positive_patients)
        if evaluated_positive_patients > 0
        else 0.0
    )
    macro_spill = float(spill_total / evaluated_patients) if evaluated_patients > 0 else 0.0
    macro_negative_fp = (
        float(negative_fp_total / evaluated_negative_patients)
        if evaluated_negative_patients > 0
        else 0.0
    )
    holdout_objective = float(
        macro_positive_auprc
        - (config.spill_penalty_lambda * macro_spill)
        - (config.spill_penalty_lambda * macro_negative_fp)
    )
    holdout_metrics: dict[str, float | int | str] = {
        "Macro_AUPRC_in_ROI": macro_positive_auprc,
        "Macro_AUPRC_in_ROI_Positive": macro_positive_auprc,
        "Macro_Spill": macro_spill,
        "Macro_Spill_All": macro_spill,
        "Macro_Negative_FP": macro_negative_fp,
        "Objective_AUPRC_minus_lambdaSpill": holdout_objective,
        "Objective_Composite": holdout_objective,
        "N_eval_patients": int(evaluated_patients),
        "N_eval_positive_patients": int(evaluated_positive_patients),
        "N_eval_negative_patients": int(evaluated_negative_patients),
        "N_pos_patients_total": int(positive_patient_count),
        "N_neg_patients_total": int(negative_patient_count),
        "Decision_threshold": decision_threshold,
        "Spatial_patient_policy": config.spatial_patient_policy,
        "Spill_lambda": float(config.spill_penalty_lambda),
    }
    LOGGER.info(
        "Holdout evaluation complete: objective=%.4f macro_auprc=%.4f spill=%.4f.",
        holdout_objective,
        macro_positive_auprc,
        macro_spill,
    )
    return OptimizationResult(
        semantic_indices=semantic_indices,
        spatial_indices=spatial_indices,
        semantic_weights=best_semantic_weights,
        spatial_weights=best_spatial_weights,
        roi_threshold=best_roi_threshold,
        decision_threshold=decision_threshold,
        calibration_metrics=calibration_metrics,
        holdout_metrics=holdout_metrics,
    )
