from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
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
from helpers.ensemble_postprocessing import (
    PostprocessingConfig,
    confusion_counts_from_binary_masks,
    count_positive_prediction_patches,
    postprocessing_config_to_payload,
    threshold_and_filter_components,
    truth_pixel_counts,
)
from helpers.training.gpu import GPUNormalizer
from helpers.training.runtime import autocast_ctx, setup_precision
from helpers.training.utils import clear_gpu

LOGGER = logging.getLogger(__name__)
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5
_WEIGHT_SKIP_THRESHOLD = 1e-4
_BYTES_PER_GIB = 1024**3

ValidationDataloaderFactory = Callable[[set[str] | None], DataLoader[Any]]


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
    postprocessing_config: PostprocessingConfig
    calibration_metrics: dict[str, float | int | str]
    validation_calibration_summary: dict[str, Any]
    holdout_metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class CachePredictionRuntime:
    total_samples: int
    height: int
    width: int
    trues_path: Path
    pids_path: Path
    trues_memmap: np.memmap[Any, Any]
    normalizer: GPUNormalizer


@dataclass(frozen=True)
class ThresholdCalibrationConfig:
    patient_ids: list[str]
    local_map: dict[str, slice]
    global_indices: npt.NDArray[np.int64]
    truth_memmap: np.memmap[Any, Any]
    prediction_memmaps: list[np.memmap[Any, Any]]
    semantic_indices: list[int]
    semantic_weights: list[float]
    spatial_indices: list[int]
    spatial_weights: list[float]
    roi_context_scale: int
    roi_threshold: float
    optimizer_config: EnsembleOptimizerConfig | None = None


@dataclass(frozen=True)
class OptimizationPreparation:
    patient_map: dict[str, list[int]]
    prediction_memmaps: list[np.memmap[Any, Any]]
    truth_memmap: np.memmap[Any, Any]
    holdout_idx: npt.NDArray[np.int64]
    holdout_local_map: dict[str, slice]
    holdout_patients: list[str]
    calibration_idx: npt.NDArray[np.int64]
    calibration_local_map: dict[str, slice]
    calibration_patients: list[str]
    optimization_idx: npt.NDArray[np.int64]
    optimization_local_map: dict[str, slice]
    optimization_patients: list[str]
    semantic_indices: list[int]
    spatial_indices: list[int]
    optimization_truth: npt.NDArray[np.uint8] | None
    optimization_semantic_cache: dict[str, npt.NDArray[np.uint16]] | None
    optimization_spatial_cache: dict[str, npt.NDArray[np.uint16]] | None
    optimization_truth_cache: dict[str, npt.NDArray[np.uint8]] | None
    optimization_gt_density_by_patient: dict[str, float] | None
    optimization_positive_patients: set[str]
    height: int
    width: int


@dataclass(frozen=True)
class HoldoutEvaluation:
    macro_positive_auprc: float
    macro_spill: float
    holdout_objective: float
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class _Rule6CandidateKey:
    threshold: float
    min_component_area_px: int
    min_patient_positive_patches: int


@dataclass(frozen=True)
class _Rule6CandidateScore:
    key: _Rule6CandidateKey
    macro_rule6: float
    positive_dice: float
    positive_tpr: float
    negative_clean_rate: float
    positive_patient_count: int
    negative_patient_count: int


@dataclass(frozen=True)
class _Rule6CalibrationResult:
    decision_threshold: float
    postprocessing_config: PostprocessingConfig
    metrics: dict[str, float | int | str]
    summary: dict[str, Any]


@dataclass(frozen=True)
class CacheModelPredictionConfig:
    model: nn.Module
    dataloader: DataLoader[Any]
    model_index: int
    runtime: CachePredictionRuntime
    device: torch.device
    first_pass: bool
    patient_ids: list[str]
    progress_bar: tqdm[Any]
    cache_dir: Path


@dataclass(frozen=True)
class _OptimizationCacheBuildInput:
    prediction_memmaps: list[np.memmap[Any, Any]]
    truth_memmap: np.memmap[Any, Any]
    optimization_idx: npt.NDArray[np.int64]
    optimization_local_map: dict[str, slice]
    optimization_patients: list[str]
    semantic_indices: list[int]
    spatial_indices: list[int]
    height: int
    width: int
    max_cache_bytes: int


@dataclass(frozen=True)
class _SpatialObjectivePatientCache:
    is_positive_patient: bool
    roi_truth: npt.NDArray[np.uint8]
    roi_predictions: npt.NDArray[np.uint16]
    total_mass_by_model: npt.NDArray[np.float64]
    outside_mass_by_model: npt.NDArray[np.float64]
    negative_mean_by_model: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _StreamingPredictionContext:
    config: EnsembleOptimizerConfig
    dataloader_factory: ValidationDataloaderFactory
    models: list[nn.Module]
    semantic_indices: list[int]
    semantic_weights: list[float]
    spatial_indices: list[int]
    spatial_weights: list[float]
    device: torch.device
    roi_threshold: float


@dataclass(frozen=True)
class _StreamPreparationRequest:
    config: EnsembleOptimizerConfig
    prediction_paths: list[Path]
    pids_path: Path
    total_samples: int
    height: int
    width: int
    truth_memmap: np.memmap[Any, Any]
    stream: str


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
    return _compute_auprc_from_roi_vectors(y_true_roi, y_pred_roi)


def _compute_auprc_from_roi_vectors(
    y_true_roi: Any,
    y_pred_roi: Any,
) -> float:
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


def _initialize_prediction_cache(
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    cache_dir: Path,
) -> CachePredictionRuntime:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(cast(Any, dataloader.dataset))
    batch_example = next(batch for batch in dataloader if batch is not None)
    images = batch_example[0]
    height = int(images.shape[2])
    width = int(images.shape[3])
    trues_path = cache_dir / "trues.dat"
    pids_path = cache_dir / "pids.json"
    trues_memmap = np.memmap(
        trues_path,
        dtype="uint8",
        mode="w+",
        shape=(total_samples, height, width),
    )
    normalizer = GPUNormalizer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device,
    )
    return CachePredictionRuntime(
        total_samples=total_samples,
        height=height,
        width=width,
        trues_path=trues_path,
        pids_path=pids_path,
        trues_memmap=trues_memmap,
        normalizer=normalizer,
    )


def _cache_single_model_predictions(config: CacheModelPredictionConfig) -> tuple[Path, int, bool]:
    architecture = str(getattr(config.model, "arch_name", "UNK"))
    config.progress_bar.set_postfix({"model": f"{config.model_index + 1}", "arch": architecture})
    prediction_path = config.cache_dir / f"pred_{config.model_index}.dat"
    prediction_memmap = np.memmap(
        prediction_path,
        dtype="uint16",
        mode="w+",
        shape=(config.runtime.total_samples, config.runtime.height, config.runtime.width),
    )
    write_position = 0
    for batch in config.dataloader:
        if batch is None:
            config.progress_bar.update(1)
            continue
        batch_images, batch_masks, batch_patient_ids = batch
        batch_size = int(batch_images.shape[0])
        if config.first_pass:
            config.runtime.trues_memmap[write_position : write_position + batch_size] = cast(
                npt.NDArray[np.uint8],
                batch_masks[:, 1, :, :].cpu().numpy().astype("uint8"),
            )
            config.patient_ids.extend(str(patient_id) for patient_id in batch_patient_ids)
        batch_images = config.runtime.normalizer(batch_images.to(config.device))
        predictions = predict_with_tta_batched(config.model, batch_images, architecture)
        predictions_np = predictions.detach().float().cpu().numpy()
        predictions_np = np.nan_to_num(predictions_np, nan=0.0, posinf=1.0, neginf=0.0)
        predictions_np = np.clip(predictions_np, 0.0, 1.0)
        prediction_memmap[write_position : write_position + batch_size] = (
            predictions_np * 65535
        ).astype(np.uint16)
        write_position += batch_size
        config.progress_bar.update(1)
    prediction_memmap.flush()
    config.model.cpu()
    clear_gpu()
    if config.first_pass:
        config.runtime.trues_memmap.flush()
    return prediction_path, write_position, False


def cache_predictions_sequential(
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    cache_dir: Path,
) -> tuple[list[Path], Path, Path, int, int, int, np.memmap[Any, Any]]:
    runtime = _initialize_prediction_cache(dataloader, device=device, cache_dir=cache_dir)
    prediction_paths: list[Path] = []
    patient_ids: list[str] = []

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
            prediction_path, last_written, first_pass = _cache_single_model_predictions(
                CacheModelPredictionConfig(
                    model=model,
                    dataloader=dataloader,
                    model_index=model_index,
                    runtime=runtime,
                    device=device,
                    first_pass=first_pass,
                    patient_ids=patient_ids,
                    progress_bar=progress_bar,
                    cache_dir=cache_dir,
                )
            )
            prediction_paths.append(prediction_path)

    if last_written != runtime.total_samples:
        raise RuntimeError(
            "Cached "
            f"{last_written} samples but expected {runtime.total_samples}. "
            "Dataset produced invalid items."
        )

    runtime.pids_path.write_text(json.dumps(patient_ids), encoding="utf-8")
    return (
        prediction_paths,
        runtime.trues_path,
        runtime.pids_path,
        runtime.total_samples,
        runtime.height,
        runtime.width,
        runtime.trues_memmap,
    )


def _close_memmap(memmap: np.memmap[Any, Any]) -> None:
    memmap.flush()
    mmap_handle = getattr(memmap, "_mmap", None)
    close = getattr(mmap_handle, "close", None)
    if callable(close):
        close()


def _close_prediction_memmaps(memmaps: list[np.memmap[Any, Any]]) -> None:
    for memmap in memmaps:
        _close_memmap(memmap)


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _open_prediction_memmaps(
    prediction_paths: list[Path],
    *,
    total_samples: int,
    height: int,
    width: int,
) -> list[np.memmap[Any, Any]]:
    return [
        np.memmap(path, dtype="uint16", mode="r", shape=(total_samples, height, width))
        for path in prediction_paths
    ]


def _cache_prediction_group(
    *,
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    model_indices: list[int],
    dataloader: DataLoader[Any],
    device: torch.device,
    cache_name: str,
) -> tuple[list[Path], Path, Path, int, int, int, np.memmap[Any, Any]]:
    selected_models = [models[index] for index in model_indices]
    return cache_predictions_sequential(
        selected_models,
        dataloader,
        device=device,
        cache_dir=config.pred_cache_dir / cache_name,
    )


def _cache_bytes_to_gib(byte_count: int) -> float:
    return byte_count / _BYTES_PER_GIB


def _log_staged_cache_estimate(
    config: EnsembleOptimizerConfig,
    *,
    optimization_samples: int,
    height: int,
    width: int,
    semantic_model_count: int,
    spatial_model_count: int,
) -> None:
    truth_bytes = optimization_samples * height * width
    fixed_roi_bytes = truth_bytes
    semantic_bytes = truth_bytes + (
        optimization_samples * height * width * 2 * semantic_model_count
    )
    spatial_bytes = (
        truth_bytes
        + fixed_roi_bytes
        + (optimization_samples * height * width * 2 * spatial_model_count)
    )
    peak_bytes = max(semantic_bytes, spatial_bytes)
    LOGGER.info(
        "Staged prediction cache estimate: optimization_samples=%s, shape=%sx%s, "
        "semantic_peak=%.2f GiB, spatial_peak=%.2f GiB, estimated_peak=%.2f GiB, cache_dir=%s.",
        optimization_samples,
        height,
        width,
        _cache_bytes_to_gib(semantic_bytes),
        _cache_bytes_to_gib(spatial_bytes),
        _cache_bytes_to_gib(peak_bytes),
        config.pred_cache_dir,
    )
    try:
        free_bytes = shutil.disk_usage(config.pred_cache_dir.parent).free
    except OSError:
        return
    if peak_bytes > free_bytes:
        LOGGER.warning(
            "Estimated staged prediction-cache peak %.2f GiB exceeds available free space "
            "%.2f GiB under %s.",
            _cache_bytes_to_gib(peak_bytes),
            _cache_bytes_to_gib(free_bytes),
            config.pred_cache_dir.parent,
        )


def _weighted_ensemble_from_u16_cache(
    u16_arrays: list[npt.NDArray[np.generic]], weights: list[float]
) -> npt.NDArray[np.float32]:
    accumulator: npt.NDArray[np.float32] | None = None
    scale = 1.0 / 65535.0
    for weight, values in zip(weights, u16_arrays, strict=False):
        if weight <= _WEIGHT_SKIP_THRESHOLD:
            continue
        contribution = cast(npt.NDArray[np.float32], values.astype(np.float32) * (weight * scale))
        accumulator = contribution if accumulator is None else (accumulator + contribution)
    if accumulator is None:
        return cast(npt.NDArray[np.float32], np.zeros_like(u16_arrays[0], dtype=np.float32))
    return accumulator


def _effective_weight_array(weights: list[float]) -> npt.NDArray[np.float32]:
    effective_weights = np.asarray(weights, dtype=np.float32)
    effective_weights[effective_weights <= _WEIGHT_SKIP_THRESHOLD] = 0.0
    return effective_weights


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


def _dice_from_counts(counts: dict[str, int]) -> float:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    if tp + fn == 0.0:
        return 1.0 if tp + fp == 0.0 else 0.0
    denominator = (2.0 * tp) + fp + fn
    return float((2.0 * tp) / denominator) if denominator > 0.0 else 0.0


def _tpr_from_counts(counts: dict[str, int]) -> float:
    tp = float(counts["tp"])
    fn = float(counts["fn"])
    denominator = tp + fn
    return float(tp / denominator) if denominator > 0.0 else 0.0


def _rule6_threshold_values(config: EnsembleOptimizerConfig) -> tuple[float, ...]:
    values: list[float] = []
    current = config.decision_threshold_min
    while current <= config.decision_threshold_max + (config.decision_threshold_step / 2.0):
        values.append(round(float(current), 6))
        current += config.decision_threshold_step
    return tuple(values)


def _rule6_candidate_keys(config: EnsembleOptimizerConfig) -> tuple[_Rule6CandidateKey, ...]:
    return tuple(
        _Rule6CandidateKey(
            threshold=threshold,
            min_component_area_px=min_component_area_px,
            min_patient_positive_patches=min_patient_positive_patches,
        )
        for threshold in _rule6_threshold_values(config)
        for min_component_area_px in config.min_component_area_px_candidates
        for min_patient_positive_patches in config.min_patient_positive_patches_candidates
    )


def _zero_rule6_calibration_result(config: EnsembleOptimizerConfig) -> _Rule6CalibrationResult:
    postprocessing_config = PostprocessingConfig(
        min_component_area_px=0,
        min_patient_positive_patches=1,
    )
    summary: dict[str, Any] = {
        "objective": "balanced_rule6",
        "status": "no_calibration_patients",
        "candidate_grid": _rule6_candidate_grid_summary(config),
        "selected": {
            "decision_threshold": float(config.decision_threshold_min),
            **postprocessing_config_to_payload(postprocessing_config),
        },
    }
    return _Rule6CalibrationResult(
        decision_threshold=float(config.decision_threshold_min),
        postprocessing_config=postprocessing_config,
        metrics={
            "Calibration_metric": "Macro_Rule6_Dice",
            "Calibration_objective": "balanced_rule6",
            "Calibration_threshold": float(config.decision_threshold_min),
            "Calibration_macro_rule6": 0.0,
            "Calibration_positive_dice": 0.0,
            "Calibration_positive_tpr": 0.0,
            "Calibration_negative_clean_rate": 0.0,
            "Calibration_baseline_positive_dice": 0.0,
            "Calibration_baseline_positive_tpr": 0.0,
            "Calibration_positive_dice_drop_tolerance": float(config.pos_dice_drop_tolerance),
            "Calibration_positive_tpr_drop_tolerance": float(config.pos_tpr_drop_tolerance),
            "Calibration_min_component_area_px": 0,
            "Calibration_min_patient_positive_patches": 1,
            "Calibration_n_patients": 0,
            "Calibration_n_positive_patients": 0,
            "Calibration_n_negative_patients": 0,
        },
        summary=summary,
    )


def _rule6_candidate_grid_summary(config: EnsembleOptimizerConfig) -> dict[str, Any]:
    thresholds = _rule6_threshold_values(config)
    return {
        "decision_threshold_min": float(config.decision_threshold_min),
        "decision_threshold_max": float(config.decision_threshold_max),
        "decision_threshold_step": float(config.decision_threshold_step),
        "decision_threshold_values": [float(value) for value in thresholds],
        "min_component_area_px_candidates": [
            int(value) for value in config.min_component_area_px_candidates
        ],
        "min_patient_positive_patches_candidates": [
            int(value) for value in config.min_patient_positive_patches_candidates
        ],
        "candidate_count": int(
            len(thresholds)
            * len(config.min_component_area_px_candidates)
            * len(config.min_patient_positive_patches_candidates)
        ),
    }


def _score_rule6_candidate(
    *,
    key: _Rule6CandidateKey,
    positive_dice_total: float,
    positive_tpr_total: float,
    negative_clean_total: float,
    positive_patient_count: int,
    negative_patient_count: int,
) -> _Rule6CandidateScore:
    total_patients = positive_patient_count + negative_patient_count
    positive_dice = (
        float(positive_dice_total / positive_patient_count) if positive_patient_count > 0 else 0.0
    )
    positive_tpr = (
        float(positive_tpr_total / positive_patient_count) if positive_patient_count > 0 else 0.0
    )
    negative_clean_rate = (
        float(negative_clean_total / negative_patient_count) if negative_patient_count > 0 else 0.0
    )
    macro_rule6 = (
        float((positive_dice_total + negative_clean_total) / total_patients)
        if total_patients > 0
        else 0.0
    )
    return _Rule6CandidateScore(
        key=key,
        macro_rule6=macro_rule6,
        positive_dice=positive_dice,
        positive_tpr=positive_tpr,
        negative_clean_rate=negative_clean_rate,
        positive_patient_count=positive_patient_count,
        negative_patient_count=negative_patient_count,
    )


def _candidate_sort_key(
    candidate: _Rule6CandidateScore,
) -> tuple[float, float, float, float, float, int, float]:
    return (
        candidate.macro_rule6,
        candidate.negative_clean_rate,
        candidate.positive_tpr,
        candidate.positive_dice,
        -float(candidate.key.min_component_area_px),
        -int(candidate.key.min_patient_positive_patches),
        -float(candidate.key.threshold),
    )


def _candidate_to_summary(candidate: _Rule6CandidateScore) -> dict[str, float | int]:
    return {
        "decision_threshold": float(candidate.key.threshold),
        "min_component_area_px": int(candidate.key.min_component_area_px),
        "min_patient_positive_patches": int(candidate.key.min_patient_positive_patches),
        "macro_rule6": float(candidate.macro_rule6),
        "positive_dice": float(candidate.positive_dice),
        "positive_tpr": float(candidate.positive_tpr),
        "negative_clean_rate": float(candidate.negative_clean_rate),
        "positive_patient_count": int(candidate.positive_patient_count),
        "negative_patient_count": int(candidate.negative_patient_count),
    }


def _calibrate_rule6_from_patient_predictions(
    *,
    optimizer_config: EnsembleOptimizerConfig,
    patient_predictions: Iterable[
        tuple[str, npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
    ],
) -> _Rule6CalibrationResult:
    candidate_keys = _rule6_candidate_keys(optimizer_config)
    positive_dice_totals = np.zeros(len(candidate_keys), dtype=np.float64)
    positive_tpr_totals = np.zeros(len(candidate_keys), dtype=np.float64)
    negative_clean_totals = np.zeros(len(candidate_keys), dtype=np.float64)
    positive_patient_count = 0
    negative_patient_count = 0
    processed_patients = 0

    threshold_values = _rule6_threshold_values(optimizer_config)
    area_candidates = optimizer_config.min_component_area_px_candidates
    patch_candidates = optimizer_config.min_patient_positive_patches_candidates
    candidate_index = {key: index for index, key in enumerate(candidate_keys)}

    for _patient_id, spatial_prediction, patient_truth, roi_mask in patient_predictions:
        processed_patients += 1
        gated_prediction = cast(
            npt.NDArray[np.float32],
            spatial_prediction.astype(np.float32) * roi_mask.astype(np.float32),
        )
        truth_counts = truth_pixel_counts(patient_truth)
        is_positive_patient = truth_counts["positive"] > 0
        if is_positive_patient:
            positive_patient_count += 1
        else:
            negative_patient_count += 1

        suppressed_counts = {
            "tp": 0,
            "fp": 0,
            "fn": int(truth_counts["positive"]),
            "tn": int(truth_counts["negative"]),
        }
        for threshold in threshold_values:
            for min_component_area_px in area_candidates:
                predictions = threshold_and_filter_components(
                    gated_prediction,
                    decision_threshold=threshold,
                    min_component_area_px=min_component_area_px,
                )
                positive_patch_count = count_positive_prediction_patches(predictions)
                unsuppressed_counts = confusion_counts_from_binary_masks(
                    predictions,
                    patient_truth,
                )
                for min_patient_positive_patches in patch_candidates:
                    key = _Rule6CandidateKey(
                        threshold=threshold,
                        min_component_area_px=min_component_area_px,
                        min_patient_positive_patches=min_patient_positive_patches,
                    )
                    index = candidate_index[key]
                    counts = (
                        suppressed_counts
                        if positive_patch_count < min_patient_positive_patches
                        else unsuppressed_counts
                    )
                    if is_positive_patient:
                        positive_dice_totals[index] += _dice_from_counts(counts)
                        positive_tpr_totals[index] += _tpr_from_counts(counts)
                    else:
                        negative_clean_totals[index] += 1.0 if counts["fp"] == 0 else 0.0

    if processed_patients == 0:
        return _zero_rule6_calibration_result(optimizer_config)
    if positive_patient_count == 0 or negative_patient_count == 0:
        raise RuntimeError(
            "balanced_rule6 calibration requires at least one positive and one negative "
            "calibration patient."
        )

    scores = [
        _score_rule6_candidate(
            key=key,
            positive_dice_total=float(positive_dice_totals[index]),
            positive_tpr_total=float(positive_tpr_totals[index]),
            negative_clean_total=float(negative_clean_totals[index]),
            positive_patient_count=positive_patient_count,
            negative_patient_count=negative_patient_count,
        )
        for index, key in enumerate(candidate_keys)
    ]
    # Anchor sensitivity guardrails to the least-filtered baseline, not to the
    # best unfiltered candidate selected by the same threshold sweep.
    baseline_key = _Rule6CandidateKey(
        threshold=threshold_values[0],
        min_component_area_px=0,
        min_patient_positive_patches=1,
    )
    baseline = scores[candidate_index[baseline_key]]
    minimum_allowed_positive_dice = (
        baseline.positive_dice - optimizer_config.pos_dice_drop_tolerance
    )
    minimum_allowed_positive_tpr = baseline.positive_tpr - optimizer_config.pos_tpr_drop_tolerance
    accepted_candidates = [
        score
        for score in scores
        if score.positive_dice >= minimum_allowed_positive_dice
        and score.positive_tpr >= minimum_allowed_positive_tpr
    ]
    selected = max(accepted_candidates, key=_candidate_sort_key)
    postprocessing_config = PostprocessingConfig(
        min_component_area_px=selected.key.min_component_area_px,
        min_patient_positive_patches=selected.key.min_patient_positive_patches,
    )
    summary = {
        "objective": "balanced_rule6",
        "positive_dice_drop_tolerance": float(optimizer_config.pos_dice_drop_tolerance),
        "positive_tpr_drop_tolerance": float(optimizer_config.pos_tpr_drop_tolerance),
        "minimum_allowed_positive_dice": float(minimum_allowed_positive_dice),
        "minimum_allowed_positive_tpr": float(minimum_allowed_positive_tpr),
        "candidate_grid": _rule6_candidate_grid_summary(optimizer_config),
        "baseline_unfiltered": _candidate_to_summary(baseline),
        "selected": _candidate_to_summary(selected),
        "postprocessing_config": postprocessing_config_to_payload(postprocessing_config),
        "accepted_candidate_count": int(len(accepted_candidates)),
        "rejected_candidate_count": int(len(scores) - len(accepted_candidates)),
    }
    return _Rule6CalibrationResult(
        decision_threshold=float(selected.key.threshold),
        postprocessing_config=postprocessing_config,
        metrics={
            "Calibration_metric": "Macro_Rule6_Dice",
            "Calibration_objective": "balanced_rule6",
            "Calibration_threshold": float(selected.key.threshold),
            "Calibration_macro_rule6": float(selected.macro_rule6),
            "Calibration_positive_dice": float(selected.positive_dice),
            "Calibration_positive_tpr": float(selected.positive_tpr),
            "Calibration_negative_clean_rate": float(selected.negative_clean_rate),
            "Calibration_baseline_positive_dice": float(baseline.positive_dice),
            "Calibration_baseline_positive_tpr": float(baseline.positive_tpr),
            "Calibration_positive_dice_drop_tolerance": float(
                optimizer_config.pos_dice_drop_tolerance
            ),
            "Calibration_positive_tpr_drop_tolerance": float(
                optimizer_config.pos_tpr_drop_tolerance
            ),
            "Calibration_min_component_area_px": int(selected.key.min_component_area_px),
            "Calibration_min_patient_positive_patches": int(
                selected.key.min_patient_positive_patches
            ),
            "Calibration_n_patients": int(positive_patient_count + negative_patient_count),
            "Calibration_n_positive_patients": int(positive_patient_count),
            "Calibration_n_negative_patients": int(negative_patient_count),
            "Calibration_accepted_candidate_count": int(len(accepted_candidates)),
            "Calibration_rejected_candidate_count": int(len(scores) - len(accepted_candidates)),
        },
        summary=summary,
    )


def _calibrate_decision_threshold(
    config: ThresholdCalibrationConfig,
) -> _Rule6CalibrationResult:
    optimizer_config = config.optimizer_config or EnsembleOptimizerConfig(
        master_manifest_path=Path("."),
        metadata_dir=Path("."),
        output_dir=Path("."),
        local_data_dir=Path("."),
        pred_cache_dir=Path("."),
        stage_input_locally=True,
        overwrite_output=True,
        seed=24,
        batch_size=32,
        workers=1,
        sort_metric="best_val_auprc_pixel_score",
        val_calibration_frac=0.25,
        val_holdout_frac=0.20,
        semantic_architectures=("SWIN",),
        spatial_architectures=("FPN",),
        roi_context_scale=config.roi_context_scale,
        roi_max_median=0.60,
        roi_empty_max=0.50,
        roi_min_pos_recall=0.80,
        spill_penalty_lambda=0.10,
        spatial_patient_policy="all",
        num_trials_semantic=1,
        num_trials_spatial=1,
    )

    return _calibrate_rule6_from_patient_predictions(
        optimizer_config=optimizer_config,
        patient_predictions=_iter_cached_rule6_patient_predictions(config),
    )


def _iter_cached_rule6_patient_predictions(
    config: ThresholdCalibrationConfig,
) -> Iterable[tuple[str, npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
    for patient_id in config.patient_ids:
        local_slice = config.local_map[patient_id]
        patient_global_indices = config.global_indices[local_slice]
        semantic_prediction = _weighted_ensemble_from_u16_cache(
            [
                config.prediction_memmaps[index][patient_global_indices]
                for index in config.semantic_indices
            ],
            config.semantic_weights,
        )
        roi_mask = (
            generate_roi_batch(
                torch.from_numpy(semantic_prediction),
                config.roi_context_scale,
                config.roi_threshold,
            )
            .numpy()
            .astype(np.uint8)
        )
        spatial_prediction = _weighted_ensemble_from_u16_cache(
            [
                config.prediction_memmaps[index][patient_global_indices]
                for index in config.spatial_indices
            ],
            config.spatial_weights,
        )
        patient_truth = config.truth_memmap[patient_global_indices].astype(np.uint8)
        yield patient_id, spatial_prediction, patient_truth, roi_mask


def _collect_truth_from_dataloader(dataloader: DataLoader[Any]) -> npt.NDArray[np.uint8]:
    truth_batches: list[npt.NDArray[np.uint8]] = []
    for batch in dataloader:
        if batch is None:
            continue
        _batch_images, batch_masks, _batch_patient_ids = batch
        truth_batches.append(
            cast(
                npt.NDArray[np.uint8],
                batch_masks[:, 1, :, :].cpu().numpy().astype("uint8"),
            )
        )
    if not truth_batches:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return cast(npt.NDArray[np.uint8], np.concatenate(truth_batches, axis=0))


def _batch_predictions_to_u16(predictions: torch.Tensor) -> npt.NDArray[np.uint16]:
    predictions_np = predictions.detach().float().cpu().numpy()
    predictions_np = np.nan_to_num(predictions_np, nan=0.0, posinf=1.0, neginf=0.0)
    predictions_np = np.clip(predictions_np, 0.0, 1.0)
    return cast(npt.NDArray[np.uint16], (predictions_np * 65535).astype(np.uint16))


def _stream_weighted_prediction_for_patient(
    *,
    models: list[nn.Module],
    model_indices: list[int],
    weights: list[float],
    dataloader: DataLoader[Any],
    device: torch.device,
    truth: npt.NDArray[np.uint8] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    truth = truth if truth is not None else _collect_truth_from_dataloader(dataloader)
    if truth.size == 0:
        return np.zeros((0, 0, 0), dtype=np.float32), truth
    accumulator = cast(npt.NDArray[np.float32], np.zeros(truth.shape, dtype=np.float32))
    normalizer = GPUNormalizer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device,
    )
    for model_index, weight in zip(model_indices, weights, strict=False):
        if weight <= _WEIGHT_SKIP_THRESHOLD:
            continue
        model = models[model_index].to(device)
        architecture = str(getattr(model, "arch_name", "UNK"))
        write_position = 0
        for batch in dataloader:
            if batch is None:
                continue
            batch_images, _batch_masks, _batch_patient_ids = batch
            batch_size = int(batch_images.shape[0])
            batch_images = normalizer(batch_images.to(device))
            predictions = predict_with_tta_batched(model, batch_images, architecture)
            predictions_u16 = _batch_predictions_to_u16(predictions)
            accumulator[write_position : write_position + batch_size] += predictions_u16.astype(
                np.float32
            ) * (weight / 65535.0)
            write_position += batch_size
        if write_position != truth.shape[0]:
            raise RuntimeError(
                "Streamed "
                f"{write_position} samples but expected {truth.shape[0]} for model "
                f"{model_index}."
            )
        model.cpu()
        clear_gpu()
    return accumulator, truth


def _calibrate_decision_threshold_streaming(
    context: _StreamingPredictionContext,
    *,
    patient_ids: list[str],
) -> _Rule6CalibrationResult:
    return _calibrate_rule6_from_patient_predictions(
        optimizer_config=context.config,
        patient_predictions=_iter_streaming_rule6_patient_predictions(
            context,
            patient_ids=patient_ids,
        ),
    )


def _iter_streaming_rule6_patient_predictions(
    context: _StreamingPredictionContext,
    *,
    patient_ids: list[str],
) -> Iterable[tuple[str, npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
    for patient_id in patient_ids:
        dataloader = context.dataloader_factory({patient_id})
        semantic_prediction, patient_truth = _stream_weighted_prediction_for_patient(
            models=context.models,
            model_indices=context.semantic_indices,
            weights=context.semantic_weights,
            dataloader=dataloader,
            device=context.device,
        )
        spatial_prediction, _patient_truth = _stream_weighted_prediction_for_patient(
            models=context.models,
            model_indices=context.spatial_indices,
            weights=context.spatial_weights,
            dataloader=dataloader,
            device=context.device,
            truth=patient_truth,
        )
        roi_mask = (
            generate_roi_batch(
                torch.from_numpy(semantic_prediction),
                context.config.roi_context_scale,
                context.roi_threshold,
            )
            .numpy()
            .astype(np.uint8)
        )
        yield patient_id, spatial_prediction, patient_truth, roi_mask


def _build_patient_map(patient_ids: list[str]) -> dict[str, list[int]]:
    patient_map: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        patient_map[str(patient_id)].append(index)
    return patient_map


def _estimate_optimization_cache_bytes(
    *,
    optimization_idx: npt.NDArray[np.int64],
    semantic_model_count: int,
    spatial_model_count: int,
    height: int,
    width: int,
) -> int:
    sample_count = len(optimization_idx)
    prediction_bytes = sample_count * height * width * 2
    truth_bytes = sample_count * height * width
    return prediction_bytes * (semantic_model_count + spatial_model_count) + truth_bytes


def _stack_prediction_rows(
    prediction_memmaps: list[np.memmap[Any, Any]],
    *,
    model_indices: list[int],
    global_indices: npt.NDArray[np.int64],
    height: int,
    width: int,
) -> npt.NDArray[np.uint16]:
    if not model_indices:
        return np.zeros((0, len(global_indices), height, width), dtype=np.uint16)
    return np.stack(
        [
            np.asarray(prediction_memmaps[model_index][global_indices], dtype=np.uint16)
            for model_index in model_indices
        ],
        axis=0,
    )


def _build_optimization_patient_caches(
    request: _OptimizationCacheBuildInput,
) -> tuple[
    dict[str, npt.NDArray[np.uint16]] | None,
    dict[str, npt.NDArray[np.uint16]] | None,
    dict[str, npt.NDArray[np.uint8]] | None,
    dict[str, float] | None,
]:
    estimated_bytes = _estimate_optimization_cache_bytes(
        optimization_idx=request.optimization_idx,
        semantic_model_count=len(request.semantic_indices),
        spatial_model_count=len(request.spatial_indices),
        height=request.height,
        width=request.width,
    )
    if request.max_cache_bytes == 0:
        LOGGER.info("Skipping optimization patient cache: cache disabled by configuration.")
        return None, None, None, None
    if estimated_bytes > request.max_cache_bytes:
        LOGGER.info(
            "Skipping optimization patient cache: estimated footprint %.2f MiB "
            "exceeds limit %.2f MiB.",
            estimated_bytes / (1024 * 1024),
            request.max_cache_bytes / (1024 * 1024),
        )
        return None, None, None, None

    semantic_cache: dict[str, npt.NDArray[np.uint16]] = {}
    spatial_cache: dict[str, npt.NDArray[np.uint16]] = {}
    truth_cache: dict[str, npt.NDArray[np.uint8]] = {}
    gt_density_by_patient: dict[str, float] = {}
    for patient_id in request.optimization_patients:
        local_slice = request.optimization_local_map[patient_id]
        patient_global_indices = request.optimization_idx[local_slice]
        patient_truth = np.asarray(request.truth_memmap[patient_global_indices], dtype=np.uint8)
        truth_cache[patient_id] = patient_truth
        gt_density_by_patient[patient_id] = (
            float(np.mean(patient_truth)) if patient_truth.size else 0.0
        )
        semantic_cache[patient_id] = _stack_prediction_rows(
            request.prediction_memmaps,
            model_indices=request.semantic_indices,
            global_indices=patient_global_indices,
            height=request.height,
            width=request.width,
        )
        spatial_cache[patient_id] = _stack_prediction_rows(
            request.prediction_memmaps,
            model_indices=request.spatial_indices,
            global_indices=patient_global_indices,
            height=request.height,
            width=request.width,
        )
    return semantic_cache, spatial_cache, truth_cache, gt_density_by_patient


def _weighted_ensemble_from_stacked_u16(
    stacked_values: npt.NDArray[np.uint16], weights: list[float]
) -> npt.NDArray[np.float32]:
    if stacked_values.shape[0] == 0:
        return cast(
            npt.NDArray[np.float32],
            np.zeros(stacked_values.shape[1:], dtype=np.float32),
        )
    effective_weights = _effective_weight_array(weights)
    if not np.any(effective_weights):
        return cast(
            npt.NDArray[np.float32],
            np.zeros(stacked_values.shape[1:], dtype=np.float32),
        )
    scaled = stacked_values.astype(np.float32) * (effective_weights[:, None, None, None] / 65535.0)
    return cast(npt.NDArray[np.float32], np.sum(scaled, axis=0, dtype=np.float32))


def _weighted_roi_predictions_from_u16(
    roi_predictions: npt.NDArray[np.uint16],
    weights: list[float],
) -> npt.NDArray[np.float32]:
    if roi_predictions.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    effective_weights = _effective_weight_array(weights)
    if not np.any(effective_weights):
        return np.zeros((roi_predictions.shape[1],), dtype=np.float32)
    scaled = roi_predictions.astype(np.float32) * (effective_weights[:, None] / 65535.0)
    return cast(npt.NDArray[np.float32], np.sum(scaled, axis=0, dtype=np.float32))


def _optimization_patient_spatial_stack(
    prepared: OptimizationPreparation,
    *,
    patient_id: str,
) -> npt.NDArray[np.uint16]:
    if prepared.optimization_spatial_cache is not None:
        return prepared.optimization_spatial_cache[patient_id]
    local_slice = prepared.optimization_local_map[patient_id]
    global_indices = prepared.optimization_idx[local_slice]
    return _stack_prediction_rows(
        prepared.prediction_memmaps,
        model_indices=prepared.spatial_indices,
        global_indices=global_indices,
        height=prepared.height,
        width=prepared.width,
    )


def _build_spatial_objective_patient_cache(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    fixed_roi_mask: npt.NDArray[np.uint8],
    patient_id: str,
) -> _SpatialObjectivePatientCache | None:
    patient_truth = _optimization_patient_truth(prepared, patient_id=patient_id)
    is_positive_patient = patient_id in prepared.optimization_positive_patients
    if config.spatial_patient_policy == "positive_only" and not is_positive_patient:
        return None

    local_slice = prepared.optimization_local_map[patient_id]
    patient_roi = np.asarray(fixed_roi_mask[local_slice], dtype=np.uint8)
    spatial_stack = _optimization_patient_spatial_stack(prepared, patient_id=patient_id)
    total_mass_by_model = cast(
        npt.NDArray[np.float64],
        np.sum(spatial_stack, axis=(1, 2, 3), dtype=np.float64) / 65535.0,
    )
    outside_mask = (1 - patient_roi).astype(np.uint8, copy=False)
    outside_mass_by_model = cast(
        npt.NDArray[np.float64],
        np.sum(spatial_stack * outside_mask[None, ...], axis=(1, 2, 3), dtype=np.float64) / 65535.0,
    )
    negative_mean_by_model = (
        total_mass_by_model / max(1, int(patient_truth.size))
        if not is_positive_patient
        else np.zeros_like(total_mass_by_model)
    )

    if is_positive_patient:
        roi_flat = patient_roi.reshape(-1).astype(bool)
        truth_roi = np.asarray(patient_truth.reshape(-1)[roi_flat], dtype=np.uint8)
        roi_predictions = np.asarray(
            spatial_stack.reshape(len(prepared.spatial_indices), -1)[:, roi_flat],
            dtype=np.uint16,
        )
    else:
        truth_roi = np.zeros((0,), dtype=np.uint8)
        roi_predictions = np.zeros((len(prepared.spatial_indices), 0), dtype=np.uint16)

    return _SpatialObjectivePatientCache(
        is_positive_patient=is_positive_patient,
        roi_truth=truth_roi,
        roi_predictions=roi_predictions,
        total_mass_by_model=total_mass_by_model,
        outside_mass_by_model=outside_mass_by_model,
        negative_mean_by_model=negative_mean_by_model,
    )


def _build_spatial_objective_cache(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    fixed_roi_mask: npt.NDArray[np.uint8],
) -> list[_SpatialObjectivePatientCache]:
    patient_caches: list[_SpatialObjectivePatientCache] = []
    for patient_id in prepared.optimization_patients:
        patient_cache = _build_spatial_objective_patient_cache(
            config,
            prepared,
            fixed_roi_mask,
            patient_id,
        )
        if patient_cache is not None:
            patient_caches.append(patient_cache)
    LOGGER.info(
        "Spatial objective precompute ready for %s evaluated optimization patients.",
        len(patient_caches),
    )
    return patient_caches


def _optimization_patient_semantic_prediction(
    prepared: OptimizationPreparation,
    *,
    patient_id: str,
    weights: list[float],
) -> npt.NDArray[np.float32]:
    if prepared.optimization_semantic_cache is not None:
        return _weighted_ensemble_from_stacked_u16(
            prepared.optimization_semantic_cache[patient_id],
            weights,
        )
    local_slice = prepared.optimization_local_map[patient_id]
    global_indices = prepared.optimization_idx[local_slice]
    patient_u16 = [
        prepared.prediction_memmaps[index][global_indices] for index in prepared.semantic_indices
    ]
    return _weighted_ensemble_from_u16_cache(patient_u16, weights)


def _optimization_patient_spatial_prediction(
    prepared: OptimizationPreparation,
    *,
    patient_id: str,
    weights: list[float],
) -> npt.NDArray[np.float32]:
    if prepared.optimization_spatial_cache is not None:
        return _weighted_ensemble_from_stacked_u16(
            prepared.optimization_spatial_cache[patient_id],
            weights,
        )
    local_slice = prepared.optimization_local_map[patient_id]
    global_indices = prepared.optimization_idx[local_slice]
    patient_u16 = [
        prepared.prediction_memmaps[index][global_indices] for index in prepared.spatial_indices
    ]
    return _weighted_ensemble_from_u16_cache(patient_u16, weights)


def _optimization_patient_truth(
    prepared: OptimizationPreparation,
    *,
    patient_id: str,
) -> npt.NDArray[np.uint8]:
    if prepared.optimization_truth_cache is not None:
        return prepared.optimization_truth_cache[patient_id]
    local_slice = prepared.optimization_local_map[patient_id]
    if prepared.optimization_truth is not None:
        return np.asarray(prepared.optimization_truth[local_slice], dtype=np.uint8)
    global_indices = prepared.optimization_idx[local_slice]
    return np.asarray(prepared.truth_memmap[global_indices], dtype=np.uint8)


def _resolve_stream_indices(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
) -> tuple[list[int], list[int]]:
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
    return semantic_indices, spatial_indices


def _prepare_optimization(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    predefined_split: HoldoutSplit | None,
) -> OptimizationPreparation:
    prediction_paths, _, pids_path, total_samples, height, width, truth_memmap = (
        cache_predictions_sequential(
            models,
            dataloader,
            device=device,
            cache_dir=config.pred_cache_dir,
        )
    )
    patient_ids = cast(list[str], json.loads(pids_path.read_text(encoding="utf-8")))
    patient_map = _build_patient_map(patient_ids)
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
    semantic_indices, spatial_indices = _resolve_stream_indices(config, models)
    (
        optimization_semantic_cache,
        optimization_spatial_cache,
        optimization_truth_cache,
        optimization_gt_density_by_patient,
    ) = _build_optimization_patient_caches(
        _OptimizationCacheBuildInput(
            prediction_memmaps=prediction_memmaps,
            optimization_idx=optimization_idx,
            optimization_local_map=optimization_local_map,
            optimization_patients=optimization_patients,
            semantic_indices=semantic_indices,
            spatial_indices=spatial_indices,
            truth_memmap=truth_memmap,
            height=height,
            width=width,
            max_cache_bytes=config.optimization_cache_max_bytes,
        )
    )
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
    return OptimizationPreparation(
        patient_map=patient_map,
        prediction_memmaps=prediction_memmaps,
        truth_memmap=truth_memmap,
        holdout_idx=holdout_idx,
        holdout_local_map=holdout_local_map,
        holdout_patients=holdout_patients,
        calibration_idx=calibration_idx,
        calibration_local_map=calibration_local_map,
        calibration_patients=calibration_patients,
        optimization_idx=optimization_idx,
        optimization_local_map=optimization_local_map,
        optimization_patients=optimization_patients,
        semantic_indices=semantic_indices,
        spatial_indices=spatial_indices,
        optimization_truth=None,
        optimization_semantic_cache=optimization_semantic_cache,
        optimization_spatial_cache=optimization_spatial_cache,
        optimization_truth_cache=optimization_truth_cache,
        optimization_gt_density_by_patient=optimization_gt_density_by_patient,
        optimization_positive_patients=set(split.optimization_patients) & positive_patients,
        height=height,
        width=width,
    )


def _semantic_objective(
    trial: optuna.Trial,
    *,
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
) -> float:
    weights = _normalize_weights(
        [trial.suggest_float(f"w_sem_{i}", 0.0, 1.0) for i in range(len(prepared.semantic_indices))]
    )
    roi_threshold = trial.suggest_float("roi_thresh", 0.15, 0.60)
    roi_area_fractions: list[float] = []
    roi_positive_recalls: list[float] = []
    positive_recall_sum = 0.0
    processed_positive_patients = 0
    total_positive_patients = len(prepared.optimization_positive_patients)
    empty_rois = 0
    total_patients = len(prepared.optimization_patients)
    for patient_id in prepared.optimization_patients:
        patient_ensemble = _optimization_patient_semantic_prediction(
            prepared,
            patient_id=patient_id,
            weights=weights,
        )
        roi_mask = (
            generate_roi_batch(
                torch.from_numpy(patient_ensemble),
                config.roi_context_scale,
                roi_threshold,
            )
            .numpy()
            .astype(np.uint8)
        )
        patient_truth = _optimization_patient_truth(prepared, patient_id=patient_id)
        roi_area_fractions.append(float(np.mean(roi_mask)))
        if np.sum(roi_mask) == 0:
            empty_rois += 1
        if np.sum(patient_truth) > 0:
            intersection = np.sum((roi_mask == 1) & (patient_truth == 1))
            total_positive = np.sum(patient_truth)
            recall = float(intersection / (total_positive + 1e-7))
            roi_positive_recalls.append(recall)
            positive_recall_sum += recall
            processed_positive_patients += 1
        if empty_rois / max(1, total_patients) > config.roi_empty_max:
            raise optuna.exceptions.TrialPruned(
                f"Trivial Empty: {empty_rois / max(1, total_patients):.2f}"
            )
        if total_positive_patients > 0:
            remaining_positive_patients = total_positive_patients - processed_positive_patients
            best_possible_positive_recall = (
                positive_recall_sum + remaining_positive_patients
            ) / total_positive_patients
            if best_possible_positive_recall < config.roi_min_pos_recall:
                raise optuna.exceptions.TrialPruned(
                    f"Misses Positives: {best_possible_positive_recall:.2f}"
                )
    median_area = float(np.median(roi_area_fractions)) if roi_area_fractions else 0.0
    empty_rate = float(empty_rois / max(1, len(prepared.optimization_patients)))
    mean_positive_recall = float(np.mean(roi_positive_recalls)) if roi_positive_recalls else 0.0
    if empty_rate > config.roi_empty_max:
        raise optuna.exceptions.TrialPruned(f"Trivial Empty: {empty_rate:.2f}")
    if mean_positive_recall < config.roi_min_pos_recall:
        raise optuna.exceptions.TrialPruned(f"Misses Positives: {mean_positive_recall:.2f}")
    gt_densities = (
        [
            prepared.optimization_gt_density_by_patient[patient_id]
            for patient_id in prepared.optimization_patients
        ]
        if prepared.optimization_gt_density_by_patient is not None
        else [
            float(np.mean(_optimization_patient_truth(prepared, patient_id=patient_id)))
            for patient_id in prepared.optimization_patients
        ]
    )
    mean_gt_density = float(np.mean(gt_densities)) if gt_densities else 0.0
    dynamic_max_median = max(config.roi_max_median, mean_gt_density + 0.15)
    if median_area > dynamic_max_median:
        raise optuna.exceptions.TrialPruned(
            f"Trivial Permissive: {median_area:.2f} > {dynamic_max_median:.2f}"
        )
    return mean_positive_recall


def _run_semantic_optimization(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
) -> tuple[list[float], float]:
    semantic_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    LOGGER.info(
        "Optimizing semantic stream over %s patients with %s trials.",
        len(prepared.optimization_patients),
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
            lambda trial: _semantic_objective(trial, config=config, prepared=prepared),
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
            for i in range(len(prepared.semantic_indices))
        ]
    )
    return best_semantic_weights, float(semantic_study.best_params["roi_thresh"])


def _build_fixed_roi_mask(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    *,
    best_semantic_weights: list[float],
    best_roi_threshold: float,
) -> np.memmap[Any, Any]:
    roi_path = config.pred_cache_dir / "fixed_roi_mask.dat"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_roi_mask = np.memmap(
        roi_path,
        dtype="uint8",
        mode="w+",
        shape=(len(prepared.optimization_idx), prepared.height, prepared.width),
    )
    if prepared.optimization_semantic_cache is not None:
        for patient_id in prepared.optimization_patients:
            local_slice = prepared.optimization_local_map[patient_id]
            patient_prediction = _optimization_patient_semantic_prediction(
                prepared,
                patient_id=patient_id,
                weights=best_semantic_weights,
            )
            patient_roi = generate_roi_batch(
                torch.from_numpy(patient_prediction),
                config.roi_context_scale,
                best_roi_threshold,
            )
            fixed_roi_mask[local_slice] = patient_roi.numpy().astype(np.uint8)
        fixed_roi_mask.flush()
        return fixed_roi_mask
    chunk_size = 2048
    roi_chunk_count = max(1, int(np.ceil(len(prepared.optimization_idx) / chunk_size)))
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
        for start in range(0, len(prepared.optimization_idx), chunk_size):
            stop = min(start + chunk_size, len(prepared.optimization_idx))
            chunk_global_indices = prepared.optimization_idx[start:stop]
            chunk_u16 = [
                prepared.prediction_memmaps[index][chunk_global_indices]
                for index in prepared.semantic_indices
            ]
            chunk_probs = _weighted_ensemble_from_u16_cache(chunk_u16, best_semantic_weights)
            chunk_roi = generate_roi_batch(
                torch.from_numpy(chunk_probs),
                config.roi_context_scale,
                best_roi_threshold,
            )
            fixed_roi_mask[start:stop] = chunk_roi.numpy().astype(np.uint8)
            roi_progress.update(1)
    fixed_roi_mask.flush()
    return fixed_roi_mask


def _spatial_objective(
    trial: optuna.Trial,
    *,
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    patient_caches: list[_SpatialObjectivePatientCache],
) -> float:
    weights = _normalize_weights(
        [trial.suggest_float(f"w_spa_{i}", 0.0, 1.0) for i in range(len(prepared.spatial_indices))]
    )
    effective_weights = _effective_weight_array(weights).astype(np.float64)
    positive_auprc_total = 0.0
    spill_total = 0.0
    evaluated_patients = 0
    evaluated_positive_patients = 0
    negative_fp_total = 0.0
    evaluated_negative_patients = 0
    for patient_cache in patient_caches:
        if patient_cache.is_positive_patient:
            patient_prediction = _weighted_roi_predictions_from_u16(
                patient_cache.roi_predictions,
                weights,
            )
            positive_auprc_total += _compute_auprc_from_roi_vectors(
                patient_cache.roi_truth,
                patient_prediction,
            )
            evaluated_positive_patients += 1
        else:
            negative_fp_total += float(
                np.dot(effective_weights, patient_cache.negative_mean_by_model)
            )
            evaluated_negative_patients += 1
        mass_total = float(np.dot(effective_weights, patient_cache.total_mass_by_model) + 1e-7)
        mass_outside = float(np.dot(effective_weights, patient_cache.outside_mass_by_model))
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
        negative_fp_total / evaluated_negative_patients if evaluated_negative_patients > 0 else 0.0
    )
    return float(
        macro_positive_auprc
        - (config.spill_penalty_lambda * macro_spill)
        - (config.spill_penalty_lambda * macro_negative_fp)
    )


def _run_spatial_optimization(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    patient_caches: list[_SpatialObjectivePatientCache],
) -> list[float]:
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
            lambda trial: _spatial_objective(
                trial,
                config=config,
                prepared=prepared,
                patient_caches=patient_caches,
            ),
            n_trials=config.num_trials_spatial,
            callbacks=_optuna_callbacks(spatial_progress),
        )
    _require_completed_trials(spatial_study, stream_name="Spatial")
    LOGGER.info(
        "Spatial optimization complete: best_objective=%.4f.",
        float(spatial_study.best_value),
    )
    return _normalize_weights(
        [
            spatial_study.best_params.get(f"w_spa_{i}", 0.0)
            for i in range(len(prepared.spatial_indices))
        ]
    )


def _compute_ensemble_iterative(
    prediction_memmaps: list[np.memmap[Any, Any]],
    *,
    model_indices: list[int],
    weights: list[float],
    global_indices: npt.NDArray[np.int64],
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    weight_skip_threshold = 1e-5
    accumulator = cast(
        npt.NDArray[np.float32],
        np.zeros((len(global_indices), height, width), dtype=np.float32),
    )
    for model_index, weight in zip(model_indices, weights, strict=False):
        if weight <= weight_skip_threshold:
            continue
        chunk = prediction_memmaps[model_index][global_indices]
        accumulator += chunk.astype(np.float32) * (weight / 65535.0)
    return accumulator


def _evaluate_holdout(
    config: EnsembleOptimizerConfig,
    prepared: OptimizationPreparation,
    *,
    best_semantic_weights: list[float],
    best_spatial_weights: list[float],
    best_roi_threshold: float,
    decision_threshold: float,
) -> HoldoutEvaluation:
    positive_auprc_total = 0.0
    spill_total = 0.0
    evaluated_patients = 0
    evaluated_positive_patients = 0
    negative_fp_total = 0.0
    evaluated_negative_patients = 0
    positive_patient_count = 0
    negative_patient_count = 0
    LOGGER.info("Evaluating holdout set across %s patients.", len(prepared.holdout_patients))
    with tqdm(
        prepared.holdout_patients,
        total=len(prepared.holdout_patients),
        desc="Eval holdout",
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        file=_progress_file(),
        disable=_progress_disabled(),
    ) as holdout_progress:
        for patient_id in holdout_progress:
            gc.collect()
            local_slice = prepared.holdout_local_map[patient_id]
            global_indices = prepared.holdout_idx[local_slice]
            if len(global_indices) == 0:
                continue
            semantic_prediction = _compute_ensemble_iterative(
                prepared.prediction_memmaps,
                model_indices=prepared.semantic_indices,
                weights=best_semantic_weights,
                global_indices=global_indices,
                height=prepared.height,
                width=prepared.width,
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
            spatial_prediction = _compute_ensemble_iterative(
                prepared.prediction_memmaps,
                model_indices=prepared.spatial_indices,
                weights=best_spatial_weights,
                global_indices=global_indices,
                height=prepared.height,
                width=prepared.width,
            )
            patient_truth = prepared.truth_memmap[global_indices].astype(np.uint8)
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
    metrics: dict[str, float | int | str] = {
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
    return HoldoutEvaluation(
        macro_positive_auprc=macro_positive_auprc,
        macro_spill=macro_spill,
        holdout_objective=holdout_objective,
        metrics=metrics,
    )


def _evaluate_holdout_streaming(
    context: _StreamingPredictionContext,
    *,
    patient_ids: list[str],
    decision_threshold: float,
) -> HoldoutEvaluation:
    positive_auprc_total = 0.0
    spill_total = 0.0
    evaluated_patients = 0
    evaluated_positive_patients = 0
    negative_fp_total = 0.0
    evaluated_negative_patients = 0
    positive_patient_count = 0
    negative_patient_count = 0
    LOGGER.info("Evaluating holdout set across %s patients.", len(patient_ids))
    with tqdm(
        patient_ids,
        total=len(patient_ids),
        desc="Eval holdout",
        leave=False,
        mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        file=_progress_file(),
        disable=_progress_disabled(),
    ) as holdout_progress:
        for patient_id in holdout_progress:
            gc.collect()
            dataloader = context.dataloader_factory({patient_id})
            semantic_prediction, patient_truth = _stream_weighted_prediction_for_patient(
                models=context.models,
                model_indices=context.semantic_indices,
                weights=context.semantic_weights,
                dataloader=dataloader,
                device=context.device,
            )
            if patient_truth.size == 0:
                continue
            roi_mask = (
                generate_roi_batch(
                    torch.from_numpy(semantic_prediction),
                    context.config.roi_context_scale,
                    context.roi_threshold,
                )
                .numpy()
                .astype(np.uint8)
            )
            spatial_prediction, _patient_truth = _stream_weighted_prediction_for_patient(
                models=context.models,
                model_indices=context.spatial_indices,
                weights=context.spatial_weights,
                dataloader=dataloader,
                device=context.device,
                truth=patient_truth,
            )
            is_positive_patient = bool(np.sum(patient_truth) > 0)
            if is_positive_patient:
                positive_patient_count += 1
            else:
                negative_patient_count += 1
            if context.config.spatial_patient_policy == "positive_only" and not is_positive_patient:
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
        - (context.config.spill_penalty_lambda * macro_spill)
        - (context.config.spill_penalty_lambda * macro_negative_fp)
    )
    metrics: dict[str, float | int | str] = {
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
        "Spatial_patient_policy": context.config.spatial_patient_policy,
        "Spill_lambda": float(context.config.spill_penalty_lambda),
    }
    LOGGER.info(
        "Holdout evaluation complete: objective=%.4f macro_auprc=%.4f spill=%.4f.",
        holdout_objective,
        macro_positive_auprc,
        macro_spill,
    )
    return HoldoutEvaluation(
        macro_positive_auprc=macro_positive_auprc,
        macro_spill=macro_spill,
        holdout_objective=holdout_objective,
        metrics=metrics,
    )


def _build_stream_preparation_from_cache(
    request: _StreamPreparationRequest,
) -> OptimizationPreparation:
    patient_ids = cast(list[str], json.loads(request.pids_path.read_text(encoding="utf-8")))
    patient_map = _build_patient_map(patient_ids)
    optimization_idx, optimization_local_map, optimization_patients = build_indices_and_local_map(
        set(patient_map),
        patient_map,
    )
    prediction_memmaps = _open_prediction_memmaps(
        request.prediction_paths,
        total_samples=request.total_samples,
        height=request.height,
        width=request.width,
    )
    model_cache_indices = list(range(len(prediction_memmaps)))
    semantic_indices = model_cache_indices if request.stream == "semantic" else []
    spatial_indices = model_cache_indices if request.stream == "spatial" else []
    (
        optimization_semantic_cache,
        optimization_spatial_cache,
        optimization_truth_cache,
        optimization_gt_density_by_patient,
    ) = _build_optimization_patient_caches(
        _OptimizationCacheBuildInput(
            prediction_memmaps=prediction_memmaps,
            optimization_idx=optimization_idx,
            optimization_local_map=optimization_local_map,
            optimization_patients=optimization_patients,
            semantic_indices=semantic_indices,
            spatial_indices=spatial_indices,
            truth_memmap=request.truth_memmap,
            height=request.height,
            width=request.width,
            max_cache_bytes=request.config.optimization_cache_max_bytes,
        )
    )
    positive_patients = _compute_positive_patients(patient_map, request.truth_memmap)
    return OptimizationPreparation(
        patient_map=patient_map,
        prediction_memmaps=prediction_memmaps,
        truth_memmap=request.truth_memmap,
        holdout_idx=np.array([], dtype=np.int64),
        holdout_local_map={},
        holdout_patients=[],
        calibration_idx=np.array([], dtype=np.int64),
        calibration_local_map={},
        calibration_patients=[],
        optimization_idx=optimization_idx,
        optimization_local_map=optimization_local_map,
        optimization_patients=optimization_patients,
        semantic_indices=semantic_indices,
        spatial_indices=spatial_indices,
        optimization_truth=None,
        optimization_semantic_cache=(
            optimization_semantic_cache if request.stream == "semantic" else None
        ),
        optimization_spatial_cache=(
            optimization_spatial_cache if request.stream == "spatial" else None
        ),
        optimization_truth_cache=optimization_truth_cache,
        optimization_gt_density_by_patient=optimization_gt_density_by_patient,
        optimization_positive_patients=positive_patients,
        height=request.height,
        width=request.width,
    )


def _run_two_stream_optimization_staged(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    *,
    device: torch.device,
    predefined_split: HoldoutSplit,
    dataloader_factory: ValidationDataloaderFactory,
) -> OptimizationResult:
    semantic_indices, spatial_indices = _resolve_stream_indices(config, models)
    if config.pred_cache_dir.exists():
        shutil.rmtree(config.pred_cache_dir)
    config.pred_cache_dir.mkdir(parents=True, exist_ok=True)

    optimization_patients = set(predefined_split.optimization_patients)
    LOGGER.info(
        "Using staged prediction cache: optimization=%s calibration=%s holdout=%s patients.",
        len(optimization_patients),
        len(predefined_split.calibration_patients),
        len(predefined_split.holdout_patients),
    )
    semantic_payload = _cache_prediction_group(
        config=config,
        models=models,
        model_indices=semantic_indices,
        dataloader=dataloader_factory(optimization_patients),
        device=device,
        cache_name="semantic_optimization",
    )
    (
        semantic_prediction_paths,
        _semantic_truth_path,
        semantic_pids_path,
        total_samples,
        height,
        width,
        semantic_truth_memmap,
    ) = semantic_payload
    _log_staged_cache_estimate(
        config,
        optimization_samples=total_samples,
        height=height,
        width=width,
        semantic_model_count=len(semantic_indices),
        spatial_model_count=len(spatial_indices),
    )
    semantic_prepared = _build_stream_preparation_from_cache(
        _StreamPreparationRequest(
            config=config,
            prediction_paths=semantic_prediction_paths,
            pids_path=semantic_pids_path,
            total_samples=total_samples,
            height=height,
            width=width,
            truth_memmap=semantic_truth_memmap,
            stream="semantic",
        )
    )

    previous_optuna_verbosity = _set_optuna_warning_verbosity()
    try:
        best_semantic_weights, best_roi_threshold = _run_semantic_optimization(
            config,
            semantic_prepared,
        )
        fixed_roi_mask = _build_fixed_roi_mask(
            config,
            semantic_prepared,
            best_semantic_weights=best_semantic_weights,
            best_roi_threshold=best_roi_threshold,
        )
        _close_prediction_memmaps(semantic_prepared.prediction_memmaps)
        _remove_paths(semantic_prediction_paths)

        spatial_payload = _cache_prediction_group(
            config=config,
            models=models,
            model_indices=spatial_indices,
            dataloader=dataloader_factory(optimization_patients),
            device=device,
            cache_name="spatial_optimization",
        )
        (
            spatial_prediction_paths,
            _spatial_truth_path,
            spatial_pids_path,
            spatial_total_samples,
            spatial_height,
            spatial_width,
            spatial_truth_memmap,
        ) = spatial_payload
        if (
            spatial_total_samples != total_samples
            or spatial_height != height
            or spatial_width != width
            or spatial_pids_path.read_text(encoding="utf-8")
            != semantic_pids_path.read_text(encoding="utf-8")
        ):
            raise RuntimeError(
                "Staged semantic and spatial optimization caches do not describe the same "
                "sample order."
            )
        spatial_prepared = _build_stream_preparation_from_cache(
            _StreamPreparationRequest(
                config=config,
                prediction_paths=spatial_prediction_paths,
                pids_path=spatial_pids_path,
                total_samples=spatial_total_samples,
                height=spatial_height,
                width=spatial_width,
                truth_memmap=spatial_truth_memmap,
                stream="spatial",
            )
        )
        spatial_objective_cache = _build_spatial_objective_cache(
            config,
            spatial_prepared,
            fixed_roi_mask,
        )
        _close_prediction_memmaps(spatial_prepared.prediction_memmaps)
        _remove_paths(spatial_prediction_paths)
        best_spatial_weights = _run_spatial_optimization(
            config,
            spatial_prepared,
            spatial_objective_cache,
        )
    finally:
        _restore_optuna_verbosity(previous_optuna_verbosity)

    fixed_roi_path = config.pred_cache_dir / "fixed_roi_mask.dat"
    _close_memmap(fixed_roi_mask)
    fixed_roi_path.unlink(missing_ok=True)
    _close_memmap(semantic_truth_memmap)
    _close_memmap(spatial_truth_memmap)

    streaming_context = _StreamingPredictionContext(
        config=config,
        dataloader_factory=dataloader_factory,
        models=models,
        semantic_indices=semantic_indices,
        semantic_weights=best_semantic_weights,
        spatial_indices=spatial_indices,
        spatial_weights=best_spatial_weights,
        device=device,
        roi_threshold=best_roi_threshold,
    )
    calibration_patients = sorted(predefined_split.calibration_patients)
    LOGGER.info("Calibrating decision threshold on %s patients.", len(calibration_patients))
    calibration_result = _calibrate_decision_threshold_streaming(
        streaming_context,
        patient_ids=calibration_patients,
    )
    LOGGER.info(
        "Calibration complete: threshold=%.4f macro_rule6=%.4f min_area=%s min_patches=%s.",
        calibration_result.decision_threshold,
        float(cast(float, calibration_result.metrics.get("Calibration_macro_rule6", 0.0))),
        calibration_result.postprocessing_config.min_component_area_px,
        calibration_result.postprocessing_config.min_patient_positive_patches,
    )
    holdout_evaluation = _evaluate_holdout_streaming(
        streaming_context,
        patient_ids=sorted(predefined_split.holdout_patients),
        decision_threshold=calibration_result.decision_threshold,
    )
    shutil.rmtree(config.pred_cache_dir, ignore_errors=True)
    config.pred_cache_dir.mkdir(parents=True, exist_ok=True)
    return OptimizationResult(
        semantic_indices=semantic_indices,
        spatial_indices=spatial_indices,
        semantic_weights=best_semantic_weights,
        spatial_weights=best_spatial_weights,
        roi_threshold=best_roi_threshold,
        decision_threshold=calibration_result.decision_threshold,
        postprocessing_config=calibration_result.postprocessing_config,
        calibration_metrics=calibration_result.metrics,
        validation_calibration_summary=calibration_result.summary,
        holdout_metrics=holdout_evaluation.metrics,
    )


def run_two_stream_optimization(
    config: EnsembleOptimizerConfig,
    models: list[nn.Module],
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    predefined_split: HoldoutSplit | None = None,
    dataloader_factory: ValidationDataloaderFactory | None = None,
) -> OptimizationResult:
    if dataloader_factory is not None and predefined_split is not None:
        return _run_two_stream_optimization_staged(
            config,
            models,
            device=device,
            predefined_split=predefined_split,
            dataloader_factory=dataloader_factory,
        )

    LOGGER.info("Caching ensemble predictions for %s models.", len(models))
    prepared = _prepare_optimization(
        config,
        models,
        dataloader,
        device=device,
        predefined_split=predefined_split,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    previous_optuna_verbosity = _set_optuna_warning_verbosity()
    try:
        best_semantic_weights, best_roi_threshold = _run_semantic_optimization(config, prepared)
        fixed_roi_mask = _build_fixed_roi_mask(
            config,
            prepared,
            best_semantic_weights=best_semantic_weights,
            best_roi_threshold=best_roi_threshold,
        )
        spatial_objective_cache = _build_spatial_objective_cache(
            config,
            prepared,
            fixed_roi_mask,
        )
        best_spatial_weights = _run_spatial_optimization(
            config,
            prepared,
            spatial_objective_cache,
        )
    finally:
        _restore_optuna_verbosity(previous_optuna_verbosity)
    LOGGER.info(
        "Calibrating decision threshold on %s patients.",
        len(prepared.calibration_patients),
    )
    calibration_result = _calibrate_decision_threshold(
        ThresholdCalibrationConfig(
            patient_ids=prepared.calibration_patients,
            local_map=prepared.calibration_local_map,
            global_indices=prepared.calibration_idx,
            truth_memmap=prepared.truth_memmap,
            prediction_memmaps=prepared.prediction_memmaps,
            semantic_indices=prepared.semantic_indices,
            semantic_weights=best_semantic_weights,
            spatial_indices=prepared.spatial_indices,
            spatial_weights=best_spatial_weights,
            roi_context_scale=config.roi_context_scale,
            roi_threshold=best_roi_threshold,
            optimizer_config=config,
        )
    )
    LOGGER.info(
        "Calibration complete: threshold=%.4f macro_rule6=%.4f min_area=%s min_patches=%s.",
        calibration_result.decision_threshold,
        float(cast(float, calibration_result.metrics.get("Calibration_macro_rule6", 0.0))),
        calibration_result.postprocessing_config.min_component_area_px,
        calibration_result.postprocessing_config.min_patient_positive_patches,
    )
    holdout_evaluation = _evaluate_holdout(
        config,
        prepared,
        best_semantic_weights=best_semantic_weights,
        best_spatial_weights=best_spatial_weights,
        best_roi_threshold=best_roi_threshold,
        decision_threshold=calibration_result.decision_threshold,
    )
    return OptimizationResult(
        semantic_indices=prepared.semantic_indices,
        spatial_indices=prepared.spatial_indices,
        semantic_weights=best_semantic_weights,
        spatial_weights=best_spatial_weights,
        roi_threshold=best_roi_threshold,
        decision_threshold=calibration_result.decision_threshold,
        postprocessing_config=calibration_result.postprocessing_config,
        calibration_metrics=calibration_result.metrics,
        validation_calibration_summary=calibration_result.summary,
        holdout_metrics=holdout_evaluation.metrics,
    )
