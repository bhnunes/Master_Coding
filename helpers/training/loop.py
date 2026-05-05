from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import Any, cast

import torch
from tqdm import tqdm

from helpers.training.metrics import (
    AdvancedMetricTracker,
    RunningWeightedMetric,
    TrainingHealthTracker,
)
from helpers.training.runtime import autocast_ctx, setup_precision


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under LoggerWriter."""

    return sys.__stderr__


_TRAIN_PROGRESS_MIN_INTERVAL_SECONDS = 0.5
_VALIDATION_PROGRESS_MIN_INTERVAL_SECONDS = 5.0
_BATCH_WITH_ARTIFACT_COVARIATES = 3
_MASK_BATCH_NDIM = 4
_MASK_IMAGE_NDIM = 3
_MODEL_OUTPUT_NDIM = 4
_SINGLE_CHANNEL_COUNT = 1
_BINARY_CLASS_COUNT = 2


@dataclass(frozen=True)
class TrainEpochConfig:
    device: torch.device
    current_epoch: int
    architecture: str
    accumulation_steps: int
    amp_precision: str
    use_artifact_aware_loss: bool = False


@dataclass(frozen=True)
class ValidationEpochConfig:
    device: torch.device
    architecture: str
    amp_precision: str
    profile_timing: bool = False


@dataclass(frozen=True)
class TrainEpochRuntime:
    loss_fn: Any
    health: TrainingHealthTracker
    gpu_normalizer: torch.nn.Module
    gpu_downscale: torch.nn.Module


@dataclass(frozen=True)
class ValidationEpochRuntime:
    loss_fn: Any
    health: TrainingHealthTracker
    gpu_normalizer: torch.nn.Module


@dataclass
class _ValidationTimingStats:
    started_at: float
    dataloader_wait_seconds: float = 0.0
    prepare_seconds: float = 0.0
    forward_loss_seconds: float = 0.0
    metric_update_seconds: float = 0.0
    finalize_seconds: float = 0.0
    yielded_batches: int = 0
    processed_batches: int = 0
    processed_samples: int = 0


def _unpack_batch(
    batch_data: Any,
    *,
    health: TrainingHealthTracker,
    phase: str,
) -> tuple[Any, Any, Any | None] | None:
    try:
        if len(batch_data) == _BATCH_WITH_ARTIFACT_COVARIATES:
            images, masks, metadata = batch_data
            return images, masks, metadata
        images, masks = batch_data
        return images, masks, None
    except Exception:
        _record_skip(health, phase, "unpack_failed")
        return None


def _record_skip(health: TrainingHealthTracker, phase: str, reason: str) -> None:
    if phase == "train":
        health.train_skip(reason)
    else:
        health.val_skip(reason)


def _extract_model_output(
    outputs_raw: Any,
    *,
    health: TrainingHealthTracker,
    phase: str,
) -> torch.Tensor | None:
    if isinstance(outputs_raw, (tuple, list)):
        if len(outputs_raw) == 0:
            _record_skip(health, phase, "model_empty_tuple")
            return None
        outputs = outputs_raw[0]
    else:
        outputs = outputs_raw

    if not isinstance(outputs, torch.Tensor):
        _record_skip(health, phase, "model_output_not_tensor")
        return None
    return outputs


def _normalize_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == _MASK_BATCH_NDIM and masks.size(1) == _SINGLE_CHANNEL_COUNT:
        return masks[:, 0, :, :]
    if masks.ndim == _MASK_BATCH_NDIM and masks.size(-1) == _SINGLE_CHANNEL_COUNT:
        return masks[..., 0]
    return masks


def _validate_segmentation_tensors(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    *,
    health: TrainingHealthTracker,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    normalized_masks = _normalize_masks(masks)
    if normalized_masks.ndim != _MASK_IMAGE_NDIM:
        _record_skip(health, phase, "mask_bad_shape")
        return None
    if outputs.ndim != _MODEL_OUTPUT_NDIM:
        _record_skip(health, phase, "bad_tensor_rank")
        return None
    if outputs.shape[-2:] != normalized_masks.shape[-2:]:
        _record_skip(health, phase, "spatial_mismatch")
        return None
    if outputs.shape[1] != _BINARY_CLASS_COUNT:
        _record_skip(health, phase, "channel_mismatch")
        return None
    return outputs, normalized_masks


def _prepare_train_batch(
    batch_data: Any,
    *,
    config: TrainEpochConfig,
    runtime: TrainEpochRuntime,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    unpacked = _unpack_batch(batch_data, health=runtime.health, phase="train")
    if unpacked is None:
        return None
    images, masks, artifact_covariates = unpacked
    if images is None or masks is None:
        runtime.health.train_skip("images_or_masks_none")
        return None
    if images.shape[0] == 0:
        runtime.health.train_skip("batch_size_zero")
        return None

    images = images.to(config.device, non_blocking=True, memory_format=torch.channels_last)
    masks = masks.to(config.device, non_blocking=True, dtype=torch.long)
    if artifact_covariates is not None:
        artifact_covariates = artifact_covariates.to(
            config.device,
            non_blocking=True,
            dtype=torch.float32,
        )
    images = runtime.gpu_normalizer(images)
    images = runtime.gpu_downscale(images)
    return images, masks, artifact_covariates


def _compute_train_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    artifact_covariates: torch.Tensor | None,
    *,
    amp_dtype: torch.dtype,
    config: TrainEpochConfig,
    runtime: TrainEpochRuntime,
) -> torch.Tensor | None:
    with autocast_ctx(images, amp_dtype):
        outputs = _extract_model_output(model(images), health=runtime.health, phase="train")
        if outputs is None:
            return None
        validated = _validate_segmentation_tensors(
            outputs,
            masks,
            health=runtime.health,
            phase="train",
        )
        if validated is None:
            return None
        outputs, masks = validated
        if config.use_artifact_aware_loss and artifact_covariates is not None:
            return cast(
                torch.Tensor,
                runtime.loss_fn(outputs, masks, artifact_covariates=artifact_covariates),
            )
        return cast(torch.Tensor, runtime.loss_fn(outputs, masks))


def _prepare_validation_batch(
    batch_data: Any,
    *,
    config: ValidationEpochConfig,
    runtime: ValidationEpochRuntime,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    unpacked = _unpack_batch(batch_data, health=runtime.health, phase="val")
    if unpacked is None:
        return None
    images, masks, _ = unpacked
    if images is None or masks is None:
        runtime.health.val_skip("images_or_masks_none")
        return None
    try:
        batch_size = images.size(0)
    except Exception:
        runtime.health.val_skip("images_no_batch_dim")
        return None
    if batch_size == 0:
        runtime.health.val_skip("batch_size_zero")
        return None

    images = images.to(config.device, non_blocking=True, memory_format=torch.channels_last)
    images = runtime.gpu_normalizer(images)
    masks = masks.to(config.device, non_blocking=True, dtype=torch.long)
    return images, masks, batch_size


def _synchronize_validation_timing(config: ValidationEpochConfig) -> None:
    if config.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(config.device)


def _time_prepare_validation_batch(
    batch_data: Any,
    *,
    config: ValidationEpochConfig,
    runtime: ValidationEpochRuntime,
    timing: _ValidationTimingStats | None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if timing is None:
        return _prepare_validation_batch(batch_data, config=config, runtime=runtime)

    _synchronize_validation_timing(config)
    started_at = time.perf_counter()
    prepared = _prepare_validation_batch(batch_data, config=config, runtime=runtime)
    _synchronize_validation_timing(config)
    timing.prepare_seconds += time.perf_counter() - started_at
    return prepared


def _compute_validation_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    amp_dtype: torch.dtype,
    config: ValidationEpochConfig,
    runtime: ValidationEpochRuntime,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    with autocast_ctx(images, amp_dtype):
        outputs = _extract_model_output(model(images), health=runtime.health, phase="val")
        if outputs is None:
            return None
        validated = _validate_segmentation_tensors(
            outputs,
            masks,
            health=runtime.health,
            phase="val",
        )
        if validated is None:
            return None
        outputs, masks = validated
        loss = cast(torch.Tensor, runtime.loss_fn(outputs, masks))
    return outputs, masks, loss


def _time_compute_validation_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    amp_dtype: torch.dtype,
    config: ValidationEpochConfig,
    runtime: ValidationEpochRuntime,
    timing: _ValidationTimingStats | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if timing is None:
        return _compute_validation_outputs(
            model,
            images,
            masks,
            amp_dtype=amp_dtype,
            config=config,
            runtime=runtime,
        )

    _synchronize_validation_timing(config)
    started_at = time.perf_counter()
    outputs = _compute_validation_outputs(
        model,
        images,
        masks,
        amp_dtype=amp_dtype,
        config=config,
        runtime=runtime,
    )
    _synchronize_validation_timing(config)
    timing.forward_loss_seconds += time.perf_counter() - started_at
    return outputs


def _time_validation_metric_update(
    *,
    outputs: torch.Tensor,
    masks: torch.Tensor,
    loss: torch.Tensor,
    tracker: AdvancedMetricTracker,
    config: ValidationEpochConfig,
    timing: _ValidationTimingStats | None,
) -> float:
    if timing is None:
        batch_loss = loss.item()
        tracker.update(outputs, masks)
        return float(batch_loss)

    _synchronize_validation_timing(config)
    started_at = time.perf_counter()
    batch_loss = loss.item()
    tracker.update(outputs, masks)
    _synchronize_validation_timing(config)
    timing.metric_update_seconds += time.perf_counter() - started_at
    return float(batch_loss)


def _process_validation_batch(
    batch_data: Any,
    *,
    model: torch.nn.Module,
    amp_dtype: torch.dtype,
    config: ValidationEpochConfig,
    runtime: ValidationEpochRuntime,
    tracker: AdvancedMetricTracker,
    timing: _ValidationTimingStats | None,
) -> tuple[float, int] | None:
    if batch_data is None:
        runtime.health.val_skip("dataloader_none_batch")
        return None
    prepared = _time_prepare_validation_batch(
        batch_data,
        config=config,
        runtime=runtime,
        timing=timing,
    )
    if prepared is None:
        return None
    images, masks, batch_size = prepared
    batch_outputs = _time_compute_validation_outputs(
        model,
        images,
        masks,
        amp_dtype=amp_dtype,
        config=config,
        runtime=runtime,
        timing=timing,
    )
    if batch_outputs is None:
        return None
    outputs, masks, loss = batch_outputs

    if not torch.isfinite(loss):
        runtime.health.val_naninf_loss()
        runtime.health.val_skip("naninf_loss")
        return None

    batch_loss = _time_validation_metric_update(
        outputs=outputs,
        masks=masks,
        loss=loss,
        tracker=tracker,
        config=config,
        timing=timing,
    )
    if timing is not None:
        timing.processed_batches += 1
        timing.processed_samples += batch_size
    return batch_loss, batch_size


def _backward_loss(loss: torch.Tensor, scaler: Any | None) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
        return
    torch.autograd.backward(loss)


def _maybe_step_optimizer(
    *,
    optimizer: Any,
    scaler: Any | None,
    current_accumulation_steps: int,
    accumulation_steps: int,
) -> int:
    if current_accumulation_steps % accumulation_steps != 0:
        return current_accumulation_steps
    if scaler is not None:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return 0


def _maybe_update_validation_postfix(
    *,
    pbar: Any,
    batch_loss: float,
    running_loss: float,
    num_samples_processed: int,
    last_postfix_update: float,
) -> float:
    now = time.monotonic()
    if (
        num_samples_processed > 0
        and now - last_postfix_update >= _VALIDATION_PROGRESS_MIN_INTERVAL_SECONDS
    ):
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg_loss=f"{running_loss / num_samples_processed:.4f}",
            refresh=False,
        )
        return now
    return last_postfix_update


def _finalize_validation_epoch(
    *,
    health: TrainingHealthTracker,
    num_samples_processed: int,
    running_loss: float,
    tracker: AdvancedMetricTracker,
) -> tuple[float, dict[str, float] | None]:
    if num_samples_processed == 0:
        health.val_skip("no_samples_processed")
        health.mark_val_invalid_metrics()
        return 0.0, None
    epoch_loss = running_loss / num_samples_processed
    return epoch_loss, tracker.compute_and_reset(health=health)


def _finalize_validation_epoch_with_timing(
    *,
    health: TrainingHealthTracker,
    num_samples_processed: int,
    running_loss: float,
    tracker: AdvancedMetricTracker,
    config: ValidationEpochConfig,
    timing: _ValidationTimingStats | None,
) -> tuple[float, dict[str, float] | None]:
    if timing is None:
        return _finalize_validation_epoch(
            health=health,
            num_samples_processed=num_samples_processed,
            running_loss=running_loss,
            tracker=tracker,
        )

    _synchronize_validation_timing(config)
    started_at = time.perf_counter()
    result = _finalize_validation_epoch(
        health=health,
        num_samples_processed=num_samples_processed,
        running_loss=running_loss,
        tracker=tracker,
    )
    _synchronize_validation_timing(config)
    timing.finalize_seconds += time.perf_counter() - started_at
    _print_validation_timing(timing)
    return result


def _format_timing_component(name: str, seconds: float, total_seconds: float) -> str:
    pct = (seconds / total_seconds * 100.0) if total_seconds > 0.0 else 0.0
    return f"{name}={seconds:.2f}s ({pct:.1f}%)"


def _print_validation_timing(timing: _ValidationTimingStats) -> None:
    total_seconds = max(time.perf_counter() - timing.started_at, 1e-9)
    measured_seconds = (
        timing.dataloader_wait_seconds
        + timing.prepare_seconds
        + timing.forward_loss_seconds
        + timing.metric_update_seconds
        + timing.finalize_seconds
    )
    other_seconds = max(total_seconds - measured_seconds, 0.0)
    samples_per_second = timing.processed_samples / total_seconds
    components = [
        _format_timing_component("wait", timing.dataloader_wait_seconds, total_seconds),
        _format_timing_component("prepare", timing.prepare_seconds, total_seconds),
        _format_timing_component("forward_loss", timing.forward_loss_seconds, total_seconds),
        _format_timing_component("metrics", timing.metric_update_seconds, total_seconds),
        _format_timing_component("finalize", timing.finalize_seconds, total_seconds),
        _format_timing_component("other", other_seconds, total_seconds),
    ]
    print(
        "[Validation timing] "
        f"total={total_seconds:.2f}s "
        f"samples={timing.processed_samples} "
        f"batches={timing.processed_batches}/{timing.yielded_batches} "
        f"samples_per_second={samples_per_second:.2f} | " + " | ".join(components)
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: Any,
    dataloader: Iterable[Any],
    runtime: TrainEpochRuntime,
    config: TrainEpochConfig,
) -> tuple[float, dict[str, str]]:
    """Run one training epoch and return average loss plus AMP mode description."""

    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    if hasattr(runtime.loss_fn, "set_epoch"):
        runtime.loss_fn.set_epoch(config.current_epoch)
    if hasattr(runtime.loss_fn, "set_ohem_enabled"):
        runtime.loss_fn.set_ohem_enabled(True)

    amp_dtype, scaler, precision_log = setup_precision(
        config.architecture,
        amp_precision=config.amp_precision,
    )
    tracker_loss = RunningWeightedMetric()
    optimizer.zero_grad(set_to_none=True)
    current_accumulation_steps = 0
    total_batches = len(dataloader) if isinstance(dataloader, Sized) else None

    pbar = tqdm(
        enumerate(dataloader),
        total=total_batches,
        desc=f"Train E{config.current_epoch + 1}",
        leave=False,
        mininterval=_TRAIN_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        position=0,
        file=_progress_file(),
    )

    for _, batch_data in pbar:
        if batch_data is None:
            runtime.health.train_skip("dataloader_none_batch")
            continue
        prepared = _prepare_train_batch(
            batch_data,
            config=config,
            runtime=runtime,
        )
        if prepared is None:
            continue
        images, masks, artifact_covariates = prepared
        loss = _compute_train_loss(
            model,
            images,
            masks,
            artifact_covariates,
            amp_dtype=amp_dtype,
            config=config,
            runtime=runtime,
        )
        if loss is None:
            continue

        if not torch.isfinite(loss):
            runtime.health.train_naninf_loss()
            runtime.health.train_skip("naninf_loss")
            continue

        loss = loss / config.accumulation_steps
        _backward_loss(loss, scaler)

        real_loss = loss.item() * config.accumulation_steps
        tracker_loss.update(real_loss * images.shape[0], images.shape[0])
        current_accumulation_steps += 1

        current_accumulation_steps = _maybe_step_optimizer(
            optimizer=optimizer,
            scaler=scaler,
            current_accumulation_steps=current_accumulation_steps,
            accumulation_steps=config.accumulation_steps,
        )
        if current_accumulation_steps == 0:
            pbar.set_postfix(loss=f"{tracker_loss.get_average():.4f}")

    return tracker_loss.get_average(), precision_log


def validate_epoch(
    model: torch.nn.Module,
    optimizer: Any,
    dataloader: Iterable[Any],
    runtime: ValidationEpochRuntime,
    config: ValidationEpochConfig,
) -> tuple[float, dict[str, float] | None]:
    """Run one validation epoch and return average loss plus metric bundle."""

    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    if hasattr(runtime.loss_fn, "set_ohem_enabled"):
        runtime.loss_fn.set_ohem_enabled(False)
    if hasattr(runtime.loss_fn, "set_epoch"):
        runtime.loss_fn.set_epoch(None)

    amp_dtype, _, _ = setup_precision(
        config.architecture,
        amp_precision=config.amp_precision,
    )
    tracker = AdvancedMetricTracker(device=config.device, metric_bins=2048)
    running_loss = 0.0
    num_samples_processed = 0
    timing = _ValidationTimingStats(time.perf_counter()) if config.profile_timing else None

    pbar = tqdm(
        dataloader,
        desc="Validate",
        leave=False,
        mininterval=_VALIDATION_PROGRESS_MIN_INTERVAL_SECONDS,
        dynamic_ncols=True,
        position=0,
        file=_progress_file(),
    )
    last_postfix_update = 0.0
    last_batch_finished_at = time.perf_counter()
    with torch.inference_mode():
        for batch_data in pbar:
            if timing is not None:
                timing.yielded_batches += 1
                batch_ready_at = time.perf_counter()
                timing.dataloader_wait_seconds += batch_ready_at - last_batch_finished_at
            try:
                processed = _process_validation_batch(
                    batch_data,
                    model=model,
                    amp_dtype=amp_dtype,
                    config=config,
                    runtime=runtime,
                    tracker=tracker,
                    timing=timing,
                )
                if processed is None:
                    continue
                batch_loss, batch_size = processed
                running_loss += batch_loss * batch_size
                num_samples_processed += batch_size

                last_postfix_update = _maybe_update_validation_postfix(
                    pbar=pbar,
                    batch_loss=batch_loss,
                    running_loss=running_loss,
                    num_samples_processed=num_samples_processed,
                    last_postfix_update=last_postfix_update,
                )
            finally:
                if timing is not None:
                    _synchronize_validation_timing(config)
                    last_batch_finished_at = time.perf_counter()

    try:
        return _finalize_validation_epoch_with_timing(
            health=runtime.health,
            num_samples_processed=num_samples_processed,
            running_loss=running_loss,
            tracker=tracker,
            config=config,
            timing=timing,
        )
    finally:
        del tracker
