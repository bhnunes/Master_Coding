from __future__ import annotations

import sys
from collections.abc import Iterable, Sized
from typing import Any

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


def train_epoch(
    model: torch.nn.Module,
    optimizer: Any,
    dataloader: Iterable[Any],
    device: torch.device,
    current_epoch: int,
    loss_fn: Any,
    health: TrainingHealthTracker,
    architecture: str,
    accumulation_steps: int,
    amp_precision: str,
    gpu_normalizer: torch.nn.Module,
    gpu_downscale: torch.nn.Module,
    use_artifact_aware_loss: bool = False,
) -> tuple[float, dict[str, str]]:
    """Run one training epoch and return average loss plus AMP mode description."""

    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()

    amp_dtype, scaler, precision_log = setup_precision(architecture, amp_precision=amp_precision)
    tracker_loss = RunningWeightedMetric()
    optimizer.zero_grad(set_to_none=True)
    current_accumulation_steps = 0
    total_batches = len(dataloader) if isinstance(dataloader, Sized) else None

    pbar = tqdm(
        enumerate(dataloader),
        total=total_batches,
        desc=f"Train E{current_epoch + 1}",
        leave=False,
        mininterval=0.5,
        dynamic_ncols=True,
        position=0,
        file=_progress_file(),
    )

    for _, batch_data in pbar:
        if batch_data is None:
            health.train_skip("dataloader_none_batch")
            continue
        try:
            if len(batch_data) == 3:
                images, masks, artifact_covariates = batch_data
            else:
                images, masks = batch_data
                artifact_covariates = None
        except Exception:
            health.train_skip("unpack_failed")
            continue
        if images is None or masks is None:
            health.train_skip("images_or_masks_none")
            continue
        if images.shape[0] == 0:
            health.train_skip("batch_size_zero")
            continue

        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)
        if artifact_covariates is not None:
            artifact_covariates = artifact_covariates.to(
                device, non_blocking=True, dtype=torch.float32
            )
        images = gpu_normalizer(images)
        images = gpu_downscale(images)

        if masks.ndim == 4 and masks.size(1) == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 4 and masks.size(-1) == 1:
            masks = masks[..., 0]
        if masks.ndim != 3:
            health.train_skip("mask_bad_shape")
            continue

        with autocast_ctx(images, amp_dtype):
            outputs_raw = model(images)
            if isinstance(outputs_raw, (tuple, list)):
                if len(outputs_raw) == 0:
                    health.train_skip("model_empty_tuple")
                    continue
                outputs = outputs_raw[0]
            else:
                outputs = outputs_raw

            if not isinstance(outputs, torch.Tensor):
                health.train_skip("model_output_not_tensor")
                continue
            if masks.ndim != 3 or outputs.ndim != 4:
                health.train_skip("bad_tensor_rank")
                continue
            if outputs.shape[-2:] != masks.shape[-2:]:
                health.train_skip("spatial_mismatch")
                continue
            if outputs.shape[1] != 2:
                health.train_skip("channel_mismatch")
                continue

            if use_artifact_aware_loss and artifact_covariates is not None:
                loss = loss_fn(outputs, masks, artifact_covariates=artifact_covariates)
            else:
                loss = loss_fn(outputs, masks)

        if not torch.isfinite(loss):
            health.train_naninf_loss()
            health.train_skip("naninf_loss")
            continue

        loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        real_loss = loss.item() * accumulation_steps
        tracker_loss.update(real_loss * images.shape[0], images.shape[0])
        current_accumulation_steps += 1

        if current_accumulation_steps % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            current_accumulation_steps = 0
            pbar.set_postfix(loss=f"{tracker_loss.get_average():.4f}")

    return tracker_loss.get_average(), precision_log


def validate_epoch(
    model: torch.nn.Module,
    optimizer: Any,
    dataloader: Iterable[Any],
    device: torch.device,
    loss_fn: Any,
    health: TrainingHealthTracker,
    architecture: str,
    amp_precision: str,
    gpu_normalizer: torch.nn.Module,
) -> tuple[float, dict[str, float] | None]:
    """Run one validation epoch and return average loss plus metric bundle."""

    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()

    amp_dtype, _, _ = setup_precision(architecture, amp_precision=amp_precision)
    tracker = AdvancedMetricTracker(device=device, metric_bins=2048)
    running_loss = 0.0
    num_samples_processed = 0

    pbar = tqdm(
        dataloader,
        desc="Validate",
        leave=False,
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        file=_progress_file(),
    )
    with torch.inference_mode():
        for batch_data in pbar:
            if batch_data is None:
                health.val_skip("dataloader_none_batch")
                continue
            try:
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                else:
                    images, masks = batch_data
            except Exception:
                health.val_skip("unpack_failed")
                continue
            if images is None or masks is None:
                health.val_skip("images_or_masks_none")
                continue
            try:
                batch_size = images.size(0)
            except Exception:
                health.val_skip("images_no_batch_dim")
                continue
            if batch_size == 0:
                health.val_skip("batch_size_zero")
                continue

            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            images = gpu_normalizer(images)
            masks = masks.to(device, non_blocking=True, dtype=torch.long)

            with autocast_ctx(images, amp_dtype):
                outputs_raw = model(images)
                if isinstance(outputs_raw, (tuple, list)):
                    if len(outputs_raw) == 0:
                        health.val_skip("model_empty_tuple")
                        continue
                    outputs = outputs_raw[0]
                else:
                    outputs = outputs_raw
                if not isinstance(outputs, torch.Tensor):
                    health.val_skip("model_output_not_tensor")
                    continue
                if masks.ndim == 4 and masks.size(1) == 1:
                    masks = masks[:, 0, :, :]
                elif masks.ndim == 4 and masks.size(-1) == 1:
                    masks = masks[..., 0]
                if masks.ndim != 3:
                    health.val_skip("mask_bad_shape")
                    continue
                if outputs.ndim != 4 or outputs.shape[1] != 2:
                    health.val_skip("channel_mismatch")
                    continue
                if outputs.shape[-2:] != masks.shape[-2:]:
                    health.val_skip("spatial_mismatch")
                    continue

                loss = loss_fn(outputs, masks)

            if not torch.isfinite(loss):
                health.val_naninf_loss()
                health.val_skip("naninf_loss")
                continue

            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            num_samples_processed += batch_size
            tracker.update(outputs, masks)

            if num_samples_processed > 0:
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    avg_loss=f"{running_loss / num_samples_processed:.4f}",
                )

    try:
        if num_samples_processed == 0:
            health.val_skip("no_samples_processed")
            return 0.0, None
        epoch_loss = running_loss / num_samples_processed
        return epoch_loss, tracker.compute_and_reset(health=health)
    finally:
        del tracker
