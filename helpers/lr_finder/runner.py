from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.amp.grad_scaler import GradScaler

from helpers.lr_finder.analysis import compute_curve_stats
from helpers.lr_finder.config import LRFinderConfig, ModelPlan
from helpers.lr_finder.data import build_train_loader, prepare_training_data
from helpers.lr_finder.reporting import RunRecord
from helpers.lr_finder.search_space import BCEDiceParams, sample_bcedice_params
from helpers.training.device import require_cuda_device
from helpers.training.gpu import GPUDownscale, GPUNormalizer
from helpers.training.losses import BCEDiceHybridLossConfig, BCEDiceHybridLossPaper
from helpers.training.models import create_model
from helpers.training.runtime import autocast_ctx, seed_everything, setup_precision

logger = logging.getLogger(__name__)

_COLAB_INLINE_MPL_BACKEND = "module://matplotlib_inline.backend_inline"


class _NonFiniteLossError(RuntimeError):
    """Signal an LR-range test that diverged to a non-finite loss."""


@dataclass(frozen=True)
class ScreeningOutputs:
    records: list[RunRecord]
    lhs_samples_path: Path
    summary_all_path: Path
    architecture_summary_paths: dict[str, Path]
    architecture_trial_stats: dict[str, dict[str, int]]
    completed_trials: int
    failed_trials: int
    source_split_name: str | None = None
    training_dataset_provenance: dict[str, Any] | None = None
    validation_dataset_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class LRFinderRunConfig:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_loader: Any
    device: torch.device
    end_lr: float
    num_iter: int
    architecture: str
    amp_precision: str
    gpu_normalizer: GPUNormalizer
    gpu_downscale: GPUDownscale


@dataclass(frozen=True)
class LossConfigRunContext:
    device: torch.device
    data_bundle: Any
    model_plan: ModelPlan
    params: BCEDiceParams
    config_index: int
    gpu_normalizer: GPUNormalizer
    gpu_downscale: GPUDownscale
    initial_state_dict: dict[str, torch.Tensor]


class _SnapshotProgressReporter:
    """Emit compact periodic progress snapshots instead of a live progress bar."""

    def __init__(self, total_trials: int) -> None:
        self._total_trials = total_trials
        self._completed = 0
        self._started_at = time.monotonic()
        self._snapshot_every = max(1, min(25, max(5, total_trials // 20)))
        self._next_snapshot_at = min(total_trials, self._snapshot_every)

    def advance(
        self,
        increment: int,
        *,
        architecture: str,
        encoder: str,
        sample_index: int,
        total_samples: int,
        completed_trials: int,
        failed_trials: int,
    ) -> None:
        self._completed += increment
        if self._completed < self._next_snapshot_at and self._completed < self._total_trials:
            return

        elapsed_seconds = max(1.0, time.monotonic() - self._started_at)
        rate = self._completed / elapsed_seconds
        percent_complete = (
            (100.0 * self._completed / self._total_trials) if self._total_trials else 100.0
        )
        print(
            "Progress: "
            f"{self._completed}/{self._total_trials} ({percent_complete:.1f}%) | "
            f"arch={architecture} | encoder={encoder} | "
            f"sample={sample_index}/{total_samples} | "
            f"ok={completed_trials} | fail={failed_trials} | "
            f"rate={rate:.2f} trials/s"
        )
        while self._next_snapshot_at <= self._completed:
            self._next_snapshot_at += self._snapshot_every

    def finish(self) -> None:
        return None


def configure_execution_mode(execution_mode: str) -> None:
    if execution_mode == "FAST_DEV":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def clear_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
        except Exception:
            pass


def _prepare_lr_finder_batch(
    batch_data: Any,
    *,
    device: torch.device,
    gpu_normalizer: GPUNormalizer,
    gpu_downscale: GPUDownscale,
) -> tuple[torch.Tensor, torch.Tensor]:
    required_batch_items = 2
    squeezed_mask_rank = 3
    batched_mask_rank = 4
    single_channel_count = 1
    if batch_data is None:
        raise RuntimeError("LR finder received an empty batch.")
    if len(batch_data) < required_batch_items:
        raise RuntimeError("LR finder batch is missing images or masks.")

    images, masks = batch_data[:2]
    if images is None or masks is None:
        raise RuntimeError("LR finder batch contains missing images or masks.")
    if images.shape[0] == 0:
        raise RuntimeError("LR finder batch has zero samples.")

    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
    masks = masks.to(device, non_blocking=True, dtype=torch.long)
    images = gpu_normalizer(images)
    images = gpu_downscale(images)

    if masks.ndim == batched_mask_rank and masks.size(1) == single_channel_count:
        masks = masks[:, 0, :, :]
    elif masks.ndim == batched_mask_rank and masks.size(-1) == single_channel_count:
        masks = masks[..., 0]
    if masks.ndim != squeezed_mask_rank:
        raise RuntimeError(f"LR finder masks must be rank-3 after squeeze, got {masks.ndim}.")
    return images, masks


def _validate_lr_finder_outputs(outputs_raw: Any, masks: torch.Tensor) -> torch.Tensor:
    output_rank = 4
    binary_class_count = 2
    if isinstance(outputs_raw, (tuple, list)):
        if len(outputs_raw) == 0:
            raise RuntimeError("LR finder model returned an empty tuple/list output.")
        outputs = outputs_raw[0]
    else:
        outputs = outputs_raw

    if not isinstance(outputs, torch.Tensor):
        raise RuntimeError("LR finder model output is not a tensor.")
    if outputs.ndim != output_rank:
        raise RuntimeError(f"LR finder model output must be rank-4, got {outputs.ndim}.")
    if outputs.shape[1] != binary_class_count:
        raise RuntimeError(f"LR finder model output must have 2 channels, got {outputs.shape[1]}.")
    if outputs.shape[-2:] != masks.shape[-2:]:
        raise RuntimeError(
            "LR finder model output spatial dimensions do not match mask dimensions."
        )
    return outputs


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _build_lr_finder_train_batch(
    *,
    amp_dtype: torch.dtype,
    gpu_normalizer: GPUNormalizer,
    gpu_downscale: GPUDownscale,
    scaler: GradScaler | None,
) -> Any:
    def _train_batch_patched(
        self: Any,
        train_iter: Any,
        accumulation_steps: int,
        non_blocking_transfer: bool = True,
        **kwargs: Any,
    ) -> float:
        del accumulation_steps, non_blocking_transfer, kwargs
        self.model.train()
        try:
            batch_data = next(train_iter)
        except StopIteration:
            return float("nan")
        images, masks = _prepare_lr_finder_batch(
            batch_data,
            device=self.device,
            gpu_normalizer=gpu_normalizer,
            gpu_downscale=gpu_downscale,
        )
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(images, amp_dtype):
            outputs = _validate_lr_finder_outputs(self.model(images), masks)
            loss = self.criterion(outputs, masks)
        if not torch.isfinite(loss):
            raise _NonFiniteLossError("LR finder produced a non-finite loss.")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return float(loss.item())

    return _train_batch_patched


def _extract_lr_finder_history(
    history: dict[str, Any] | None,
) -> dict[str, npt.NDArray[np.float64]]:
    if history is None or "lr" not in history or "loss" not in history:
        return {
            "lr": np.array([], dtype=np.float64),
            "loss": np.array([], dtype=np.float64),
        }
    return {
        "lr": np.asarray(history["lr"], dtype=np.float64),
        "loss": np.asarray(history["loss"], dtype=np.float64),
    }


def _dispose_lr_finder(lr_finder: Any) -> None:
    if hasattr(lr_finder, "model"):
        lr_finder.model = None
    if hasattr(lr_finder, "optimizer"):
        lr_finder.optimizer = None
    if hasattr(lr_finder, "criterion"):
        lr_finder.criterion = None


def _ensure_headless_matplotlib_backend() -> None:
    if os.environ.get("MPLBACKEND") == _COLAB_INLINE_MPL_BACKEND:
        os.environ["MPLBACKEND"] = "Agg"


def run_lr_finder_once(config: LRFinderRunConfig) -> dict[str, npt.NDArray[np.float64]]:
    _ensure_headless_matplotlib_backend()
    from torch_lr_finder import LRFinder

    amp_dtype, scaler, _ = setup_precision(config.architecture, amp_precision=config.amp_precision)
    lr_finder = LRFinder(config.model, config.optimizer, config.criterion, device=config.device)

    lr_finder._train_batch = _build_lr_finder_train_batch(
        amp_dtype=amp_dtype,
        gpu_normalizer=config.gpu_normalizer,
        gpu_downscale=config.gpu_downscale,
        scaler=scaler,
    ).__get__(lr_finder, LRFinder)
    history: dict[str, Any] | None = None
    try:
        lr_finder.range_test(
            config.train_loader,
            end_lr=config.end_lr,
            num_iter=config.num_iter,
            step_mode="exp",
        )
        history = lr_finder.history
    except _NonFiniteLossError:
        history = getattr(lr_finder, "history", None)
        completed_steps = 0 if history is None else len(history.get("loss", []))
        logger.info(
            (
                "LR finder stopped early after non-finite loss: architecture=%s "
                "completed_steps=%s requested_steps=%s"
            ),
            config.architecture,
            completed_steps,
            config.num_iter,
        )
    except Exception:
        logger.exception("LR finder range test failed for architecture %s", config.architecture)
        raise
    finally:
        _dispose_lr_finder(lr_finder)
        del lr_finder

    return _extract_lr_finder_history(history)


def _load_matplotlib_pyplot() -> Any:
    _ensure_headless_matplotlib_backend()

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return pyplot


def plot_stability_curves(
    all_lrs: list[npt.NDArray[np.float64]],
    all_losses: list[npt.NDArray[np.float64]],
    *,
    title: str,
    out_png: Path,
    skip_start: int = 10,
    skip_end: int = 5,
) -> None:
    plt = _load_matplotlib_pyplot()
    plt.figure(figsize=(10, 6))
    for index, (lrs, losses) in enumerate(zip(all_lrs, all_losses, strict=True)):
        lower_bound = skip_start
        upper_bound = max(lower_bound + 1, len(lrs) - skip_end)
        if lower_bound >= upper_bound:
            continue
        plt.plot(
            np.log10(lrs[lower_bound:upper_bound]),
            losses[lower_bound:upper_bound],
            alpha=0.6,
            linewidth=1.5,
            label=f"Run {index + 1}",
        )
    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _capture_pretrained_model_state(
    model_plan: ModelPlan, device: torch.device
) -> dict[str, torch.Tensor]:
    logger.info(
        "Loading pretrained encoder once for architecture=%s encoder=%s",
        model_plan.architecture,
        model_plan.encoder,
    )
    model = None
    try:
        model = create_model(model_plan.architecture, model_plan.encoder).to(device)
        return _clone_state_dict_to_cpu(model)
    finally:
        if model is not None:
            del model
        gc.collect()
        clear_gpu()


def _run_single_loss_config(
    config: LRFinderConfig,
    context: LossConfigRunContext,
) -> tuple[RunRecord | None, int, int]:
    repeated_lrs: list[npt.NDArray[np.float64]] = []
    repeated_losses: list[npt.NDArray[np.float64]] = []
    repeated_stats = []
    completed_trials = 0
    failed_trials = 0

    for repeat_index in range(config.num_repeats):
        current_seed = config.seed + (context.config_index * 100) + repeat_index
        seed_everything(current_seed)
        train_loader = build_train_loader(
            context.data_bundle.dataset,
            context.data_bundle.sample_weights,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=current_seed,
        )
        model = None
        optimizer = None
        criterion = None
        try:
            model = create_model(
                context.model_plan.architecture,
                context.model_plan.encoder,
                validation=True,
            ).to(context.device)
            model.load_state_dict(context.initial_state_dict)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.optimizer_start_lr,
                weight_decay=config.optimizer_weight_decay,
            )
            criterion = BCEDiceHybridLossPaper(
                BCEDiceHybridLossConfig(
                    alpha=context.params.alpha,
                    beta=context.params.beta,
                    gamma=context.params.gamma,
                )
            )
            history = run_lr_finder_once(
                LRFinderRunConfig(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=train_loader,
                    device=context.device,
                    end_lr=config.end_lr,
                    num_iter=config.num_iter,
                    architecture=context.model_plan.architecture,
                    amp_precision=config.amp_precision,
                    gpu_normalizer=context.gpu_normalizer,
                    gpu_downscale=context.gpu_downscale,
                )
            )
            stats = compute_curve_stats(history["lr"], history["loss"], skip_start=10, skip_end=5)
            if not np.isfinite(stats.min_loss):
                failed_trials += 1
                logger.info(
                    (
                        "LR finder repeat produced no finite minimum loss: "
                        "architecture=%s alpha=%.3f beta=%.3f gamma=%.3f repeat=%s"
                    ),
                    context.model_plan.architecture,
                    context.params.alpha,
                    context.params.beta,
                    context.params.gamma,
                    repeat_index + 1,
                )
                continue
            repeated_lrs.append(history["lr"])
            repeated_losses.append(history["loss"])
            repeated_stats.append(stats)
            completed_trials += 1
        except Exception:
            logger.exception(
                (
                    "LR finder repeat failed: architecture=%s alpha=%.3f "
                    "beta=%.3f gamma=%.3f repeat=%s"
                ),
                context.model_plan.architecture,
                context.params.alpha,
                context.params.beta,
                context.params.gamma,
                repeat_index + 1,
            )
            failed_trials += 1
        finally:
            del train_loader
            if criterion is not None:
                del criterion
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
            gc.collect()
            clear_gpu()

    if not repeated_stats:
        return None, completed_trials, failed_trials

    output_dir = config.output_dir / context.model_plan.architecture
    tag_base = (
        f"Loss_alpha_{context.params.alpha:.3f}_beta_{context.params.beta:.3f}"
        f"_gamma_{context.params.gamma:.3f}"
    )
    plot_path = output_dir / f"{context.config_index:03d}_{tag_base}.png"
    plot_stability_curves(
        repeated_lrs,
        repeated_losses,
        title=(
            f"{context.model_plan.architecture} | Stability (N={config.num_repeats})\n"
            f"α={context.params.alpha:.3f}, β={context.params.beta:.3f}, "
            f"γ={context.params.gamma:.3f}"
        ),
        out_png=plot_path,
    )
    record = RunRecord(
        architecture=context.model_plan.architecture,
        encoder=context.model_plan.encoder,
        alpha=context.params.alpha,
        beta=context.params.beta,
        gamma=context.params.gamma,
        median_min_loss=float(np.median([stats.min_loss for stats in repeated_stats])),
        plot_path=plot_path,
        csv_path=output_dir / f"SUMMARY_{context.model_plan.architecture}_STABILITY.csv",
    )
    return record, completed_trials, failed_trials


def run_lr_finder_screening(config: LRFinderConfig) -> ScreeningOutputs:
    configure_execution_mode(config.execution_mode)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    lhs_samples = sample_bcedice_params(config.num_lhs_samples, config.search_space, config.seed)
    lhs_samples_path = config.output_dir / "LHS_SAMPLES.json"
    lhs_samples_path.write_text(
        json.dumps([asdict(sample) for sample in lhs_samples], indent=2), encoding="utf-8"
    )

    device = require_cuda_device()
    data_bundle = prepare_training_data(config, normalizer_device=device)
    gpu_normalizer = GPUNormalizer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device,
    )
    gpu_downscale = GPUDownscale(p=0.07).to(device)

    records: list[RunRecord] = []
    completed_trials = 0
    failed_trials = 0
    architecture_summary_paths: dict[str, Path] = {}
    architecture_trial_stats: dict[str, dict[str, int]] = {}
    total_trials = len(config.model_plans) * len(lhs_samples) * config.num_repeats
    progress = _SnapshotProgressReporter(total_trials)
    try:
        for model_plan in config.model_plans:
            initial_state_dict = _capture_pretrained_model_state(model_plan, device)
            architecture_completed_before = completed_trials
            architecture_failed_before = failed_trials
            for config_index, params in enumerate(lhs_samples, start=1):
                record, ok_count, fail_count = _run_single_loss_config(
                    config,
                    LossConfigRunContext(
                        device=device,
                        data_bundle=data_bundle,
                        model_plan=model_plan,
                        params=params,
                        config_index=config_index,
                        gpu_normalizer=gpu_normalizer,
                        gpu_downscale=gpu_downscale,
                        initial_state_dict=initial_state_dict,
                    ),
                )
                completed_trials += ok_count
                failed_trials += fail_count
                progress.advance(
                    config.num_repeats,
                    architecture=model_plan.architecture,
                    encoder=model_plan.encoder,
                    sample_index=config_index,
                    total_samples=len(lhs_samples),
                    completed_trials=completed_trials,
                    failed_trials=failed_trials,
                )
                if record is not None:
                    records.append(record)

            architecture_records = [
                item for item in records if item.architecture == model_plan.architecture
            ]
            if architecture_records:
                architecture_summary_path = (
                    config.output_dir
                    / model_plan.architecture
                    / f"SUMMARY_{model_plan.architecture}_STABILITY.csv"
                )
                pd.DataFrame([asdict(item) for item in architecture_records]).to_csv(
                    architecture_summary_path,
                    index=False,
                )
                architecture_summary_paths[model_plan.architecture] = architecture_summary_path
            architecture_trial_stats[model_plan.architecture] = {
                "valid_records": len(architecture_records),
                "completed_trials": completed_trials - architecture_completed_before,
                "failed_trials": failed_trials - architecture_failed_before,
            }
    finally:
        progress.finish()
        close_dataset = getattr(data_bundle.dataset, "close", None)
        if callable(close_dataset):
            close_dataset()

    summary_all_path = config.output_dir / "SUMMARY_ALL.csv"
    pd.DataFrame([asdict(record) for record in records]).to_csv(summary_all_path, index=False)
    return ScreeningOutputs(
        records=records,
        lhs_samples_path=lhs_samples_path,
        summary_all_path=summary_all_path,
        architecture_summary_paths=architecture_summary_paths,
        architecture_trial_stats=architecture_trial_stats,
        completed_trials=completed_trials,
        failed_trials=failed_trials,
        source_split_name=getattr(data_bundle, "source_split_name", None),
        training_dataset_provenance=getattr(data_bundle, "training_provenance", None),
        validation_dataset_provenance=getattr(data_bundle, "validation_provenance", None),
    )
