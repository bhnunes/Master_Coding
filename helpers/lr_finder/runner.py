from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from tqdm import tqdm

from helpers.lr_finder.analysis import compute_curve_stats
from helpers.lr_finder.config import LRFinderConfig, ModelPlan
from helpers.lr_finder.data import build_train_loader, prepare_training_data
from helpers.lr_finder.reporting import RunRecord
from helpers.lr_finder.search_space import BCEDiceParams, sample_bcedice_params
from helpers.training.gpu import GPUDownscale, GPUNormalizer
from helpers.training.losses import BCEDiceHybridLossPaper
from helpers.training.models import create_model
from helpers.training.runtime import autocast_ctx, seed_everything, setup_precision

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


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
    if batch_data is None:
        raise RuntimeError("LR finder received an empty batch.")
    if len(batch_data) < 2:
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

    if masks.ndim == 4 and masks.size(1) == 1:
        masks = masks[:, 0, :, :]
    elif masks.ndim == 4 and masks.size(-1) == 1:
        masks = masks[..., 0]
    if masks.ndim != 3:
        raise RuntimeError(f"LR finder masks must be rank-3 after squeeze, got {masks.ndim}.")
    return images, masks


def _validate_lr_finder_outputs(outputs_raw: Any, masks: torch.Tensor) -> torch.Tensor:
    if isinstance(outputs_raw, (tuple, list)):
        if len(outputs_raw) == 0:
            raise RuntimeError("LR finder model returned an empty tuple/list output.")
        outputs = outputs_raw[0]
    else:
        outputs = outputs_raw

    if not isinstance(outputs, torch.Tensor):
        raise RuntimeError("LR finder model output is not a tensor.")
    if outputs.ndim != 4:
        raise RuntimeError(f"LR finder model output must be rank-4, got {outputs.ndim}.")
    if outputs.shape[1] != 2:
        raise RuntimeError(f"LR finder model output must have 2 channels, got {outputs.shape[1]}.")
    if outputs.shape[-2:] != masks.shape[-2:]:
        raise RuntimeError(
            "LR finder model output spatial dimensions do not match mask dimensions."
        )
    return outputs


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def run_lr_finder_once(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: Any,
    device: torch.device,
    end_lr: float,
    num_iter: int,
    architecture: str,
    amp_precision: str,
    gpu_normalizer: GPUNormalizer,
    gpu_downscale: GPUDownscale,
) -> dict[str, npt.NDArray[np.float64]]:
    from torch_lr_finder import LRFinder

    amp_dtype, scaler, _ = setup_precision(architecture, amp_precision=amp_precision)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

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

    lr_finder._train_batch = _train_batch_patched.__get__(lr_finder, LRFinder)
    history: dict[str, Any] | None = None
    try:
        lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
        history = lr_finder.history
    except _NonFiniteLossError:
        history = getattr(lr_finder, "history", None)
        completed_steps = 0 if history is None else len(history.get("loss", []))
        logger.info(
            (
                "LR finder stopped early after non-finite loss: architecture=%s "
                "completed_steps=%s requested_steps=%s"
            ),
            architecture,
            completed_steps,
            num_iter,
        )
    except Exception:
        logger.exception("LR finder range test failed for architecture %s", architecture)
        raise
    finally:
        if hasattr(lr_finder, "model"):
            lr_finder.model = None
        if hasattr(lr_finder, "optimizer"):
            lr_finder.optimizer = None
        if hasattr(lr_finder, "criterion"):
            lr_finder.criterion = None
        del lr_finder

    if history is None or "lr" not in history or "loss" not in history:
        return {
            "lr": np.array([], dtype=np.float64),
            "loss": np.array([], dtype=np.float64),
        }
    return {
        "lr": np.asarray(history["lr"], dtype=np.float64),
        "loss": np.asarray(history["loss"], dtype=np.float64),
    }


def plot_stability_curves(
    all_lrs: list[npt.NDArray[np.float64]],
    all_losses: list[npt.NDArray[np.float64]],
    *,
    title: str,
    out_png: Path,
    skip_start: int = 10,
    skip_end: int = 5,
) -> None:
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
    *,
    device: torch.device,
    data_bundle: Any,
    model_plan: ModelPlan,
    params: BCEDiceParams,
    config_index: int,
    gpu_normalizer: GPUNormalizer,
    gpu_downscale: GPUDownscale,
    initial_state_dict: dict[str, torch.Tensor],
) -> tuple[RunRecord | None, int, int]:
    repeated_lrs: list[npt.NDArray[np.float64]] = []
    repeated_losses: list[npt.NDArray[np.float64]] = []
    repeated_stats = []
    completed_trials = 0
    failed_trials = 0

    for repeat_index in range(config.num_repeats):
        current_seed = config.seed + (config_index * 100) + repeat_index
        seed_everything(current_seed)
        train_loader = build_train_loader(
            data_bundle.dataset,
            data_bundle.sample_weights,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=current_seed,
        )
        model = None
        optimizer = None
        criterion = None
        try:
            model = create_model(model_plan.architecture, model_plan.encoder, validation=True).to(
                device
            )
            model.load_state_dict(initial_state_dict)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.optimizer_start_lr,
                weight_decay=config.optimizer_weight_decay,
            )
            criterion = BCEDiceHybridLossPaper(
                alpha=params.alpha,
                beta=params.beta,
                gamma=params.gamma,
            )
            history = run_lr_finder_once(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                device=device,
                end_lr=config.end_lr,
                num_iter=config.num_iter,
                architecture=model_plan.architecture,
                amp_precision=config.amp_precision,
                gpu_normalizer=gpu_normalizer,
                gpu_downscale=gpu_downscale,
            )
            stats = compute_curve_stats(history["lr"], history["loss"], skip_start=10, skip_end=5)
            if not np.isfinite(stats.min_loss):
                failed_trials += 1
                logger.info(
                    (
                        "LR finder repeat produced no finite minimum loss: "
                        "architecture=%s alpha=%.3f beta=%.3f gamma=%.3f repeat=%s"
                    ),
                    model_plan.architecture,
                    params.alpha,
                    params.beta,
                    params.gamma,
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
                model_plan.architecture,
                params.alpha,
                params.beta,
                params.gamma,
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

    output_dir = config.output_dir / model_plan.architecture
    tag_base = f"Loss_alpha_{params.alpha:.3f}_beta_{params.beta:.3f}_gamma_{params.gamma:.3f}"
    plot_path = output_dir / f"{config_index:03d}_{tag_base}.png"
    plot_stability_curves(
        repeated_lrs,
        repeated_losses,
        title=(
            f"{model_plan.architecture} | Stability (N={config.num_repeats})\n"
            f"α={params.alpha:.3f}, β={params.beta:.3f}, γ={params.gamma:.3f}"
        ),
        out_png=plot_path,
    )
    record = RunRecord(
        architecture=model_plan.architecture,
        encoder=model_plan.encoder,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        median_min_loss=float(np.median([stats.min_loss for stats in repeated_stats])),
        plot_path=plot_path,
        csv_path=output_dir / f"SUMMARY_{model_plan.architecture}_STABILITY.csv",
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

    data_bundle = prepare_training_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    progress_bar = tqdm(total=total_trials, desc="Stage 8 LR Finder", unit="trial")
    try:
        for model_plan in config.model_plans:
            initial_state_dict = _capture_pretrained_model_state(model_plan, device)
            architecture_completed_before = completed_trials
            architecture_failed_before = failed_trials
            for config_index, params in enumerate(lhs_samples, start=1):
                progress_bar.set_postfix_str(
                    (
                        f"arch={model_plan.architecture} encoder={model_plan.encoder} "
                        f"config={config_index}/{len(lhs_samples)}"
                    ),
                    refresh=False,
                )
                record, ok_count, fail_count = _run_single_loss_config(
                    config,
                    device=device,
                    data_bundle=data_bundle,
                    model_plan=model_plan,
                    params=params,
                    config_index=config_index,
                    gpu_normalizer=gpu_normalizer,
                    gpu_downscale=gpu_downscale,
                    initial_state_dict=initial_state_dict,
                )
                completed_trials += ok_count
                failed_trials += fail_count
                progress_bar.update(config.num_repeats)
                progress_bar.set_postfix(
                    {
                        "arch": model_plan.architecture,
                        "config": f"{config_index}/{len(lhs_samples)}",
                        "ok": completed_trials,
                        "fail": failed_trials,
                    },
                    refresh=False,
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
        progress_bar.close()
        data_bundle.dataset.close()

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
    )
