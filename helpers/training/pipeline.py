from __future__ import annotations

import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from helpers.training.checkpointing import (
    EarlyStoppingCheckpoint,
    OHEMCheckpointSettings,
    TrainingMetadataRequest,
)
from helpers.training.loop import (
    TrainEpochConfig,
    TrainEpochRuntime,
    ValidationEpochConfig,
    ValidationEpochRuntime,
)
from helpers.training.metrics import TrainingHealthTracker
from helpers.training.reporting import TrainingEmailContext


@dataclass
class BestMetricState:
    val_auprc: float | None = None
    val_mcc: float | None = None
    val_auroc: float | None = None
    val_loss: float | None = None


@dataclass
class EpochRunState:
    training_successful: bool
    best: BestMetricState
    metadata_best_path: str
    amp_log: dict[str, str]


@dataclass(frozen=True)
class RunHParams:
    experiment_name: str
    architecture: str
    encoder: str
    optimizer_name: str
    base_learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    workers: int
    seed: int
    patience: int
    train_len: int
    val_len: int
    alpha_bce: float
    beta_dice_bg: float
    gamma_dice_fg: float
    run_ohem: bool
    ohem_start_epoch: int
    ohem_ratio: float
    ohem_min_kept: int


@dataclass(frozen=True)
class EpochRunConfig:
    health: TrainingHealthTracker
    start_epoch: int
    num_epochs: int
    architecture: str
    unleashed: bool
    train_epoch_fn: Any
    validate_epoch_fn: Any
    early_stopping: Any
    track_epoch_metrics_fn: Any
    loss_fn: Any
    device: torch.device
    accumulation_steps: int
    amp_precision: str
    gpu_normalizer: torch.nn.Module
    gpu_downscale: torch.nn.Module
    use_artifact_aware_loss: bool = False
    profile_validation_timing: bool = False
    run: Any | None = None


@dataclass(frozen=True)
class FinalizeArtifactsConfig:
    best: BestMetricState
    metadata_best_path: str
    fallback_checkpoint_path: str | None
    device: torch.device
    load_checkpoint_fn: Any
    get_previous_metrics_fn: Any
    save_metadata_fn: Any
    save_metadata_kwargs: dict[str, Any]
    create_email_body_fn: Callable[[TrainingEmailContext], str]
    send_email_fn: Any
    email_sender: str
    email_recipients: list[str]
    email_password: str
    experiment_name: str


@dataclass(frozen=True)
class EpochMetricSnapshot:
    current_epoch_num: int
    train_loss: float
    val_loss: float
    val_auprc: float
    val_auroc: float
    val_mcc_star: float


def _progress_write(message: str) -> None:
    """Write progress messages without breaking live tqdm rendering."""

    print(message, file=sys.__stdout__)


def build_run_hparams(config: RunHParams) -> dict[str, Any]:
    """Build the Aim hparams payload for a single architecture run.

    The payload records whether Stage 9 used the baseline BCE+Dice loss or the
    OHEM-enhanced variant so experiment comparisons and resume provenance remain
    explicit.
    """

    return {
        "base_learning_rate": config.base_learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "workers": config.workers,
        "seed": config.seed,
        "label": config.experiment_name,
        "optimizer": config.optimizer_name,
        "loss": "BCEDiceHybridLossPaper",
        "model": f"{config.architecture}_{config.encoder}",
        "encoder_weights": "IMAGENET",
        "patience": config.patience,
        "train_len": config.train_len,
        "val_len": config.val_len,
        "loss_alpha_bce": config.alpha_bce,
        "loss_beta_dice_bg": config.beta_dice_bg,
        "loss_gamma_dice_fg": config.gamma_dice_fg,
        "run_ohem": config.run_ohem,
        "ohem_start_epoch": config.ohem_start_epoch,
        "ohem_ratio": config.ohem_ratio,
        "ohem_min_kept": config.ohem_min_kept,
    }


def _validation_failure_reason(health: TrainingHealthTracker) -> str:
    if health.epoch["val_naninf_loss"] > 0:
        return "NAN_INF_VALIDATION_LOSS"
    if health.epoch["val_skip_reasons"].get("no_samples_processed", 0) > 0:
        return "NO_VALIDATION_SAMPLES"
    if health.epoch["val_collapsed"] == 1:
        return "COLLAPSE"
    if health.epoch["val_invalid_metrics"] == 1:
        return "INVALID_METRICS"
    if health.epoch["val_skipped_batches"] > 0:
        return "SKIPPED_VALIDATION_BATCHES"
    return "UNKNOWN"


def _handle_failed_validation(
    *,
    current_epoch_num: int,
    health: TrainingHealthTracker,
) -> bool:
    reason = _validation_failure_reason(health)
    _progress_write(
        f"\n[Epoch {current_epoch_num}] Validation failed ({reason}). Skipping checkpoint."
    )
    health.log_epoch(current_epoch_num)
    if not health.should_emergency_stop():
        return False
    _progress_write("\n!!! EMERGENCY STOP !!!")
    _progress_write(f"Model collapsed {health.current_consecutive_collapses} times in a row.")
    _progress_write("Terminating training to save resources.")
    return True


def _log_epoch_summary(
    *,
    num_epochs: int,
    epoch_start: float,
    metrics: EpochMetricSnapshot,
) -> None:
    epoch_duration = time.time() - epoch_start
    mins, secs = divmod(epoch_duration, 60)
    summary = (
        f"\nE{metrics.current_epoch_num}/{num_epochs} [{int(mins):02d}m{int(secs):02d}s] "
        f"TrL:{metrics.train_loss:.4f} | ValL:{metrics.val_loss:.4f} | "
        f"AUPRC:{metrics.val_auprc:.4f} | AUROC:{metrics.val_auroc:.4f} | "
        f"MCC*:{metrics.val_mcc_star:.4f}"
    )
    _progress_write(summary)


def _track_epoch_metrics(
    *,
    config: EpochRunConfig,
    current_epoch_num: int,
    train_loss: float,
    val_auprc: float,
    val_auroc: float,
    val_mcc_star: float,
) -> None:
    config.track_epoch_metrics_fn(
        config.run,
        epoch=current_epoch_num,
        train_loss=train_loss,
        val_auprc=val_auprc,
        val_auroc=val_auroc,
        val_mcc_star=val_mcc_star,
    )


def _save_best_checkpoint(
    *,
    config: EpochRunConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetricSnapshot,
) -> tuple[BestMetricState | None, str | None]:
    improvement_detected = config.early_stopping(
        EarlyStoppingCheckpoint(
            score=metrics.val_auprc,
            model=model,
            optimizer=optimizer,
            epoch=metrics.current_epoch_num,
            val_loss=metrics.val_loss,
            val_auprc=metrics.val_auprc,
            val_mcc_star=metrics.val_mcc_star,
            val_auroc=metrics.val_auroc,
        )
    )
    if not improvement_detected:
        return None, None
    _progress_write(f"  >>> New Best Model! (AUPRC: {metrics.val_auprc:.4f})")
    return (
        BestMetricState(
            val_auprc=metrics.val_auprc,
            val_mcc=metrics.val_mcc_star,
            val_auroc=metrics.val_auroc,
            val_loss=metrics.val_loss,
        ),
        str(config.early_stopping.output_best_model_path),
    )


def _log_epoch_health(health: TrainingHealthTracker, current_epoch_num: int) -> None:
    if not (health.epoch["train_skipped_batches"] or health.epoch["val_skipped_batches"]):
        return
    _progress_write(
        f"[Health] Epoch {current_epoch_num} | "
        f"Train skip={health.epoch['train_skipped_batches']} | "
        f"Val skip={health.epoch['val_skipped_batches']} | "
        f"Val collapsed={health.epoch['val_collapsed']} | "
        f"Val invalid_metrics={health.epoch['val_invalid_metrics']}"
    )


def run_training_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    val_loader: Any,
    config: EpochRunConfig,
) -> EpochRunState:
    """Run the epoch loop for a single architecture and return resulting state."""

    training_successful = True
    best = BestMetricState()
    metadata_best_path = getattr(config.early_stopping, "output_best_model_path", "")
    amp_log: dict[str, str] = {}

    for epoch in range(config.start_epoch, config.num_epochs):
        config.health.reset_epoch()
        current_epoch_num = epoch + 1
        epoch_start = time.time()

        try:
            train_loss, amp_log = config.train_epoch_fn(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                runtime=TrainEpochRuntime(
                    loss_fn=config.loss_fn,
                    health=config.health,
                    gpu_normalizer=config.gpu_normalizer,
                    gpu_downscale=config.gpu_downscale,
                ),
                config=TrainEpochConfig(
                    device=config.device,
                    current_epoch=epoch,
                    architecture=config.architecture,
                    accumulation_steps=config.accumulation_steps,
                    amp_precision=config.amp_precision,
                    use_artifact_aware_loss=config.use_artifact_aware_loss,
                ),
            )
        except Exception as error:
            print(f"\nTrain Err E{current_epoch_num}:{error}")
            training_successful = False
            break

        try:
            val_loss, val_results = config.validate_epoch_fn(
                model=model,
                optimizer=optimizer,
                dataloader=val_loader,
                runtime=ValidationEpochRuntime(
                    loss_fn=config.loss_fn,
                    health=config.health,
                    gpu_normalizer=config.gpu_normalizer,
                ),
                config=ValidationEpochConfig(
                    device=config.device,
                    architecture=config.architecture,
                    amp_precision=config.amp_precision,
                    profile_timing=config.profile_validation_timing,
                ),
            )
            if val_results is None:
                if _handle_failed_validation(
                    current_epoch_num=current_epoch_num,
                    health=config.health,
                ):
                    training_successful = False
                    break
                continue

            config.health.reset_collapse_counter()
            val_auprc = float(val_results["val_auprc"])
            val_auroc = float(val_results["val_auroc"])
            val_mcc_star = float(val_results["val_mcc_star"])
            metric_snapshot = EpochMetricSnapshot(
                current_epoch_num=current_epoch_num,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auprc=val_auprc,
                val_auroc=val_auroc,
                val_mcc_star=val_mcc_star,
            )
        except Exception as error:
            print(f"\nVal Err E{current_epoch_num}: {error}")
            traceback.print_exc()
            training_successful = False
            break

        _log_epoch_summary(
            num_epochs=config.num_epochs,
            epoch_start=epoch_start,
            metrics=metric_snapshot,
        )

        _track_epoch_metrics(
            config=config,
            current_epoch_num=current_epoch_num,
            train_loss=train_loss,
            val_auprc=val_auprc,
            val_auroc=val_auroc,
            val_mcc_star=val_mcc_star,
        )

        try:
            best_update, best_path = _save_best_checkpoint(
                config=config,
                model=model,
                optimizer=optimizer,
                metrics=metric_snapshot,
            )
            if best_update is not None and best_path is not None:
                best = best_update
                metadata_best_path = best_path
        except Exception as error:
            print(f"ES/Save Err: {error}")

        if not config.unleashed and bool(getattr(config.early_stopping, "early_stop", False)):
            _progress_write(f"Early stopping E{current_epoch_num}.")
            break

        _log_epoch_health(config.health, current_epoch_num)

    return EpochRunState(
        training_successful=training_successful,
        best=best,
        metadata_best_path=metadata_best_path,
        amp_log=amp_log,
    )


def finalize_training_artifacts(config: FinalizeArtifactsConfig) -> str | None:
    """Load the best checkpoint, persist metadata, and send completion email."""

    best_model_path = config.metadata_best_path
    if (
        not best_model_path or not os.path.exists(best_model_path)
    ) and config.fallback_checkpoint_path:
        best_model_path = config.fallback_checkpoint_path

    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Best model not found: {best_model_path}. Skip test.")
        return None

    checkpoint = config.load_checkpoint_fn(best_model_path, map_location=config.device)
    config.best.val_auprc, config.best.val_mcc, config.best.val_auroc, config.best.val_loss = (
        config.get_previous_metrics_fn(
            checkpoint,
            config.best.val_auprc,
            config.best.val_mcc,
            config.best.val_auroc,
            config.best.val_loss,
        )
    )
    metadata_kwargs = config.save_metadata_kwargs
    ohem_settings = OHEMCheckpointSettings(
        run_ohem=bool(metadata_kwargs.get("run_ohem", False)),
        ohem_start_epoch=int(metadata_kwargs.get("ohem_start_epoch", 2)),
        ohem_ratio=float(metadata_kwargs.get("ohem_ratio", 0.25)),
        ohem_min_kept=int(metadata_kwargs.get("ohem_min_kept", 1024)),
    )
    config.save_metadata_fn(
        TrainingMetadataRequest(
            best_val_score=config.best.val_auprc,
            checkpoint=cast(dict[str, Any], checkpoint),
            encoder=str(metadata_kwargs["encoder"]),
            architecture=str(metadata_kwargs["architecture"]),
            metadata_best_path=best_model_path,
            val_loss=config.best.val_loss,
            val_mcc=config.best.val_mcc,
            val_auroc=config.best.val_auroc,
            metadata_dir=str(metadata_kwargs["metadata_dir"]),
            amp_log=metadata_kwargs.get("amp_log", {}),
            base_learning_rate=float(metadata_kwargs["base_learning_rate"]),
            weight_decay=float(metadata_kwargs["weight_decay"]),
            batch_size=int(metadata_kwargs["batch_size"]),
            num_epochs=int(metadata_kwargs["num_epochs"]),
            workers=int(metadata_kwargs["workers"]),
            seed=int(metadata_kwargs["seed"]),
            dataset=metadata_kwargs["dataset"],
            validation_dataset=metadata_kwargs["validation_dataset"],
            patience=int(metadata_kwargs["patience"]),
            optimizer_name=str(metadata_kwargs["optimizer_name"]),
            alpha_bce=float(metadata_kwargs["alpha_bce"]),
            beta_dice_bg=float(metadata_kwargs["beta_dice_bg"]),
            gamma_dice_fg=float(metadata_kwargs["gamma_dice_fg"]),
            execution_mode=cast(str | None, metadata_kwargs.get("execution_mode")),
            use_artifact_aware_loss=bool(metadata_kwargs.get("use_artifact_aware_loss", False)),
            master_manifest_path=metadata_kwargs.get("master_manifest_path"),
            resume_checkpoint=metadata_kwargs.get("resume_checkpoint"),
            ohem=ohem_settings,
        )
    )

    print("Emailing...")
    email_context = TrainingEmailContext(
        checkpoint_path=best_model_path,
        encoder=str(config.save_metadata_kwargs["encoder"]),
        architecture=str(config.save_metadata_kwargs["architecture"]),
        val_loss=config.best.val_loss,
        val_auprc=config.best.val_auprc,
        val_auroc=config.best.val_auroc,
        val_mcc=config.best.val_mcc,
        optimizer_name=str(config.save_metadata_kwargs.get("optimizer_name", "unknown")),
        base_learning_rate=float(config.save_metadata_kwargs.get("base_learning_rate", 0.0)),
        weight_decay=float(config.save_metadata_kwargs.get("weight_decay", 0.0)),
        alpha_bce=float(config.save_metadata_kwargs.get("alpha_bce", 0.0)),
        beta_dice_bg=float(config.save_metadata_kwargs.get("beta_dice_bg", 0.0)),
        gamma_dice_fg=float(config.save_metadata_kwargs.get("gamma_dice_fg", 0.0)),
        use_artifact_aware_loss=bool(
            config.save_metadata_kwargs.get("use_artifact_aware_loss", False)
        ),
        master_manifest_path=config.save_metadata_kwargs.get("master_manifest_path"),
        run_ohem=bool(config.save_metadata_kwargs.get("run_ohem", False)),
        ohem_start_epoch=int(config.save_metadata_kwargs.get("ohem_start_epoch", 2)),
        ohem_ratio=float(config.save_metadata_kwargs.get("ohem_ratio", 0.25)),
        ohem_min_kept=int(config.save_metadata_kwargs.get("ohem_min_kept", 1024)),
    )
    body = config.create_email_body_fn(email_context)
    config.send_email_fn(
        f"Finished: {config.experiment_name}",
        body,
        config.email_sender,
        config.email_recipients,
        config.email_password,
    )
    return best_model_path
