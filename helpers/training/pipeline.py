from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from helpers.training.metrics import TrainingHealthTracker


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


def _progress_write(message: str) -> None:
    """Write progress messages without breaking live tqdm rendering."""

    print(message, file=sys.__stdout__)


def build_run_hparams(
    experiment_name: str,
    architecture: str,
    encoder: str,
    optimizer_name: str,
    base_learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    workers: int,
    seed: int,
    patience: int,
    train_len: int,
    val_len: int,
    alpha_bce: float,
    beta_dice_bg: float,
    gamma_dice_fg: float,
    run_ohem: bool,
    ohem_start_epoch: int,
    ohem_ratio: float,
    ohem_min_kept: int,
) -> dict[str, Any]:
    """Build the Aim hparams payload for a single architecture run.

    The payload records whether Stage 9 used the baseline BCE+Dice loss or the
    OHEM-enhanced variant so experiment comparisons and resume provenance remain
    explicit.
    """

    return {
        "base_learning_rate": base_learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "workers": workers,
        "seed": seed,
        "label": experiment_name,
        "optimizer": optimizer_name,
        "loss": "BCEDiceHybridLossPaper",
        "model": f"{architecture}_{encoder}",
        "encoder_weights": "IMAGENET",
        "patience": patience,
        "train_len": train_len,
        "val_len": val_len,
        "loss_alpha_bce": alpha_bce,
        "loss_beta_dice_bg": beta_dice_bg,
        "loss_gamma_dice_fg": gamma_dice_fg,
        "run_ohem": run_ohem,
        "ohem_start_epoch": ohem_start_epoch,
        "ohem_ratio": ohem_ratio,
        "ohem_min_kept": ohem_min_kept,
    }


def run_training_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    val_loader: Any,
    health: TrainingHealthTracker,
    start_epoch: int,
    num_epochs: int,
    architecture: str,
    unleashed: bool,
    train_epoch_fn: Any,
    validate_epoch_fn: Any,
    early_stopping: Any,
    track_epoch_metrics_fn: Any,
    loss_fn: Any,
    device: torch.device,
    accumulation_steps: int,
    amp_precision: str,
    gpu_normalizer: torch.nn.Module,
    gpu_downscale: torch.nn.Module,
    use_artifact_aware_loss: bool = False,
    run: Any | None = None,
) -> EpochRunState:
    """Run the epoch loop for a single architecture and return resulting state."""

    training_successful = True
    best = BestMetricState()
    metadata_best_path = getattr(early_stopping, "output_best_model_path", "")
    amp_log: dict[str, str] = {}

    for epoch in range(start_epoch, num_epochs):
        health.reset_epoch()
        current_epoch_num = epoch + 1
        epoch_start = time.time()

        try:
            train_loss, amp_log = train_epoch_fn(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                current_epoch=epoch,
                loss_fn=loss_fn,
                health=health,
                architecture=architecture,
                accumulation_steps=accumulation_steps,
                amp_precision=amp_precision,
                gpu_normalizer=gpu_normalizer,
                gpu_downscale=gpu_downscale,
                use_artifact_aware_loss=use_artifact_aware_loss,
            )
        except Exception as error:
            print(f"\nTrain Err E{current_epoch_num}:{error}")
            training_successful = False
            break

        try:
            val_loss, val_results = validate_epoch_fn(
                model=model,
                optimizer=optimizer,
                dataloader=val_loader,
                device=device,
                loss_fn=loss_fn,
                health=health,
                architecture=architecture,
                amp_precision=amp_precision,
                gpu_normalizer=gpu_normalizer,
            )
            if val_results is None:
                reason = "UNKNOWN"
                if health.epoch["val_collapsed"] == 1:
                    reason = "COLLAPSE"
                elif health.epoch["val_invalid_metrics"] == 1:
                    reason = "INVALID_METRICS"
                _progress_write(
                    f"\n[Epoch {current_epoch_num}] Validation failed ({reason}). "
                    "Skipping checkpoint."
                )
                if health.should_emergency_stop():
                    _progress_write("\n!!! EMERGENCY STOP !!!")
                    _progress_write(
                        f"Model collapsed {health.current_consecutive_collapses} times in a row."
                    )
                    _progress_write("Terminating training to save resources.")
                    training_successful = False
                    break
                continue

            health.reset_collapse_counter()
            val_auprc = float(val_results["val_auprc"])
            val_auroc = float(val_results["val_auroc"])
            val_mcc_star = float(val_results["val_mcc_star"])
            score = val_auprc
        except Exception as error:
            print(f"\nVal Err E{current_epoch_num}: {error}")
            traceback.print_exc()
            training_successful = False
            break

        epoch_duration = time.time() - epoch_start
        mins, secs = divmod(epoch_duration, 60)
        summary = (
            f"\nE{current_epoch_num}/{num_epochs} [{int(mins):02d}m{int(secs):02d}s] "
            f"TrL:{train_loss:.4f} | ValL:{val_loss:.4f} | "
            f"AUPRC:{val_auprc:.4f} | AUROC:{val_auroc:.4f} | MCC*:{val_mcc_star:.4f}"
        )
        _progress_write(summary)

        track_epoch_metrics_fn(
            run,
            epoch=current_epoch_num,
            train_loss=train_loss,
            val_auprc=val_auprc,
            val_auroc=val_auroc,
            val_mcc_star=val_mcc_star,
        )

        try:
            improvement_detected = early_stopping(
                score,
                model,
                optimizer,
                current_epoch_num,
                val_loss,
                val_auprc,
                val_mcc_star,
                val_auroc,
            )
            if improvement_detected:
                best = BestMetricState(
                    val_auprc=val_auprc,
                    val_mcc=val_mcc_star,
                    val_auroc=val_auroc,
                    val_loss=val_loss,
                )
                metadata_best_path = str(early_stopping.output_best_model_path)
                _progress_write(f"  >>> New Best Model! (AUPRC: {val_auprc:.4f})")
        except Exception as error:
            print(f"ES/Save Err: {error}")

        if not unleashed and bool(getattr(early_stopping, "early_stop", False)):
            _progress_write(f"Early stopping E{current_epoch_num}.")
            break

        if health.epoch["train_skipped_batches"] or health.epoch["val_skipped_batches"]:
            _progress_write(
                f"[Health] Epoch {current_epoch_num} | "
                f"Train skip={health.epoch['train_skipped_batches']} | "
                f"Val skip={health.epoch['val_skipped_batches']} | "
                f"Val collapsed={health.epoch['val_collapsed']} | "
                f"Val invalid_metrics={health.epoch['val_invalid_metrics']}"
            )

    return EpochRunState(
        training_successful=training_successful,
        best=best,
        metadata_best_path=metadata_best_path,
        amp_log=amp_log,
    )


def finalize_training_artifacts(
    best: BestMetricState,
    metadata_best_path: str,
    fallback_checkpoint_path: str | None,
    device: torch.device,
    load_checkpoint_fn: Any,
    get_previous_metrics_fn: Any,
    save_metadata_fn: Any,
    save_metadata_kwargs: dict[str, Any],
    create_email_body_fn: Any,
    send_email_fn: Any,
    email_sender: str,
    email_recipients: list[str],
    email_password: str,
    experiment_name: str,
) -> str | None:
    """Load the best checkpoint, persist metadata, and send completion email."""

    best_model_path = metadata_best_path
    if (not best_model_path or not os.path.exists(best_model_path)) and fallback_checkpoint_path:
        best_model_path = fallback_checkpoint_path

    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Best model not found: {best_model_path}. Skip test.")
        return None

    checkpoint = load_checkpoint_fn(best_model_path, map_location=device)
    best.val_auprc, best.val_mcc, best.val_auroc, best.val_loss = get_previous_metrics_fn(
        checkpoint,
        best.val_auprc,
        best.val_mcc,
        best.val_auroc,
        best.val_loss,
    )
    save_metadata_fn(
        best_val_score=best.val_auprc,
        checkpoint=checkpoint,
        metadata_best_path=best_model_path,
        val_loss=best.val_loss,
        val_mcc=best.val_mcc,
        val_auroc=best.val_auroc,
        **save_metadata_kwargs,
    )

    print("Emailing...")
    body = create_email_body_fn(
        checkpoint_path=best_model_path,
        encoder=save_metadata_kwargs["encoder"],
        architecture=save_metadata_kwargs["architecture"],
        val_loss=best.val_loss,
        val_auprc=best.val_auprc,
        val_auroc=best.val_auroc,
        val_mcc=best.val_mcc,
        optimizer_name=str(save_metadata_kwargs.get("optimizer_name", "unknown")),
        base_learning_rate=float(save_metadata_kwargs.get("base_learning_rate", 0.0)),
        weight_decay=float(save_metadata_kwargs.get("weight_decay", 0.0)),
        alpha_bce=float(save_metadata_kwargs.get("alpha_bce", 0.0)),
        beta_dice_bg=float(save_metadata_kwargs.get("beta_dice_bg", 0.0)),
        gamma_dice_fg=float(save_metadata_kwargs.get("gamma_dice_fg", 0.0)),
        use_artifact_aware_loss=bool(save_metadata_kwargs.get("use_artifact_aware_loss", False)),
        artifact_index_path=save_metadata_kwargs.get("artifact_index_path"),
        run_ohem=bool(save_metadata_kwargs.get("run_ohem", False)),
        ohem_start_epoch=int(save_metadata_kwargs.get("ohem_start_epoch", 2)),
        ohem_ratio=float(save_metadata_kwargs.get("ohem_ratio", 0.25)),
        ohem_min_kept=int(save_metadata_kwargs.get("ohem_min_kept", 1024)),
    )
    send_email_fn(
        f"Finished: {experiment_name}",
        body,
        email_sender,
        email_recipients,
        email_password,
    )
    return best_model_path
