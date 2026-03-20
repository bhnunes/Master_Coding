from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from helpers.training_metrics import TrainingHealthTracker


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
) -> dict[str, Any]:
    """Build the Aim hparams payload for a single architecture run."""

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
    run: Any | None,
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
                print(
                    f"\n[Epoch {current_epoch_num}] Validation failed ({reason}). "
                    "Skipping checkpoint."
                )
                if health.should_emergency_stop():
                    print("\n!!! EMERGENCY STOP !!!")
                    print(f"Model collapsed {health.current_consecutive_collapses} times in a row.")
                    print("Terminating training to save resources.")
                    training_successful = False
                    break
                health.log_epoch(current_epoch_num)
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
        print(
            f"\nE{current_epoch_num}/{num_epochs} [{int(mins):02d}m{int(secs):02d}s] "
            f"Tr L:{train_loss:.4f}|Val AUPRC:{val_auprc:.4f} "
            f"AUROC:{val_auroc:.4f} MCC*:{val_mcc_star:.4f}"
        )

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
                print(f"  >>> New Best Model! (AUPRC: {val_auprc:.4f})")
        except Exception as error:
            print(f"ES/Save Err: {error}")

        if not unleashed and bool(getattr(early_stopping, "early_stop", False)):
            print(f"Early stopping E{current_epoch_num}.")
            break

        print(f"Epoch {epoch} completed.")
        health.log_epoch(current_epoch_num)

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
        best_model_path,
        save_metadata_kwargs["encoder"],
        save_metadata_kwargs["architecture"],
    )
    send_email_fn(
        f"Finished: {experiment_name}",
        body,
        email_sender,
        email_recipients,
        email_password,
    )
    return best_model_path
