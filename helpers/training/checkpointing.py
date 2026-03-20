from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch


class EarlyStopping:
    """Save the single best checkpoint and stop after patience is exhausted."""

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = True,
        delta: float = 0.0001,
        output_best_model_path: str = "best_model.pth",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score: float | None = None
        self.delta = delta
        self.output_best_model_path = output_best_model_path
        self._current_best_checkpoint_on_disk_path: str | None = None

        output_dir = os.path.dirname(self.output_best_model_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def set_initial_best_checkpoint_path(self, path: str | None) -> None:
        if path and os.path.exists(path):
            self._current_best_checkpoint_on_disk_path = path
            if self.verbose:
                print(f"EarlyStopping: Initial best checkpoint set to {os.path.basename(path)}")
        else:
            self._current_best_checkpoint_on_disk_path = None
            if self.verbose:
                print("EarlyStopping: No initial best checkpoint path provided/found.")

    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        val_auprc: float,
        val_mcc_star: float,
        val_auroc: float,
    ) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss,
                model,
                optimizer,
                epoch,
                val_auprc,
                val_mcc_star,
                score,
                val_auroc,
            )
            return True

        improvement_detected = score > self.best_score + self.delta
        if improvement_detected:
            previous_best_score_for_log = self.best_score
            self.best_score = score
            self.save_checkpoint(
                val_loss,
                model,
                optimizer,
                epoch,
                val_auprc,
                val_mcc_star,
                score,
                val_auroc,
                previous_best_score_for_log,
            )
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    "EarlyStopping counter: "
                    f"{self.counter} out of {self.patience} "
                    f"(Best score: {self.best_score:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True

        return improvement_detected

    def save_checkpoint(
        self,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_auprc: float,
        val_mcc_star: float,
        score: float,
        val_auroc: float,
        previous_best_score_for_log: float | None = None,
    ) -> None:
        if self.verbose and previous_best_score_for_log is None:
            print(f"Initial best score: {score:.6f}. Saving model...")

        original_model = getattr(model, "_orig_mod", model)
        model_to_save = cast(torch.nn.Module, original_model)
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_score": score,
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "val_mcc_star": val_mcc_star,
            "is_compiled": hasattr(model, "_orig_mod"),
        }

        try:
            torch.save(save_dict, self.output_best_model_path)
            if self.verbose:
                print(f"  New best model saved to: {self.output_best_model_path}")
            self._current_best_checkpoint_on_disk_path = self.output_best_model_path
        except Exception as error:
            print(f"Error saving best model checkpoint to {self.output_best_model_path}: {error}")


def load_checkpoint_for_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    checkpoint_path: str | None,
    device: torch.device,
) -> int:
    """Restore model, optimizer, and early-stopping state from a checkpoint."""

    start_epoch = 0
    if not checkpoint_path:
        print("No resume checkpoint provided: training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    checkpoint_path = checkpoint_path.strip()
    if checkpoint_path == "":
        print("Empty resume checkpoint string: training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    if not os.path.exists(checkpoint_path):
        print(f"Resume checkpoint not found at {checkpoint_path}. Training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    print(f"\n*** Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        print("Checkpoint missing 'model_state_dict'. Training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    if hasattr(model, "_orig_mod"):
        compiled_model = cast(torch.nn.Module, model._orig_mod)
        compiled_model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded from checkpoint.")
        except Exception as error:
            print(f"Warning: could not load optimizer state: {error}")

    best_score = checkpoint.get("best_val_score")
    if best_score is not None:
        early_stopping.best_score = float(best_score)
        early_stopping.counter = 0
        print(f"Loaded EarlyStopping best_val_score = {best_score:.6f}")

    early_stopping.set_initial_best_checkpoint_path(checkpoint_path)
    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch
    print(
        f"Last finished epoch in checkpoint: {last_epoch}. Next epoch will be {last_epoch + 1}.\n"
    )
    return start_epoch


def get_previous_metrics(
    checkpoint: dict[str, Any],
    best_val_score_for_architecture_auprc: float | None,
    best_val_score_for_architecture_mcc: float | None,
    best_val_score_for_architecture_auroc: float | None,
    best_val_score_for_architecture_loss: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Read saved validation metrics from a checkpoint."""

    best_val_score_for_architecture_auprc = checkpoint.get(
        "val_auprc", checkpoint.get("best_val_score")
    )
    best_val_score_for_architecture_mcc = checkpoint.get("val_mcc_star", 0.0)
    best_val_score_for_architecture_auroc = checkpoint.get("val_auroc", 0.0)
    best_val_score_for_architecture_loss = checkpoint.get("val_loss", 0.0)

    print(
        "\n"
        "  Restored best metrics from checkpoint: "
        f"VAL_AUPRC={best_val_score_for_architecture_auprc:.4f},\n"
        f"  VAL_AUROC={best_val_score_for_architecture_auroc:.4f},\n"
        f"  VAL_MCC={best_val_score_for_architecture_mcc:.4f},\n"
        f"  VAL_LOSS={best_val_score_for_architecture_loss:.4f}\n"
    )

    return (
        best_val_score_for_architecture_auprc,
        best_val_score_for_architecture_mcc,
        best_val_score_for_architecture_auroc,
        best_val_score_for_architecture_loss,
    )


def save_metadata(
    best_val_score: float | None,
    checkpoint: dict[str, Any],
    encoder: str,
    architecture: str,
    metadata_best_path: str,
    val_loss: float | None,
    val_mcc: float | None,
    val_auroc: float | None,
    metadata_dir: str,
    amp_log: Any,
    base_learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    workers: int,
    seed: int,
    dataset: str,
    patience: int,
    optimizer_name: str,
    alpha_bce: float,
    beta_dice_bg: float,
    gamma_dice_fg: float,
) -> None:
    """Persist model metadata next to the best checkpoint."""

    import json

    meta_filename = os.path.join(metadata_dir, os.path.basename(metadata_best_path))
    meta_filename = meta_filename.replace(".pth", "_meta.json")

    metadata = {
        "best_val_auprc_pixel_score": best_val_score,
        "pixel_val_loss": val_loss,
        "pixel_val_mcc": val_mcc,
        "pixel_val_auroc": val_auroc,
        "best_model_epoch": checkpoint.get("epoch", "?"),
        "checkpoint_path": metadata_best_path,
        "encoder": encoder,
        "architecture": architecture,
        "hyperparameters": {
            "amp_precision": amp_log,
            "Learning_rate": base_learning_rate,
            "Weight_Decay": weight_decay,
            "Batch_Size": batch_size,
            "Num_Epochs": num_epochs,
            "Workers": workers,
            "Seed": seed,
            "Dataset": dataset,
            "Patience": patience,
            "Optimizer": optimizer_name,
            "Loss_Function": "BCEDiceHybrid",
            "Loss_Weights": {
                "alpha_bce": alpha_bce,
                "beta_dice_bg": beta_dice_bg,
                "gamma_dice_fg": gamma_dice_fg,
            },
        },
    }

    try:
        with Path(meta_filename).open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle, indent=4)
        print(f"Saved metadata to: {meta_filename}")
    except Exception as error:
        print(f"Error saving metadata file {meta_filename}: {error}")
