from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from importlib import import_module
from os import PathLike
from typing import Any


def _format_metric(name: str, value: float | None) -> str:
    """Format one optional metric line for email output."""

    rendered = "n/a" if value is None else f"{value:.4f}"
    return f"{name}: {rendered}"


def create_email_body(
    checkpoint_path: str,
    encoder: str,
    architecture: str,
    val_loss: float | None,
    val_auprc: float | None,
    val_auroc: float | None,
    val_mcc: float | None,
    optimizer_name: str,
    base_learning_rate: float,
    weight_decay: float,
    alpha_bce: float,
    beta_dice_bg: float,
    gamma_dice_fg: float,
    use_artifact_aware_loss: bool,
    artifact_index_path: str | PathLike[str] | None,
) -> str:
    """Create the completion email body for a finished training run."""

    artifact_index_display = (
        os.fspath(artifact_index_path) if artifact_index_path is not None else "n/a"
    )
    artifact_mode_display = "enabled" if use_artifact_aware_loss else "disabled"
    return (
        f"Training {architecture} finished.\n\n"
        f"Checkpoint Path: {checkpoint_path}\n\n"
        f"--- ENCODER: {encoder} ---\n"
        f"\nBest validation metrics\n"
        f"{_format_metric('val_loss', val_loss)}\n"
        f"{_format_metric('val_auprc', val_auprc)}\n"
        f"{_format_metric('val_auroc', val_auroc)}\n"
        f"{_format_metric('val_mcc', val_mcc)}\n"
        f"\nRun settings\n"
        f"optimizer: {optimizer_name}\n"
        f"learning_rate: {base_learning_rate}\n"
        f"weight_decay: {weight_decay}\n"
        f"loss_alpha_bce: {alpha_bce}\n"
        f"loss_beta_dice_bg: {beta_dice_bg}\n"
        f"loss_gamma_dice_fg: {gamma_dice_fg}\n"
        f"artifact_aware_loss: {artifact_mode_display}\n"
        f"artifact_index_path: {artifact_index_display}\n"
    )


def send_email(
    subject: str,
    body: str,
    sender: str,
    recipients: list[str],
    password: str,
) -> None:
    """Send a completion email through Gmail SMTP."""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
            smtp_server.login(sender, password)
            smtp_server.sendmail(sender, recipients, msg.as_string())
        print("Email sent successfully!")
    except Exception as error:
        print(f"Error sending email: {error}")


def ensure_aim_repo(repo_path: str, aim_module: Any | None = None) -> None:
    """Create the Aim repository directory when it does not exist yet."""

    if aim_module is None:
        aim_module = import_module("aim")

    if not os.path.exists(repo_path):
        print(f"Aim repository not found at {repo_path}. Initializing...")
        os.makedirs(repo_path, exist_ok=True)
        try:
            repo = aim_module.Repo.init(repo_path)
            print(f"Aim repository initialized at: {repo.path}")
        except Exception as error:
            print(f"An unexpected error occurred during Aim repo setup: {error}")
    else:
        print(f"Using existing Aim repository at: {repo_path}")


def create_aim_run(
    experiment_name: str,
    repo_path: str,
    hparams: dict[str, Any],
    run_factory: Any | None = None,
) -> Any | None:
    """Create an Aim run and seed it with hparams."""

    print(f"Init Aim: {experiment_name}")
    try:
        if run_factory is None:
            run_factory = import_module("aim").Run
        run = run_factory(experiment=experiment_name, repo=repo_path)
        run["hparams"] = hparams
        print("Aim run initialized.")
        return run
    except Exception as error:
        print(f"Aim Init Err: {error}.")
        return None


def track_epoch_metrics(
    run: Any | None,
    epoch: int,
    train_loss: float,
    val_auprc: float,
    val_auroc: float,
    val_mcc_star: float,
) -> None:
    """Track per-epoch metrics into Aim when a run exists."""

    if run is None:
        return
    try:
        run.track(train_loss, "loss", epoch=epoch, context={"subset": "train"})
        run.track(val_auprc, "pixel_auprc", epoch=epoch, context={"subset": "val"})
        run.track(val_auroc, "pixel_auroc", epoch=epoch, context={"subset": "val"})
        run.track(val_mcc_star, "pixel_mcc", epoch=epoch, context={"subset": "val"})
    except Exception as error:
        print(f"Aim Log Err: {error}")


def close_aim_run(run: Any | None) -> None:
    """Close an Aim run if one exists."""

    if run is not None:
        run.close()
        print("Aim run closed.")
