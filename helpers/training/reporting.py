from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from importlib import import_module
from typing import Any


def create_email_body(checkpoint_path: str, encoder: str, architecture: str) -> str:
    """Create the completion email body for a finished training run."""

    return (
        f"Training {architecture} finished.\n\n"
        f"Checkpoint Path: {checkpoint_path}\n\n"
        f"--- ENCODER: {encoder} ---\n"
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
