from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.training.metrics import TrainingHealthTracker
from helpers.training.pipeline import (
    BestMetricState,
    build_run_hparams,
    finalize_training_artifacts,
    run_training_epochs,
)


def test_build_run_hparams_preserves_existing_fields() -> None:
    hparams = build_run_hparams(
        experiment_name="exp",
        architecture="UNET++",
        encoder="resnet34",
        optimizer_name="AdamW",
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=4,
        seed=7,
        patience=3,
        train_len=20,
        val_len=5,
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
        run_ohem=True,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    assert hparams["label"] == "exp"
    assert hparams["model"] == "UNET++_resnet34"
    assert hparams["loss"] == "BCEDiceHybridLossPaper"
    assert hparams["loss_gamma_dice_fg"] == 0.8
    assert hparams["run_ohem"] is True


@dataclass
class _EarlyStoppingStub:
    early_stop: bool = False
    output_best_model_path: str = "best.pth"
    _current_best_checkpoint_on_disk_path: str | None = None
    calls: list[tuple[float, int]] | None = None

    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        v_loss: float,
        val_auprc: float,
        val_mcc_star: float,
        val_auroc: float,
    ) -> bool:
        del model, optimizer, v_loss, val_auprc, val_mcc_star, val_auroc
        if self.calls is None:
            self.calls = []
        self.calls.append((score, epoch))
        self._current_best_checkpoint_on_disk_path = self.output_best_model_path
        return True


def test_run_training_epochs_updates_best_metrics() -> None:
    health = TrainingHealthTracker(name="run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    early_stopping = _EarlyStoppingStub()

    metric_state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        health=health,
        start_epoch=0,
        num_epochs=1,
        architecture="UNET++",
        unleashed=False,
        train_epoch_fn=lambda **kwargs: (0.4, {"amp": "ok"}),
        validate_epoch_fn=lambda **kwargs: (
            0.3,
            {"val_auprc": 0.8, "val_auroc": 0.7, "val_mcc_star": 0.6, "fg_prevalence_at_05": 0.5},
        ),
        early_stopping=early_stopping,
        track_epoch_metrics_fn=lambda *args, **kwargs: None,
        loss_fn=object(),
        device=torch.device("cpu"),
        accumulation_steps=1,
        amp_precision="fp16",
        gpu_normalizer=torch.nn.Identity(),
        gpu_downscale=torch.nn.Identity(),
        run=None,
    )

    assert metric_state.training_successful is True
    assert metric_state.best.val_auprc == 0.8
    assert metric_state.best.val_loss == 0.3
    assert metric_state.amp_log == {"amp": "ok"}


def test_finalize_training_artifacts_saves_metadata_and_sends_email(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pth"
    checkpoint_path.write_bytes(b"x")
    calls: dict[str, Any] = {}

    def _load_checkpoint(path: str, map_location: torch.device) -> dict[str, float]:
        del map_location
        calls["load"] = path
        return {"score": 1.0}

    def _get_previous_metrics(
        checkpoint: dict[str, float],
        val_auprc: float | None,
        val_mcc: float | None,
        val_auroc: float | None,
        val_loss: float | None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        del checkpoint
        return (val_auprc, val_mcc, val_auroc, val_loss)

    def _save_metadata(**kwargs: Any) -> None:
        calls["metadata"] = kwargs

    def _create_email_body(**kwargs: Any) -> str:
        calls["email_body"] = kwargs
        return f"{kwargs['checkpoint_path']}|{kwargs['encoder']}|{kwargs['architecture']}"

    def _send_email(
        subject: str, body: str, sender: str, recipients: list[str], password: str
    ) -> None:
        calls["email"] = (subject, body, sender, recipients, password)

    best = BestMetricState(val_auprc=0.8, val_mcc=0.7, val_auroc=0.75, val_loss=0.3)
    final_path = finalize_training_artifacts(
        best=best,
        metadata_best_path=str(checkpoint_path),
        fallback_checkpoint_path=None,
        device=torch.device("cpu"),
        load_checkpoint_fn=_load_checkpoint,
        get_previous_metrics_fn=_get_previous_metrics,
        save_metadata_fn=_save_metadata,
        save_metadata_kwargs={
            "architecture": "UNET++",
            "encoder": "resnet34",
            "metadata_dir": "meta",
            "optimizer_name": "AdamW",
            "base_learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "alpha_bce": 0.6,
            "beta_dice_bg": 0.2,
            "gamma_dice_fg": 0.8,
            "artifact_index_path": "artifact.parquet",
            "use_artifact_aware_loss": True,
            "run_ohem": True,
            "ohem_start_epoch": 2,
            "ohem_ratio": 0.25,
            "ohem_min_kept": 1024,
        },
        create_email_body_fn=_create_email_body,
        send_email_fn=_send_email,
        email_sender="sender@example.com",
        email_recipients=["a@example.com"],
        email_password="secret",
        experiment_name="exp",
    )

    assert final_path == str(checkpoint_path)
    assert calls["load"] == str(checkpoint_path)
    assert calls["metadata"]["best_val_score"] == 0.8
    assert calls["email"][0] == "Finished: exp"
    assert calls["email_body"]["val_auprc"] == 0.8
    assert calls["email_body"]["val_loss"] == 0.3
    assert calls["email_body"]["use_artifact_aware_loss"] is True
    assert calls["email_body"]["run_ohem"] is True


def test_run_training_epochs_stops_when_training_step_raises() -> None:
    health = TrainingHealthTracker(name="run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        health=health,
        start_epoch=0,
        num_epochs=1,
        architecture="UNET++",
        unleashed=False,
        train_epoch_fn=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("train failed")),
        validate_epoch_fn=lambda **kwargs: (0.0, None),
        early_stopping=_EarlyStoppingStub(),
        track_epoch_metrics_fn=lambda *args, **kwargs: None,
        loss_fn=object(),
        device=torch.device("cpu"),
        accumulation_steps=1,
        amp_precision="fp16",
        gpu_normalizer=torch.nn.Identity(),
        gpu_downscale=torch.nn.Identity(),
        run=None,
    )

    assert state.training_successful is False


def test_run_training_epochs_emergency_stops_after_repeated_validation_collapse() -> None:
    health = TrainingHealthTracker(name="run", patience_collapse=1)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fake_validate(**kwargs: Any) -> tuple[float, None]:
        del kwargs
        health.mark_val_collapsed()
        return 0.3, None

    state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        health=health,
        start_epoch=0,
        num_epochs=2,
        architecture="UNET++",
        unleashed=False,
        train_epoch_fn=lambda **kwargs: (0.4, {"amp": "ok"}),
        validate_epoch_fn=fake_validate,
        early_stopping=_EarlyStoppingStub(),
        track_epoch_metrics_fn=lambda *args, **kwargs: None,
        loss_fn=object(),
        device=torch.device("cpu"),
        accumulation_steps=1,
        amp_precision="fp16",
        gpu_normalizer=torch.nn.Identity(),
        gpu_downscale=torch.nn.Identity(),
        run=None,
    )

    assert state.training_successful is False


def test_finalize_training_artifacts_uses_fallback_checkpoint(tmp_path: Path) -> None:
    fallback_path = tmp_path / "fallback.pth"
    fallback_path.write_bytes(b"x")
    calls: dict[str, Any] = {}

    best = BestMetricState(val_auprc=0.9, val_mcc=0.8, val_auroc=0.7, val_loss=0.2)
    result = finalize_training_artifacts(
        best=best,
        metadata_best_path=str(tmp_path / "missing.pth"),
        fallback_checkpoint_path=str(fallback_path),
        device=torch.device("cpu"),
        load_checkpoint_fn=lambda path, map_location: (
            calls.setdefault("load", path) or {"epoch": 1}
        ),
        get_previous_metrics_fn=lambda checkpoint, a, b, c, d: (a, b, c, d),
        save_metadata_fn=lambda **kwargs: calls.setdefault("metadata", kwargs),
        save_metadata_kwargs={
            "architecture": "UNET++",
            "encoder": "resnet34",
            "metadata_dir": "meta",
        },
        create_email_body_fn=lambda **kwargs: (
            f"{kwargs['checkpoint_path']}|{kwargs['encoder']}|{kwargs['architecture']}"
        ),
        send_email_fn=lambda *args: calls.setdefault("email", args),
        email_sender="sender@example.com",
        email_recipients=["a@example.com"],
        email_password="secret",
        experiment_name="exp",
    )

    assert result == str(fallback_path)
    assert calls["load"] == str(fallback_path)


def test_finalize_training_artifacts_returns_none_when_no_checkpoint_exists(tmp_path: Path) -> None:
    result = finalize_training_artifacts(
        best=BestMetricState(),
        metadata_best_path=str(tmp_path / "missing.pth"),
        fallback_checkpoint_path=None,
        device=torch.device("cpu"),
        load_checkpoint_fn=lambda path, map_location: {"epoch": 1},
        get_previous_metrics_fn=lambda checkpoint, a, b, c, d: (a, b, c, d),
        save_metadata_fn=lambda **kwargs: None,
        save_metadata_kwargs={
            "architecture": "UNET++",
            "encoder": "resnet34",
            "metadata_dir": "meta",
        },
        create_email_body_fn=lambda path, encoder, architecture: path,
        send_email_fn=lambda *args: None,
        email_sender="sender@example.com",
        email_recipients=["a@example.com"],
        email_password="secret",
        experiment_name="exp",
    )

    assert result is None
