from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.training.checkpointing import EarlyStoppingCheckpoint, TrainingMetadataRequest
from helpers.training.metrics import TrainingHealthTracker
from helpers.training.pipeline import (
    BestMetricState,
    EpochRunConfig,
    FinalizeArtifactsConfig,
    RunHParams,
    build_run_hparams,
    finalize_training_artifacts,
    run_training_epochs,
)
from helpers.training.reporting import TrainingEmailContext, create_email_body

BASE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 4
SEED = 7
PATIENCE = 3
TRAIN_LEN = 20
VAL_LEN = 5
ALPHA_BCE = 0.6
BETA_DICE_BG = 0.2
GAMMA_DICE_FG = 0.8
OHEM_START_EPOCH = 2
OHEM_RATIO = 0.25
OHEM_MIN_KEPT = 1024
TRAIN_LOSS = 0.4
VAL_LOSS = 0.3
VAL_AUPRC = 0.8
VAL_AUROC = 0.7
VAL_MCC = 0.6
FG_PREVALENCE = 0.5
FALLBACK_VAL_AUPRC = 0.9
FALLBACK_VAL_MCC = 0.8
FALLBACK_VAL_AUROC = 0.7
FALLBACK_VAL_LOSS = 0.2


def _metadata_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "architecture": "UNET++",
        "encoder": "resnet34",
        "metadata_dir": "meta",
        "amp_log": {"amp": "ok"},
        "base_learning_rate": BASE_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "workers": NUM_WORKERS,
        "seed": SEED,
        "dataset": {
            "path": "train",
            "sha256": "train-sha",
            "source_signature": "train-source",
        },
        "validation_dataset": {
            "path": "validation",
            "sha256": "validation-sha",
            "source_signature": "validation-source",
        },
        "patience": PATIENCE,
        "optimizer_name": "AdamW",
        "alpha_bce": ALPHA_BCE,
        "beta_dice_bg": BETA_DICE_BG,
        "gamma_dice_fg": GAMMA_DICE_FG,
    }
    payload.update(overrides)
    return payload


def test_build_run_hparams_preserves_existing_fields() -> None:
    hparams = build_run_hparams(
        RunHParams(
            experiment_name="exp",
            architecture="UNET++",
            encoder="resnet34",
            optimizer_name="AdamW",
            base_learning_rate=BASE_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            workers=NUM_WORKERS,
            seed=SEED,
            patience=PATIENCE,
            train_len=TRAIN_LEN,
            val_len=VAL_LEN,
            alpha_bce=ALPHA_BCE,
            beta_dice_bg=BETA_DICE_BG,
            gamma_dice_fg=GAMMA_DICE_FG,
            run_ohem=True,
            ohem_start_epoch=OHEM_START_EPOCH,
            ohem_ratio=OHEM_RATIO,
            ohem_min_kept=OHEM_MIN_KEPT,
        )
    )

    assert hparams["label"] == "exp"
    assert hparams["model"] == "UNET++_resnet34"
    assert hparams["loss"] == "BCEDiceHybridLossPaper"
    assert hparams["loss_gamma_dice_fg"] == GAMMA_DICE_FG
    assert hparams["run_ohem"] is True


@dataclass
class _EarlyStoppingStub:
    early_stop: bool = False
    output_best_model_path: str = "best.pth"
    _current_best_checkpoint_on_disk_path: str | None = None
    calls: list[tuple[float, int]] | None = None

    def __call__(
        self,
        checkpoint: EarlyStoppingCheckpoint,
    ) -> bool:
        if self.calls is None:
            self.calls = []
        self.calls.append((checkpoint.score, checkpoint.epoch))
        self._current_best_checkpoint_on_disk_path = self.output_best_model_path
        return True


def test_run_training_epochs_updates_best_metrics() -> None:
    health = TrainingHealthTracker(name="run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)
    early_stopping = _EarlyStoppingStub()

    metric_state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        config=EpochRunConfig(
            health=health,
            start_epoch=0,
            num_epochs=1,
            architecture="UNET++",
            unleashed=False,
            train_epoch_fn=lambda **kwargs: (TRAIN_LOSS, {"amp": "ok"}),
            validate_epoch_fn=lambda **kwargs: (
                VAL_LOSS,
                {
                    "val_auprc": VAL_AUPRC,
                    "val_auroc": VAL_AUROC,
                    "val_mcc_star": VAL_MCC,
                    "fg_prevalence_at_05": FG_PREVALENCE,
                },
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
        ),
    )

    assert metric_state.training_successful is True
    assert metric_state.best.val_auprc == VAL_AUPRC
    assert metric_state.best.val_loss == VAL_LOSS
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

    def _save_metadata(request: TrainingMetadataRequest) -> None:
        calls["metadata"] = request

    def _create_email_body(context: TrainingEmailContext) -> str:
        calls["email_body"] = context
        return f"{context.checkpoint_path}|{context.encoder}|{context.architecture}"

    def _send_email(
        subject: str, body: str, sender: str, recipients: list[str], password: str
    ) -> None:
        calls["email"] = (subject, body, sender, recipients, password)

    best = BestMetricState(
        val_auprc=VAL_AUPRC,
        val_mcc=VAL_AUROC,
        val_auroc=0.75,
        val_loss=VAL_LOSS,
    )
    final_path = finalize_training_artifacts(
        FinalizeArtifactsConfig(
            best=best,
            metadata_best_path=str(checkpoint_path),
            fallback_checkpoint_path=None,
            device=torch.device("cpu"),
            load_checkpoint_fn=_load_checkpoint,
            get_previous_metrics_fn=_get_previous_metrics,
            save_metadata_fn=_save_metadata,
            save_metadata_kwargs=_metadata_kwargs(
                master_manifest_path="master_manifest.sqlite",
                use_artifact_aware_loss=True,
                run_ohem=True,
                ohem_start_epoch=OHEM_START_EPOCH,
                ohem_ratio=OHEM_RATIO,
                ohem_min_kept=OHEM_MIN_KEPT,
            ),
            create_email_body_fn=_create_email_body,
            send_email_fn=_send_email,
            email_sender="sender@example.com",
            email_recipients=["a@example.com"],
            email_password="secret",
            experiment_name="exp",
        )
    )

    assert final_path == str(checkpoint_path)
    assert calls["load"] == str(checkpoint_path)
    metadata_request = calls["metadata"]
    assert isinstance(metadata_request, TrainingMetadataRequest)
    assert metadata_request.best_val_score == VAL_AUPRC
    assert metadata_request.metadata_best_path == str(checkpoint_path)
    assert metadata_request.ohem.run_ohem is True
    assert calls["email"][0] == "Finished: exp"
    email_context = calls["email_body"]
    assert isinstance(email_context, TrainingEmailContext)
    assert email_context.val_auprc == VAL_AUPRC
    assert email_context.val_loss == VAL_LOSS
    assert email_context.use_artifact_aware_loss is True
    assert email_context.run_ohem is True


def test_run_training_epochs_stops_when_training_step_raises() -> None:
    health = TrainingHealthTracker(name="run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)

    state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        config=EpochRunConfig(
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
        ),
    )

    assert state.training_successful is False


def test_run_training_epochs_emergency_stops_after_repeated_validation_collapse() -> None:
    health = TrainingHealthTracker(name="run", patience_collapse=1)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)

    def fake_validate(**kwargs: Any) -> tuple[float, None]:
        del kwargs
        health.mark_val_collapsed()
        return VAL_LOSS, None

    state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=[object()],
        val_loader=[object()],
        config=EpochRunConfig(
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
        ),
    )

    assert state.training_successful is False


def test_finalize_training_artifacts_uses_fallback_checkpoint(tmp_path: Path) -> None:
    fallback_path = tmp_path / "fallback.pth"
    fallback_path.write_bytes(b"x")
    calls: dict[str, Any] = {}

    best = BestMetricState(
        val_auprc=FALLBACK_VAL_AUPRC,
        val_mcc=FALLBACK_VAL_MCC,
        val_auroc=FALLBACK_VAL_AUROC,
        val_loss=FALLBACK_VAL_LOSS,
    )
    result = finalize_training_artifacts(
        FinalizeArtifactsConfig(
            best=best,
            metadata_best_path=str(tmp_path / "missing.pth"),
            fallback_checkpoint_path=str(fallback_path),
            device=torch.device("cpu"),
            load_checkpoint_fn=lambda path, map_location: (
                calls.setdefault("load", path) or {"epoch": 1}
            ),
            get_previous_metrics_fn=lambda checkpoint, a, b, c, d: (a, b, c, d),
            save_metadata_fn=lambda request: calls.setdefault("metadata", request),
            save_metadata_kwargs=_metadata_kwargs(),
            create_email_body_fn=lambda context: (
                f"{context.checkpoint_path}|{context.encoder}|{context.architecture}"
            ),
            send_email_fn=lambda *args: calls.setdefault("email", args),
            email_sender="sender@example.com",
            email_recipients=["a@example.com"],
            email_password="secret",
            experiment_name="exp",
        )
    )

    assert result == str(fallback_path)
    assert calls["load"] == str(fallback_path)


def test_finalize_training_artifacts_returns_none_when_no_checkpoint_exists(tmp_path: Path) -> None:
    result = finalize_training_artifacts(
        FinalizeArtifactsConfig(
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
            create_email_body_fn=lambda context: context.checkpoint_path,
            send_email_fn=lambda *args: None,
            email_sender="sender@example.com",
            email_recipients=["a@example.com"],
            email_password="secret",
            experiment_name="exp",
        )
    )

    assert result is None


def test_finalize_training_artifacts_uses_real_email_body_contract(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pth"
    checkpoint_path.write_bytes(b"x")
    calls: dict[str, Any] = {}

    result = finalize_training_artifacts(
        FinalizeArtifactsConfig(
            best=BestMetricState(
                val_auprc=VAL_AUPRC,
                val_mcc=VAL_MCC,
                val_auroc=VAL_AUROC,
                val_loss=VAL_LOSS,
            ),
            metadata_best_path=str(checkpoint_path),
            fallback_checkpoint_path=None,
            device=torch.device("cpu"),
            load_checkpoint_fn=lambda path, map_location: {"epoch": 1},
            get_previous_metrics_fn=lambda checkpoint, a, b, c, d: (a, b, c, d),
            save_metadata_fn=lambda request: calls.setdefault("metadata", request),
            save_metadata_kwargs=_metadata_kwargs(
                master_manifest_path="master_manifest.sqlite",
                use_artifact_aware_loss=True,
                run_ohem=True,
            ),
            create_email_body_fn=create_email_body,
            send_email_fn=lambda *args: calls.setdefault("email", args),
            email_sender="sender@example.com",
            email_recipients=["a@example.com"],
            email_password="secret",
            experiment_name="exp",
        )
    )

    assert result == str(checkpoint_path)
    email_args = calls["email"]
    assert email_args[0] == "Finished: exp"
    assert "Checkpoint Path: " + str(checkpoint_path) in email_args[1]
    assert "val_auprc: 0.8000" in email_args[1]
    assert "artifact_aware_loss: enabled" in email_args[1]
    assert "run_ohem: enabled" in email_args[1]
