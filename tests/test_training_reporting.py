from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from helpers.training.reporting import (
    TrainingEmailContext,
    close_aim_run,
    create_aim_run,
    create_email_body,
    ensure_aim_repo,
    send_email,
    track_epoch_metrics,
)


def test_create_email_body_includes_checkpoint_and_encoder() -> None:
    body = create_email_body(
        TrainingEmailContext(
            checkpoint_path="/tmp/best.pth",
            encoder="resnet34",
            architecture="UNET++",
            val_loss=0.2,
            val_auprc=0.8,
            val_auroc=0.7,
            val_mcc=0.6,
            optimizer_name="AdamW",
            base_learning_rate=1e-3,
            weight_decay=1e-4,
            alpha_bce=0.1,
            beta_dice_bg=0.2,
            gamma_dice_fg=0.3,
            use_artifact_aware_loss=True,
            master_manifest_path="/tmp/master_manifest.sqlite",
            run_ohem=True,
            ohem_start_epoch=2,
            ohem_ratio=0.25,
            ohem_min_kept=1024,
        )
    )

    assert "UNET++" in body
    assert "resnet34" in body
    assert "/tmp/best.pth" in body
    assert "val_auprc: 0.8000" in body
    assert "loss_gamma_dice_fg: 0.3" in body
    assert "artifact_aware_loss: enabled" in body
    assert "run_ohem: enabled" in body


def test_send_email_uses_smtp_ssl(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    class DummySmtp:
        def __enter__(self) -> DummySmtp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: Any,
        ) -> None:
            return None

        def login(self, sender: str, password: str) -> None:
            captured["login"] = (sender, password)

        def sendmail(self, sender: str, recipients: list[str], message: str) -> None:
            captured["sendmail"] = (sender, recipients, message)

    monkeypatch.setattr(
        "helpers.training.reporting.smtplib.SMTP_SSL", lambda host, port: DummySmtp()
    )

    send_email("Done", "Body", "sender@example.com", ["a@example.com"], "secret")

    assert captured["login"] == ("sender@example.com", "secret")
    sendmail_args = cast(tuple[str, list[str], str], captured["sendmail"])
    assert "Subject: Done" in sendmail_args[2]


def test_ensure_aim_repo_initializes_missing_repo(tmp_path: Any) -> None:
    calls: dict[str, object] = {}

    class DummyRepo:
        path = "repo-path"

    fake_aim: Any = SimpleNamespace(
        Repo=SimpleNamespace(
            init=lambda path: calls.setdefault("init_path", path) or DummyRepo(),
        )
    )

    ensure_aim_repo(str(tmp_path / "aim"), aim_module=fake_aim)

    assert calls["init_path"] == str(tmp_path / "aim")


def test_create_aim_run_sets_hparams() -> None:
    captured: dict[str, object] = {}

    class DummyRun(dict[str, object]):
        def __init__(self, experiment: str, repo: str) -> None:
            super().__init__()
            captured["experiment"] = experiment
            captured["repo"] = repo

    run = create_aim_run(
        run_factory=DummyRun,
        experiment_name="exp",
        repo_path="repo",
        hparams={"lr": 1e-3},
    )

    assert run is not None
    assert captured == {"experiment": "exp", "repo": "repo"}
    assert run["hparams"] == {"lr": 1e-3}


def test_create_aim_run_returns_none_when_factory_fails() -> None:
    run = create_aim_run(
        run_factory=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        experiment_name="exp",
        repo_path="repo",
        hparams={"lr": 1e-3},
    )

    assert run is None


def test_track_epoch_metrics_records_train_and_validation_values() -> None:
    class DummyRun:
        def __init__(self) -> None:
            self.calls: list[tuple[float, str, int, dict[str, str]]] = []

        def track(self, value: float, name: str, epoch: int, context: dict[str, str]) -> None:
            self.calls.append((value, name, epoch, context))

    run = DummyRun()

    track_epoch_metrics(
        run, epoch=3, train_loss=0.5, val_auprc=0.7, val_auroc=0.8, val_mcc_star=0.6
    )

    assert run.calls == [
        (0.5, "loss", 3, {"subset": "train"}),
        (0.7, "pixel_auprc", 3, {"subset": "val"}),
        (0.8, "pixel_auroc", 3, {"subset": "val"}),
        (0.6, "pixel_mcc", 3, {"subset": "val"}),
    ]


def test_close_aim_run_is_safe_for_none() -> None:
    close_aim_run(None)


def test_track_epoch_metrics_ignores_missing_run() -> None:
    track_epoch_metrics(
        None, epoch=1, train_loss=0.1, val_auprc=0.2, val_auroc=0.3, val_mcc_star=0.4
    )


def test_track_epoch_metrics_handles_logging_errors() -> None:
    class FailingRun:
        def track(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("boom")

    track_epoch_metrics(
        FailingRun(),
        epoch=1,
        train_loss=0.1,
        val_auprc=0.2,
        val_auroc=0.3,
        val_mcc_star=0.4,
    )


def test_close_aim_run_closes_existing_run() -> None:
    calls: list[str] = []

    class DummyRun:
        def close(self) -> None:
            calls.append("closed")

    close_aim_run(DummyRun())

    assert calls == ["closed"]
