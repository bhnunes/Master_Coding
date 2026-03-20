from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from helpers.training_reporting import (
    close_aim_run,
    create_aim_run,
    create_email_body,
    ensure_aim_repo,
    send_email,
    track_epoch_metrics,
)


def test_create_email_body_includes_checkpoint_and_encoder() -> None:
    body = create_email_body("/tmp/best.pth", "resnet34", "UNET++")

    assert "UNET++" in body
    assert "resnet34" in body
    assert "/tmp/best.pth" in body


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
        "helpers.training_reporting.smtplib.SMTP_SSL", lambda host, port: DummySmtp()
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
