from pathlib import Path

import pytest
import torch

from helpers.training.checkpointing import (
    EarlyStopping,
    get_previous_metrics,
    load_checkpoint_for_resume,
    save_metadata,
)


def test_early_stopping_saves_first_best_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(checkpoint_path),
    )

    improved = early_stopping(
        0.8,
        model,
        optimizer,
        epoch=1,
        val_loss=0.2,
        val_auprc=0.8,
        val_mcc_star=0.7,
        val_auroc=0.9,
    )

    assert improved is True
    assert checkpoint_path.exists()
    assert early_stopping.best_score == 0.8
    assert early_stopping._current_best_checkpoint_on_disk_path == str(checkpoint_path)


def test_early_stopping_triggers_after_patience_without_improvement(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(checkpoint_path),
    )

    early_stopping(0.8, model, optimizer, 1, 0.2, 0.8, 0.7, 0.9)
    improved = early_stopping(0.8, model, optimizer, 2, 0.2, 0.8, 0.7, 0.9)
    early_stopping(0.8, model, optimizer, 3, 0.2, 0.8, 0.7, 0.9)

    assert improved is False
    assert early_stopping.early_stop is True
    assert early_stopping.counter == 2


def test_load_checkpoint_for_resume_restores_model_optimizer_and_score(tmp_path: Path) -> None:
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": saved_model.state_dict(),
            "optimizer_state_dict": saved_optimizer.state_dict(),
            "best_val_score": 0.91,
        },
        checkpoint_path,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(tmp_path / "best_model.pth"),
    )

    start_epoch = load_checkpoint_for_resume(
        model=model,
        optimizer=optimizer,
        early_stopping=early_stopping,
        checkpoint_path=str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert start_epoch == 4
    assert early_stopping.best_score == 0.91
    assert early_stopping.counter == 0
    assert early_stopping._current_best_checkpoint_on_disk_path == str(checkpoint_path)
    for expected, restored in zip(saved_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(expected, restored)


def test_get_previous_metrics_returns_checkpoint_metrics() -> None:
    checkpoint = {
        "val_auprc": 0.8,
        "val_mcc_star": 0.7,
        "val_auroc": 0.9,
        "val_loss": 0.2,
    }

    metrics = get_previous_metrics(checkpoint, None, None, None, None)

    assert metrics == (0.8, 0.7, 0.9, 0.2)


def test_early_stopping_can_clear_missing_initial_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(checkpoint_path))

    early_stopping.set_initial_best_checkpoint_path(str(tmp_path / "missing.pth"))

    assert early_stopping._current_best_checkpoint_on_disk_path is None


def test_early_stopping_saves_original_module_state_dict(tmp_path: Path) -> None:
    compiled_inner = torch.nn.Linear(2, 2)
    wrapper = type("CompiledModel", (torch.nn.Module,), {})()
    wrapper._orig_mod = compiled_inner
    optimizer = torch.optim.SGD(compiled_inner.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(checkpoint_path))

    early_stopping.save_checkpoint(0.2, wrapper, optimizer, 1, 0.8, 0.7, 0.8, 0.9)
    saved = torch.load(checkpoint_path, map_location="cpu")

    assert saved["is_compiled"] is True
    assert saved["model_state_dict"].keys() == compiled_inner.state_dict().keys()


def test_load_checkpoint_for_resume_handles_empty_or_missing_paths(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    assert (
        load_checkpoint_for_resume(model, optimizer, early_stopping, "", torch.device("cpu")) == 0
    )
    assert (
        load_checkpoint_for_resume(
            model,
            optimizer,
            early_stopping,
            str(tmp_path / "missing.pth"),
            torch.device("cpu"),
        )
        == 0
    )


def test_load_checkpoint_for_resume_handles_missing_model_state_dict(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.pth"
    torch.save({"epoch": 2}, checkpoint_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    start_epoch = load_checkpoint_for_resume(
        model, optimizer, early_stopping, str(checkpoint_path), torch.device("cpu")
    )

    assert start_epoch == 0
    assert early_stopping._current_best_checkpoint_on_disk_path is None


def test_load_checkpoint_for_resume_ignores_optimizer_restore_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"bad": "state"},
            "best_val_score": 0.5,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        optimizer, "load_state_dict", lambda state: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    start_epoch = load_checkpoint_for_resume(
        model, optimizer, early_stopping, str(checkpoint_path), torch.device("cpu")
    )

    assert start_epoch == 3
    assert early_stopping.best_score == 0.5


def test_save_metadata_writes_json_file(tmp_path: Path) -> None:
    save_metadata(
        best_val_score=0.9,
        checkpoint={"epoch": 5},
        encoder="resnet34",
        architecture="UNET++",
        metadata_best_path=str(tmp_path / "best_model.pth"),
        val_loss=0.2,
        val_mcc=0.7,
        val_auroc=0.8,
        metadata_dir=str(tmp_path),
        amp_log={"precision": "fp32"},
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=2,
        seed=7,
        dataset="demo",
        patience=3,
        optimizer_name="AdamW",
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
    )

    meta_path = tmp_path / "best_model_meta.json"
    assert meta_path.exists()
    contents = meta_path.read_text(encoding="utf-8")
    assert '"best_model_epoch": 5' in contents
    assert '"architecture": "UNET++"' in contents
