from pathlib import Path

import torch

from helpers.training_checkpointing import (
    EarlyStopping,
    get_previous_metrics,
    load_checkpoint_for_resume,
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
