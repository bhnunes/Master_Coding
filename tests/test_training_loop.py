from __future__ import annotations

from typing import cast

import torch

from helpers.training_loop import train_epoch, validate_epoch
from helpers.training_metrics import TrainingHealthTracker


class IdentityModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TinySegmentationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(x))


def _loss_fn(outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    targets = torch.nn.functional.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float()
    return torch.nn.functional.mse_loss(outputs, targets)


def test_train_epoch_returns_loss_and_amp_log() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="train")
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randint(0, 2, (2, 4, 4), dtype=torch.long)
    dataloader = [(images, masks)]

    baseline = model.conv.weight.detach().clone()
    loss, amp_log = train_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=torch.device("cpu"),
        current_epoch=0,
        loss_fn=_loss_fn,
        health=health,
        architecture="UNET++",
        accumulation_steps=1,
        amp_precision="fp16",
        gpu_normalizer=IdentityModule(),
        gpu_downscale=IdentityModule(),
    )

    assert loss > 0.0
    assert isinstance(amp_log, dict)
    assert amp_log["architecture"] == "UNET++"
    assert not torch.equal(baseline, model.conv.weight.detach())
    assert health.epoch["train_skipped_batches"] == 0


def test_validate_epoch_returns_metrics() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="val")
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randint(0, 2, (2, 4, 4), dtype=torch.long)
    dataloader = [(images, masks)]

    val_loss, results = validate_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=torch.device("cpu"),
        loss_fn=_loss_fn,
        health=health,
        architecture="UNET++",
        amp_precision="fp16",
        gpu_normalizer=IdentityModule(),
    )

    assert val_loss >= 0.0
    assert results is not None
    assert set(results) == {"val_auprc", "val_auroc", "val_mcc_star", "fg_prevalence_at_05"}


def test_validate_epoch_marks_skipped_batches_when_loader_yields_none() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="val")

    val_loss, results = validate_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=[None],
        device=torch.device("cpu"),
        loss_fn=_loss_fn,
        health=health,
        architecture="UNET++",
        amp_precision="fp16",
        gpu_normalizer=IdentityModule(),
    )

    assert val_loss == 0.0
    assert results is None
    assert health.run["val_skipped_batches"] == 2
    assert health.run["val_skip_reasons"]["dataloader_none_batch"] == 1
    assert health.run["val_skip_reasons"]["no_samples_processed"] == 1
