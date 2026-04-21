from __future__ import annotations

import torch

from helpers.training.loop import (
    TrainEpochConfig,
    TrainEpochRuntime,
    ValidationEpochConfig,
    ValidationEpochRuntime,
    train_epoch,
    validate_epoch,
)
from helpers.training.metrics import TrainingHealthTracker


class _TrackingLoss:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_epoch(self, epoch: int | None) -> None:
        self.calls.append(("epoch", epoch))

    def set_ohem_enabled(self, enabled: bool) -> None:
        self.calls.append(("enabled", enabled))

    def __call__(
        self, outputs: torch.Tensor, masks: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        del masks, kwargs
        return outputs.sum() * 0 + torch.tensor(1.0, device=outputs.device)


class _ConstantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = images.shape
        return torch.zeros((batch_size, 2, height, width), device=images.device) + self.bias


def test_train_epoch_enables_ohem_and_sets_current_epoch() -> None:
    loss_fn = _TrackingLoss()
    model = _ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [
        (
            torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2), dtype=torch.long),
        )
    ]

    train_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        runtime=TrainEpochRuntime(
            loss_fn=loss_fn,
            health=TrainingHealthTracker(name="train"),
            gpu_normalizer=torch.nn.Identity(),
            gpu_downscale=torch.nn.Identity(),
        ),
        config=TrainEpochConfig(
            device=torch.device("cpu"),
            current_epoch=3,
            architecture="FPN",
            accumulation_steps=1,
            amp_precision="fp32",
        ),
    )

    assert loss_fn.calls[:2] == [("epoch", 3), ("enabled", True)]


def test_validate_epoch_disables_ohem_and_clears_epoch() -> None:
    loss_fn = _TrackingLoss()
    model = _ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [
        (
            torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2), dtype=torch.long),
        )
    ]

    validate_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        runtime=ValidationEpochRuntime(
            loss_fn=loss_fn,
            health=TrainingHealthTracker(name="val"),
            gpu_normalizer=torch.nn.Identity(),
        ),
        config=ValidationEpochConfig(
            device=torch.device("cpu"),
            architecture="FPN",
            amp_precision="fp32",
        ),
    )

    assert loss_fn.calls[:2] == [("enabled", False), ("epoch", None)]
