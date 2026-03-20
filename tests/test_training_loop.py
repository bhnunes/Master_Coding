from __future__ import annotations

from typing import cast

import torch

from helpers.training.loop import train_epoch, validate_epoch
from helpers.training.metrics import TrainingHealthTracker


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


class ArtifactAwareLoss:
    def __init__(self) -> None:
        self.last_covariates: torch.Tensor | None = None

    def __call__(
        self,
        outputs: torch.Tensor,
        masks: torch.Tensor,
        *,
        artifact_covariates: torch.Tensor,
    ) -> torch.Tensor:
        self.last_covariates = artifact_covariates.detach().clone()
        targets = torch.nn.functional.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float()
        return torch.nn.functional.mse_loss(outputs, targets)


class BadOutputModel(torch.nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> object:
        del x
        return self.output


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


def test_train_epoch_passes_artifact_covariates_when_enabled() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="train")
    loss_fn = ArtifactAwareLoss()
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randint(0, 2, (2, 4, 4), dtype=torch.long)
    artifact_covariates = torch.tensor([[0.1, 0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.3, 0.0]])

    train_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=[(images, masks, artifact_covariates)],
        device=torch.device("cpu"),
        current_epoch=0,
        loss_fn=loss_fn,
        health=health,
        architecture="UNET++",
        accumulation_steps=1,
        amp_precision="fp16",
        gpu_normalizer=IdentityModule(),
        gpu_downscale=IdentityModule(),
        use_artifact_aware_loss=True,
    )

    assert loss_fn.last_covariates is not None
    assert torch.allclose(loss_fn.last_covariates, artifact_covariates)


def test_train_epoch_tracks_skip_reasons_for_invalid_batches() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="train")
    valid_images = torch.randn(1, 3, 4, 4)
    valid_masks = torch.randint(0, 2, (1, 4, 4), dtype=torch.long)
    bad_shape_masks = torch.randint(0, 2, (1, 1, 1, 4, 4), dtype=torch.long)
    dataloader = [
        None,
        object(),
        (None, valid_masks),
        (torch.randn(0, 3, 4, 4), valid_masks),
        (valid_images, bad_shape_masks),
    ]

    loss, _ = train_epoch(
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

    assert loss == 0.0
    assert health.run["train_skip_reasons"]["dataloader_none_batch"] == 1
    assert health.run["train_skip_reasons"]["unpack_failed"] == 1
    assert health.run["train_skip_reasons"]["images_or_masks_none"] == 1
    assert health.run["train_skip_reasons"]["batch_size_zero"] == 1
    assert health.run["train_skip_reasons"]["mask_bad_shape"] == 1


def test_train_epoch_skips_non_tensor_and_mismatched_outputs() -> None:
    health = TrainingHealthTracker(name="train")
    images = torch.randn(1, 3, 4, 4)
    masks = torch.randint(0, 2, (1, 4, 4), dtype=torch.long)
    model = BadOutputModel(output="bad")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    loss, _ = train_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=[(images, masks)],
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

    assert loss == 0.0
    assert health.run["train_skip_reasons"]["model_output_not_tensor"] == 1

    health = TrainingHealthTracker(name="train")
    model = BadOutputModel(output=torch.randn(1, 1, 4, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss, _ = train_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=[(images, masks)],
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

    assert loss == 0.0
    assert health.run["train_skip_reasons"]["channel_mismatch"] == 1


def test_validate_epoch_tracks_common_skip_reasons() -> None:
    model = TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    health = TrainingHealthTracker(name="val")
    valid_masks = torch.randint(0, 2, (1, 4, 4), dtype=torch.long)
    bad_shape_masks = torch.randint(0, 2, (1, 1, 1, 4, 4), dtype=torch.long)
    dataloader = [
        object(),
        (None, valid_masks),
        (torch.randn(0, 3, 4, 4), valid_masks),
        (torch.randn(1, 3, 4, 4), bad_shape_masks),
    ]

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

    assert val_loss == 0.0
    assert results is None
    assert health.run["val_skip_reasons"]["unpack_failed"] == 1
    assert health.run["val_skip_reasons"]["images_or_masks_none"] == 1
    assert health.run["val_skip_reasons"]["batch_size_zero"] == 1
    assert health.run["val_skip_reasons"]["mask_bad_shape"] == 1
