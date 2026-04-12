from __future__ import annotations

import torch

from helpers.training.losses import BCEDiceHybridLossPaper


def _build_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[4.0, -4.0], [-4.0, 4.0]],
                [[-4.0, 4.0], [4.0, -4.0]],
            ]
        ],
        dtype=torch.float32,
    )


def _build_target() -> torch.Tensor:
    return torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)


def test_bce_dice_hybrid_loss_default_matches_when_ohem_is_disabled() -> None:
    logits = _build_logits()
    target = _build_target()

    baseline = BCEDiceHybridLossPaper(alpha=0.5, beta=0.25, gamma=0.25)
    ohem_capable = BCEDiceHybridLossPaper(
        alpha=0.5,
        beta=0.25,
        gamma=0.25,
        run_ohem=True,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1,
    )

    assert torch.allclose(baseline(logits, target), ohem_capable(logits, target))


def test_bce_dice_hybrid_loss_applies_artifact_discount_after_ohem() -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, 1.0], [0.5, -1.0]],
                [[-2.0, -1.0], [-0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
    artifact_covariates = torch.tensor([[0.2, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)

    loss_fn = BCEDiceHybridLossPaper(
        alpha=0.5,
        beta=0.25,
        gamma=0.25,
        run_ohem=True,
        ohem_start_epoch=0,
        ohem_ratio=0.5,
        ohem_min_kept=1,
    )
    loss_fn.set_epoch(0)
    loss_fn.set_ohem_enabled(True)

    undiscounted = loss_fn(
        logits,
        target,
        artifact_covariates=artifact_covariates,
        apply_artifact_discount=False,
        reduction="none",
    )
    discounted = loss_fn(
        logits,
        target,
        artifact_covariates=artifact_covariates,
        apply_artifact_discount=True,
        reduction="none",
    )

    assert torch.allclose(discounted, undiscounted * 0.8)


def test_bce_dice_hybrid_loss_ohem_changes_loss_once_active() -> None:
    logits = torch.tensor(
        [
            [
                [[4.0, 4.0], [4.0, 4.0]],
                [[-4.0, -4.0], [-4.0, -4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.long)

    loss_fn = BCEDiceHybridLossPaper(
        alpha=0.5,
        beta=0.25,
        gamma=0.25,
        run_ohem=True,
        ohem_start_epoch=0,
        ohem_ratio=0.25,
        ohem_min_kept=1,
    )

    inactive_loss = loss_fn(logits, target)
    loss_fn.set_epoch(0)
    loss_fn.set_ohem_enabled(True)
    active_loss = loss_fn(logits, target)

    assert not torch.allclose(inactive_loss, active_loss)
