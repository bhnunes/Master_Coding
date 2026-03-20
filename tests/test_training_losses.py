import pytest
import torch

from helpers.training.losses import BCEDiceHybridLossPaper


def test_bce_dice_hybrid_loss_accepts_class_index_masks() -> None:
    loss_fn = BCEDiceHybridLossPaper(alpha=0.125, beta=0.157, gamma=0.173)
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0], [0.5, 1.0]],
                [[-2.0, 1.0], [-0.5, -1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    loss = loss_fn(logits, target)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_bce_dice_hybrid_loss_accepts_single_channel_masks() -> None:
    loss_fn = BCEDiceHybridLossPaper()
    logits = torch.randn(2, 2, 4, 4, dtype=torch.float32)
    target = torch.randint(0, 2, (2, 1, 4, 4), dtype=torch.long)

    loss = loss_fn(logits, target)

    assert torch.isfinite(loss)


def test_bce_dice_hybrid_loss_rejects_shape_mismatch() -> None:
    loss_fn = BCEDiceHybridLossPaper()
    logits = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    target = torch.randint(0, 2, (1, 3, 4), dtype=torch.long)

    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(logits, target)


def test_bce_dice_hybrid_loss_rejects_non_canonical_target_rank() -> None:
    loss_fn = BCEDiceHybridLossPaper()
    logits = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    target = torch.randint(0, 2, (1, 2, 4, 4), dtype=torch.long)

    with pytest.raises(AssertionError, match="Target tensor invariant"):
        loss_fn(logits, target)


def test_bce_dice_hybrid_loss_applies_artifact_discount_using_max_coverage() -> None:
    loss_fn = BCEDiceHybridLossPaper()
    logits = torch.tensor(
        [
            [[[2.0, -1.0], [0.5, 1.0]], [[-2.0, 1.0], [-0.5, -1.0]]],
            [[[1.0, -0.5], [0.1, 0.2]], [[-1.0, 0.5], [-0.1, -0.2]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=torch.long,
    )
    artifact_covariates = torch.tensor(
        [[0.2, 0.1, 0.4, 0.0, 0.3], [0.05, 0.0, 0.1, 0.0, 0.0]],
        dtype=torch.float32,
    )

    base_losses = loss_fn(logits, target, reduction="none")
    weighted_loss = loss_fn(
        logits,
        target,
        artifact_covariates=artifact_covariates,
        apply_artifact_discount=True,
    )

    expected = (base_losses * torch.tensor([0.6, 0.9])).mean()
    assert torch.isclose(weighted_loss, expected)
