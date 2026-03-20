from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceHybridLossPaper(nn.Module):
    """Paper-faithful BCE + Dice hybrid loss used by `10_training_ensemble.py`."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)

    def _dice_loss_khened(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        p = self._flatten(p)
        g = self._flatten(g)

        intersection = (p * g).sum(dim=1)
        denominator = p.pow(2).sum(dim=1) + g.pow(2).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return cast(torch.Tensor, 1.0 - dice.mean())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        elif target.ndim == 4 and target.size(-1) == 1:
            target = target[..., 0]

        assert target.ndim == 3, (
            "[BCEDiceHybridLossPaper] Target tensor invariant violated: "
            f"expected [B,H,W] class-index mask, got shape={tuple(target.shape)} "
            f"dtype={target.dtype}. One-hot encoding must NOT be applied outside the loss."
        )

        target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        if logits.shape != target_one_hot.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape}, target {target_one_hot.shape}"
            )

        probabilities = torch.softmax(logits, dim=1)
        probability_fg = probabilities[:, 1, :, :].clamp(1e-7, 1.0 - 1e-7)
        ground_truth_fg = target_one_hot[:, 1, :, :]

        ce_loss = -(
            ground_truth_fg * torch.log(probability_fg)
            + (1.0 - ground_truth_fg) * torch.log(1.0 - probability_fg)
        )
        ce_loss = ce_loss.mean()

        dice_bg = self._dice_loss_khened(probabilities[:, 0, :, :], target_one_hot[:, 0, :, :])
        dice_fg = self._dice_loss_khened(probabilities[:, 1, :, :], target_one_hot[:, 1, :, :])

        return cast(torch.Tensor, self.alpha * ce_loss + self.beta * dice_bg + self.gamma * dice_fg)
