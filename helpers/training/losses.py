from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceHybridLossPaper(nn.Module):
    """Paper-faithful BCE + Dice hybrid loss used by `10_training_ensemble.py`.

    Default behavior matches the current Stage 9 training loss: compute BCE,
    background Dice, and foreground Dice over all pixels in each sample, then
    optionally apply the artifact-aware sample discount.

    When ``run_ohem`` is enabled, the same loss can switch to pixel-level Online
    Hard Example Mining during training. In OHEM mode the loss ranks pixels by
    their BCE error inside each patch, keeps only the hardest pixels, and
    computes BCE + Dice on that mined support. Validation keeps OHEM disabled so
    loss and metrics remain comparable to the non-OHEM baseline.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        smooth: float = 1e-6,
        run_ohem: bool = False,
        ohem_start_epoch: int = 2,
        ohem_ratio: float = 0.25,
        ohem_min_kept: int = 1024,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.run_ohem = run_ohem
        self.ohem_start_epoch = ohem_start_epoch
        self.ohem_ratio = ohem_ratio
        self.ohem_min_kept = ohem_min_kept
        self._epoch: int | None = None
        self._ohem_enabled = False

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)

    def set_epoch(self, epoch: int | None) -> None:
        self._epoch = epoch

    def set_ohem_enabled(self, enabled: bool) -> None:
        self._ohem_enabled = enabled

    def _ohem_active(self) -> bool:
        """Return whether OHEM should be active for the current forward pass."""

        return (
            self.run_ohem
            and self._ohem_enabled
            and self._epoch is not None
            and self._epoch >= self.ohem_start_epoch
        )

    def _dice_loss_per_sample(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        p = self._flatten(p)
        g = self._flatten(g)

        intersection = (p * g).sum(dim=1)
        denominator = p.pow(2).sum(dim=1) + g.pow(2).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice

    def _masked_dice_loss_per_sample(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        p_flat = self._flatten(p)
        g_flat = self._flatten(g)
        mask_flat = mask.to(dtype=p_flat.dtype)

        intersection = (p_flat * g_flat * mask_flat).sum(dim=1)
        denominator = (p_flat.pow(2) * mask_flat).sum(dim=1) + (g_flat.pow(2) * mask_flat).sum(
            dim=1
        )
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice

    def _build_hard_pixel_mask(self, ce_loss_map: torch.Tensor) -> torch.Tensor:
        """Keep the top-loss pixels per sample using ratio plus a safety floor."""

        ce_flat = self._flatten(ce_loss_map)
        batch_size, num_pixels = ce_flat.shape
        keep_count = int(round(self.ohem_ratio * num_pixels))
        keep_count = min(max(keep_count, self.ohem_min_kept, 1), num_pixels)
        _, hard_indices = torch.topk(ce_flat, k=keep_count, dim=1, largest=True, sorted=False)
        hard_mask = torch.zeros_like(ce_flat, dtype=torch.bool)
        hard_mask.scatter_(1, hard_indices, True)
        return hard_mask

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        artifact_covariates: torch.Tensor | None = None,
        apply_artifact_discount: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
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
        if self._ohem_active():
            # OHEM keeps the full patch forward pass but narrows the backward
            # signal to the hardest pixels inside each sample.
            hard_mask = self._build_hard_pixel_mask(ce_loss)
            hard_counts = hard_mask.sum(dim=1).clamp_min(1).to(dtype=ce_loss.dtype)
            ce_loss = (self._flatten(ce_loss) * hard_mask.to(dtype=ce_loss.dtype)).sum(
                dim=1
            ) / hard_counts
            dice_bg = self._masked_dice_loss_per_sample(
                probabilities[:, 0, :, :],
                target_one_hot[:, 0, :, :],
                hard_mask,
            )
            dice_fg = self._masked_dice_loss_per_sample(
                probabilities[:, 1, :, :],
                target_one_hot[:, 1, :, :],
                hard_mask,
            )
        else:
            ce_loss = ce_loss.mean(dim=(1, 2))
            dice_bg = self._dice_loss_per_sample(
                probabilities[:, 0, :, :], target_one_hot[:, 0, :, :]
            )
            dice_fg = self._dice_loss_per_sample(
                probabilities[:, 1, :, :], target_one_hot[:, 1, :, :]
            )

        loss_per_sample = self.alpha * ce_loss + self.beta * dice_bg + self.gamma * dice_fg

        if artifact_covariates is not None and apply_artifact_discount:
            # Artifact awareness remains sample-level and independent from OHEM:
            # OHEM chooses which pixels matter inside a sample, then the artifact
            # covariates decide how much that sample should count in the batch.
            alpha_eff = artifact_covariates.max(dim=1).values
            loss_per_sample = loss_per_sample * (1.0 - alpha_eff)

        if reduction == "none":
            return loss_per_sample
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss_per_sample.mean()
