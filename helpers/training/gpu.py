from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional

_IMAGE_BYTE_SCALE = 255.0
_FLOAT_INPUT_UPPER_BOUND = 2.0


class GPUNormalizer(nn.Module):
    """Normalize batches on-device with safe uint8 and float handling."""

    def __init__(self, mean: Sequence[float], std: Sequence[float], device: torch.device) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        if x.dtype == torch.uint8:
            x = x.float() / _IMAGE_BYTE_SCALE
        elif x.max() > _FLOAT_INPUT_UPPER_BOUND:
            x = x / _IMAGE_BYTE_SCALE
        return (x - mean) / std


class GPUDownscale(nn.Module):
    """Apply stochastic batch downscale/upscale augmentation on GPU."""

    def __init__(
        self,
        p: float = 0.07,
        scale_range: tuple[float, float] = (0.5, 0.9),
        random_fn: Callable[[], float] = random.random,
        uniform_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        super().__init__()
        self.p = p
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
        self.random_fn = random_fn
        self.uniform_fn = uniform_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.random_fn() > self.p:
            return x

        _, _, height, width = x.shape
        scale = self.uniform_fn(self.scale_min, self.scale_max)
        small_height = int(height * scale)
        small_width = int(width * scale)
        x_small = functional.interpolate(x, size=(small_height, small_width), mode="nearest")
        return functional.interpolate(
            x_small,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
