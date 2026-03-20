from __future__ import annotations

import torch

from helpers.training.gpu import GPUDownscale, GPUNormalizer


def test_gpu_normalizer_normalizes_uint8_input() -> None:
    module = GPUNormalizer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=torch.device("cpu"))
    image = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)

    result = module(image)

    assert torch.allclose(result, torch.ones_like(result))


def test_gpu_normalizer_scales_large_float_input() -> None:
    module = GPUNormalizer(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], device=torch.device("cpu"))
    image = torch.full((1, 3, 2, 2), 128.0, dtype=torch.float32)

    result = module(image)

    assert torch.allclose(result, torch.full_like(result, 128.0 / 255.0))


def test_gpu_downscale_preserves_shape_when_applied() -> None:
    module = GPUDownscale(
        p=1.0, scale_range=(0.5, 0.5), random_fn=lambda: 0.0, uniform_fn=lambda a, b: 0.5
    )
    image = torch.randn(2, 3, 8, 8)

    result = module(image)

    assert result.shape == image.shape


def test_gpu_downscale_returns_input_when_probability_not_met() -> None:
    module = GPUDownscale(p=0.1, random_fn=lambda: 1.0, uniform_fn=lambda a, b: 0.5)
    image = torch.randn(2, 3, 8, 8)

    result = module(image)

    assert torch.equal(result, image)
