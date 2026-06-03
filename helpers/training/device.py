from __future__ import annotations

import torch


def require_cuda_device() -> torch.device:
    """Return the default CUDA device or fail fast when GPU execution is unavailable."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this pipeline stage, but torch.cuda is unavailable."
        )
    return torch.device("cuda")
