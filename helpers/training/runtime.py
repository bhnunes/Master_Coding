from __future__ import annotations

import contextlib
import os
import random

import cv2
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from helpers.cv2_compat import ensure_cv2_compat

cv2 = ensure_cv2_compat(cv2)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed each dataloader worker and disable nested OpenCV threading."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)


def autocast_ctx(
    x: torch.Tensor, amp_dtype: torch.dtype
) -> contextlib.AbstractContextManager[None]:
    """Return an autocast context only when CUDA mixed precision is active."""

    use_cuda_amp = x.is_cuda and (amp_dtype is not torch.float32)
    if use_cuda_amp:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def _set_matmul_precision_for_arch(architecture: str) -> str:
    """Set float32 matmul precision according to model family."""

    arch = architecture.upper()
    matmul_precision = "medium" if arch in {"SWIN", "SEGFORMER"} else "high"

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass

    return matmul_precision


def _resolve_amp_precision(
    amp_precision: str,
    architecture: str,
    prefer_bf16_if_available: bool = True,
) -> tuple[torch.dtype, GradScaler | None, dict[str, str]]:
    """Resolve effective AMP behavior for the active device and architecture."""

    arch = architecture.upper()
    choice = amp_precision.lower()

    if not torch.cuda.is_available():
        return (
            torch.float32,
            None,
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "float32",
                "amp_reason": "cpu_no_cuda",
            },
        )

    if arch == "DPT":
        return (
            torch.float32,
            None,
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "float32",
                "amp_reason": "arch_forced_fp32",
            },
        )

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    if choice == "fp32":
        return (
            torch.float32,
            None,
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "float32",
                "amp_reason": "user_forced_fp32",
            },
        )

    if choice == "bf16":
        if not bf16_supported:
            raise RuntimeError(
                "AMP_PRECISION='bf16' requested but CUDA BF16 is not supported "
                "on this GPU/torch build."
            )
        return (
            torch.bfloat16,
            None,
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "bfloat16",
                "amp_reason": "user_forced_bf16",
            },
        )

    if choice == "fp16":
        return (
            torch.float16,
            GradScaler("cuda"),
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "float16",
                "amp_reason": "user_forced_fp16",
            },
        )

    if choice == "auto":
        if prefer_bf16_if_available and bf16_supported:
            return (
                torch.bfloat16,
                None,
                {
                    "amp_precision_requested": choice,
                    "amp_dtype_effective": "bfloat16",
                    "amp_reason": "auto_prefer_bf16_supported",
                },
            )
        return (
            torch.float16,
            GradScaler("cuda"),
            {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "float16",
                "amp_reason": "auto_fallback_fp16",
            },
        )

    raise ValueError(
        f"Invalid AMP_PRECISION='{amp_precision}'. Use one of: fp32, bf16, fp16, auto."
    )


def setup_precision(
    architecture: str,
    amp_precision: str = "auto",
) -> tuple[torch.dtype, GradScaler | None, dict[str, str]]:
    """Set matmul precision policy and resolve effective AMP settings."""

    matmul_precision = _set_matmul_precision_for_arch(architecture)
    amp_dtype, scaler, amp_log = _resolve_amp_precision(amp_precision, architecture)
    log = {
        "architecture": architecture.upper(),
        "matmul_precision": matmul_precision,
        **amp_log,
    }
    return amp_dtype, scaler, log
