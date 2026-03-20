import random

import numpy as np
import pytest
import torch

from helpers.training_runtime import (
    _resolve_amp_precision,
    setup_precision,
    worker_init_fn,
)


def test_setup_precision_returns_float32_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    amp_dtype, scaler, log = setup_precision("FPN", amp_precision="auto")

    assert amp_dtype is torch.float32
    assert scaler is None
    assert log["amp_dtype_effective"] == "float32"
    assert log["amp_reason"] == "cpu_no_cuda"
    assert log["matmul_precision"] == "high"


def test_setup_precision_prefers_bf16_for_supported_segformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    recorded: list[str] = []

    def fake_set_float32_matmul_precision(value: str) -> None:
        recorded.append(value)

    monkeypatch.setattr(torch, "set_float32_matmul_precision", fake_set_float32_matmul_precision)

    amp_dtype, scaler, log = setup_precision("SEGFORMER", amp_precision="auto")

    assert amp_dtype is torch.bfloat16
    assert scaler is None
    assert log["amp_dtype_effective"] == "bfloat16"
    assert log["amp_reason"] == "auto_prefer_bf16_supported"
    assert log["matmul_precision"] == "medium"
    assert recorded == ["medium"]


def test_resolve_amp_precision_rejects_unsupported_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="BF16"):
        _resolve_amp_precision("bf16", "FPN")


def test_setup_precision_for_dpt_forces_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    amp_dtype, scaler, log = setup_precision("DPT", amp_precision="auto")

    assert amp_dtype is torch.float32
    assert scaler is None
    assert log["amp_dtype_effective"] == "float32"
    assert log["amp_reason"] == "arch_forced_fp32"


def test_worker_init_fn_seeds_numpy_random_and_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    np_calls: list[int] = []
    random_calls: list[int] = []
    cv2_calls: list[int] = []

    monkeypatch.setattr(torch, "initial_seed", lambda: 123456789)
    monkeypatch.setattr(np.random, "seed", lambda value: np_calls.append(value))
    monkeypatch.setattr(random, "seed", lambda value: random_calls.append(value))
    monkeypatch.setattr(
        "helpers.training_runtime.cv2.setNumThreads", lambda value: cv2_calls.append(value)
    )

    worker_init_fn(worker_id=7)

    expected_seed = 123456789 % (2**32)
    assert np_calls == [expected_seed]
    assert random_calls == [expected_seed]
    assert cv2_calls == [0]
