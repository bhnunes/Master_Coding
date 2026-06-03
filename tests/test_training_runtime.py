import contextlib
import random
from typing import cast

import numpy as np
import pytest
import torch

from helpers.training import runtime as training_runtime
from helpers.training.runtime import (
    _resolve_amp_precision,
    autocast_ctx,
    seed_everything,
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
        "helpers.training.runtime.cv2.setNumThreads", lambda value: cv2_calls.append(value)
    )

    worker_init_fn(worker_id=7)

    expected_seed = 123456789 % (2**32)
    assert np_calls == [expected_seed]
    assert random_calls == [expected_seed]
    assert cv2_calls == [0]


def test_seed_everything_sets_all_rngs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(random, "seed", lambda value: calls.setdefault("random", value))
    monkeypatch.setattr(np.random, "seed", lambda value: calls.setdefault("numpy", value))
    monkeypatch.setattr(torch, "manual_seed", lambda value: calls.setdefault("torch", value))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda value: calls.setdefault("cuda", value))
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda value: calls.setdefault("cuda_all", value)
    )

    seed_everything(123)

    assert calls == {
        "random": 123,
        "numpy": 123,
        "torch": 123,
        "cuda": 123,
        "cuda_all": 123,
    }


def test_autocast_ctx_returns_nullcontext_on_cpu() -> None:
    context_manager = autocast_ctx(torch.zeros(1), torch.float16)

    assert isinstance(context_manager, contextlib.nullcontext)


def test_autocast_ctx_uses_torch_autocast_for_cuda_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    class FakeTensor:
        is_cuda = True

    def fake_autocast(device_type: str, dtype: torch.dtype) -> contextlib.nullcontext[None]:
        calls.append((device_type, dtype))
        return contextlib.nullcontext()

    monkeypatch.setattr(
        torch,
        "autocast",
        fake_autocast,
    )

    context_manager = autocast_ctx(FakeTensor(), torch.float16)  # type: ignore[arg-type]
    assert isinstance(context_manager, contextlib.nullcontext)
    assert calls == [("cuda", torch.float16)]


def test_resolve_amp_precision_supports_fp32_and_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(training_runtime, "GradScaler", lambda *_args, **_kwargs: "scaler")

    fp32_dtype, fp32_scaler, fp32_log = _resolve_amp_precision("fp32", "FPN")
    fp16_dtype, fp16_scaler, fp16_log = _resolve_amp_precision("fp16", "FPN")

    assert fp32_dtype is torch.float32
    assert fp32_scaler is None
    assert fp32_log["amp_reason"] == "user_forced_fp32"
    assert fp16_dtype is torch.float16
    assert cast(object, fp16_scaler) == "scaler"
    assert fp16_log["amp_reason"] == "user_forced_fp16"


def test_resolve_amp_precision_rejects_unknown_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(ValueError, match="Invalid AMP_PRECISION"):
        _resolve_amp_precision("weird", "FPN")
