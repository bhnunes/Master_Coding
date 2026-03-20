from __future__ import annotations

import re
from typing import Any

import torch

from helpers.training_utils import clear_gpu, get_formatted_datetime_string


def test_get_formatted_datetime_string_matches_expected_pattern() -> None:
    value = get_formatted_datetime_string()

    assert re.fullmatch(r"\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}", value)


def test_clear_gpu_is_safe_without_cuda(monkeypatch: Any) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    clear_gpu()


def test_clear_gpu_ignores_cuda_cleanup_failures(monkeypatch: Any) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)

    clear_gpu()
