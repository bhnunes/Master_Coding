from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import torch


def get_formatted_datetime_string() -> str:
    """Return a timestamp matching the existing training artifact naming pattern."""

    return datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


def clear_gpu() -> None:
    """Best-effort CUDA cache cleanup."""

    print("Clearing GPU cache...")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            ipc_collect = cast(Any, torch.cuda.ipc_collect)
            ipc_collect()
        except Exception as error:
            print(f"[clear_gpu] Warning: {error}")
    print("GPU cache cleared.")
