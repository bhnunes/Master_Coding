from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, if available."""

    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return result.strip() or None
    except Exception:
        return None


def collect_runtime_environment() -> dict[str, Any]:
    """Collect a compact runtime snapshot for experiment provenance."""

    versions: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": get_git_commit_hash(),
    }
    try:
        import numpy as np

        versions["numpy"] = getattr(np, "__version__", None)
    except Exception:
        versions["numpy"] = None
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", None)
    except Exception:
        versions["torch"] = None
    try:
        import pandas as pd

        versions["pandas"] = getattr(pd, "__version__", None)
    except Exception:
        versions["pandas"] = None
    return versions
