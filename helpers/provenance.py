from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
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


def hash_json_payload(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 digest for a JSON-serializable payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def hash_file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a file."""

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_metadata_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a cheap content-change fingerprint based on filesystem metadata."""

    resolved = Path(path)
    stats = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "ctime_ns": int(stats.st_ctime_ns),
    }


def collect_hdf5_provenance(path: str | Path) -> dict[str, Any]:
    """Collect content-aware provenance for an HDF5 artifact."""

    resolved = Path(path)
    payload: dict[str, Any] = {
        "path": str(resolved),
        "sha256": hash_file_sha256(resolved),
        "source_signature": None,
        "selection_signature": None,
        "attrs": {},
    }
    try:
        import h5py

        with h5py.File(resolved, "r") as handle:
            source_signature = handle.attrs.get("source_signature")
            selection_signature = handle.attrs.get("selection_signature")
            attrs = {
                str(key): _normalize_hdf5_attr_value(value) for key, value in handle.attrs.items()
            }
    except Exception:
        source_signature = None
        selection_signature = None
        attrs = {}

    payload["source_signature"] = (
        source_signature.decode("utf-8")
        if isinstance(source_signature, bytes)
        else source_signature
    )
    payload["selection_signature"] = (
        selection_signature.decode("utf-8")
        if isinstance(selection_signature, bytes)
        else selection_signature
    )
    payload["attrs"] = attrs
    return payload


def _normalize_hdf5_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_split_fingerprint(
    *,
    optimization_patients: set[str],
    calibration_patients: set[str],
    holdout_patients: set[str],
) -> str:
    """Return a deterministic fingerprint for inference patient subsets."""

    return hash_json_payload(
        {
            "optimization_patients": sorted(optimization_patients),
            "calibration_patients": sorted(calibration_patients),
            "holdout_patients": sorted(holdout_patients),
        }
    )
