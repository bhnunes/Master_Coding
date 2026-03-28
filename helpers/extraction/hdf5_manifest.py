from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import h5py


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"version": 1, "shards": []}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Stage 2 shard manifest: {manifest_path}")
    payload.setdefault("version", 1)
    payload.setdefault("shards", [])
    return payload


def _manifest_entry_for_shard(manifest_path: Path, shard_path: Path) -> dict[str, Any]:
    with h5py.File(shard_path, "r") as handle:
        source_signature = handle.attrs.get("source_signature")
        row_count = len(handle["filenames"])
    if source_signature is None:
        raise ValueError(f"Shard '{shard_path}' is missing source_signature.")
    return {
        "relative_path": str(shard_path.relative_to(manifest_path.parent)),
        "row_count": int(row_count),
        "source_signature": str(source_signature),
    }


def update_stage2_shard_manifest(manifest_path: Path, shard_path: Path) -> Path:
    manifest = _read_manifest(manifest_path)
    existing = {
        str(entry["relative_path"]): entry
        for entry in cast(list[dict[str, Any]], manifest.get("shards", []))
        if isinstance(entry, dict) and "relative_path" in entry
    }
    relative_path = str(shard_path.relative_to(manifest_path.parent))
    if shard_path.exists():
        existing[relative_path] = _manifest_entry_for_shard(manifest_path, shard_path)
    else:
        existing.pop(relative_path, None)

    ordered = [existing[key] for key in sorted(existing)]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"version": 1, "shards": ordered}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest_path
