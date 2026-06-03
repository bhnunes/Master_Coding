from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _sanitize_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in value
    )
    return safe.strip("-._") or "unknown"


@contextmanager
def stage_wsi_locally(
    *,
    source_path: Path,
    cache_dir: Path | None,
    patient_id: str,
) -> Iterator[Path]:
    """Optionally copy one WSI into a local cache for temporary processing."""

    if cache_dir is None:
        yield source_path
        return

    staged_dir = cache_dir / (
        f"patient_{_sanitize_component(patient_id)}_{_sanitize_component(source_path.stem)}"
    )
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / source_path.name
    shutil.copy2(source_path, staged_path)
    try:
        yield staged_path
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
