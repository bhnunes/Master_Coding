from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def create_slide_temp_dir(temp_root: Path, zip_member_name: str) -> Path:
    """Create a clean temporary directory for one slide."""

    safe_name = Path(zip_member_name).stem.replace(" ", "_")
    temp_dir = temp_root / safe_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_directory(path: Path) -> None:
    """Remove a temporary directory if it exists."""

    if path.exists():
        shutil.rmtree(path)


def geojson_output_path(output_root: Path, zip_member_name: str) -> Path:
    """Build the GeoJSON output path for a slide zip member."""

    stem = Path(zip_member_name).stem
    member_digest = hashlib.sha256(zip_member_name.encode("utf-8")).hexdigest()[:12]
    return output_root / f"{stem}__{member_digest}.geojson"
