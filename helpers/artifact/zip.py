from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

SUPPORTED_WSI_EXTENSIONS = frozenset({".svs", ".ndpi", ".tiff", ".tif"})


class ZipSlideSource:
    """Read slide members directly from a zip archive."""

    def __init__(self, zip_path: Path) -> None:
        self.zip_path = zip_path

    def list_slide_members(self) -> list[str]:
        """Return supported WSI members without extracting the full archive."""

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            members = [
                member.filename
                for member in archive.infolist()
                if not member.is_dir()
                and Path(member.filename).suffix.lower() in SUPPORTED_WSI_EXTENSIONS
            ]
        return sorted(members)

    def extract_member(self, member_name: str, destination: Path) -> Path:
        """Extract one slide member into the destination directory."""

        destination.mkdir(parents=True, exist_ok=True)
        extracted_path = destination / Path(member_name).name
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            with archive.open(member_name, "r") as source, extracted_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        return extracted_path
