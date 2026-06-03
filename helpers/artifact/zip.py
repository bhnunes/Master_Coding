from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from helpers.provenance import hash_json_payload

SUPPORTED_WSI_EXTENSIONS = frozenset({".svs", ".ndpi", ".tiff", ".tif"})


class ZipSlideSource:
    """Read slide members directly from a zip archive."""

    def __init__(self, zip_path: Path) -> None:
        self.zip_path = zip_path

    def list_slide_members(self) -> list[str]:
        """Return supported WSI members without extracting the full archive."""

        return [
            member_name for member_name, _signature in self.list_slide_members_with_signatures()
        ]

    def list_slide_members_with_signatures(self) -> list[tuple[str, str]]:
        """Return supported WSI members with content-aware zip signatures."""

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            members = [
                (
                    member.filename,
                    hash_json_payload(
                        {
                            "zip_path": str(self.zip_path),
                            "member": member.filename,
                            "crc": member.CRC,
                            "file_size": member.file_size,
                            "compress_size": member.compress_size,
                            "date_time": member.date_time,
                        }
                    ),
                )
                for member in archive.infolist()
                if not member.is_dir()
                and Path(member.filename).suffix.lower() in SUPPORTED_WSI_EXTENSIONS
            ]
        return sorted(members, key=lambda item: item[0])

    def extract_member(self, member_name: str, destination: Path) -> Path:
        """Extract one slide member into the destination directory."""

        destination.mkdir(parents=True, exist_ok=True)
        extracted_path = destination / Path(member_name).name
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            with archive.open(member_name, "r") as source, extracted_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        return extracted_path
