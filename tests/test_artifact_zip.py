import zipfile
from pathlib import Path

from helpers.artifact.zip import SUPPORTED_WSI_EXTENSIONS, ZipSlideSource


def test_zip_slide_source_lists_supported_wsi_members_only(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("case_1.svs", b"a")
        archive.writestr("nested/case_2.ndpi", b"b")
        archive.writestr("notes.txt", b"ignore")

    source = ZipSlideSource(zip_path)

    assert source.list_slide_members() == ["case_1.svs", "nested/case_2.ndpi"]
    assert ".svs" in SUPPORTED_WSI_EXTENSIONS


def test_zip_slide_source_extracts_single_slide_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/case_1.tif", b"slide-bytes")

    source = ZipSlideSource(zip_path)
    destination = tmp_path / "extract"
    destination.mkdir()

    extracted_path = source.extract_member("nested/case_1.tif", destination)

    assert extracted_path.read_bytes() == b"slide-bytes"
    assert extracted_path.parent == destination
