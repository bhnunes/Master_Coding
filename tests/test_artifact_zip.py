import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from helpers.artifact.zip import SUPPORTED_WSI_EXTENSIONS, ZipSlideSource
from helpers.provenance import hash_json_payload


def test_zip_slide_source_lists_supported_wsi_members_only(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("case_1.svs", b"a")
        archive.writestr("nested/case_2.ndpi", b"b")
        archive.writestr("notes.txt", b"ignore")

    source = ZipSlideSource(zip_path)

    assert source.list_slide_members() == ["case_1.svs", "nested/case_2.ndpi"]
    assert ".svs" in SUPPORTED_WSI_EXTENSIONS


def test_zip_slide_source_list_slide_members_delegates_to_signature_listing(tmp_path: Path) -> None:
    source = ZipSlideSource(tmp_path / "slides.zip")

    with patch.object(
        source,
        "list_slide_members_with_signatures",
        return_value=[("b.ndpi", "sig-b"), ("a.svs", "sig-a")],
    ) as list_with_signatures:
        members = source.list_slide_members()

    list_with_signatures.assert_called_once_with()
    assert members == ["b.ndpi", "a.svs"]


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


def test_zip_slide_source_extracts_into_missing_nested_destination(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/case_1.tif", b"slide-bytes")

    source = ZipSlideSource(zip_path)
    destination = tmp_path / "nested" / "extract"

    extracted_path = source.extract_member("nested/case_1.tif", destination)

    assert extracted_path.read_bytes() == b"slide-bytes"
    assert extracted_path.parent == destination


def test_zip_slide_source_lists_members_with_expected_signatures(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("case_1.svs", b"a")
        archive.writestr("nested/case_2.ndpi", b"bb")

    source = ZipSlideSource(zip_path)
    members_with_signatures = source.list_slide_members_with_signatures()

    with zipfile.ZipFile(zip_path, "r") as archive:
        expected_members = []
        for member in archive.infolist():
            if member.is_dir() or (
                Path(member.filename).suffix.lower() not in SUPPORTED_WSI_EXTENSIONS
            ):
                continue
            expected_members.append(
                (
                    member.filename,
                    hash_json_payload(
                        {
                            "zip_path": str(zip_path),
                            "member": member.filename,
                            "crc": member.CRC,
                            "file_size": member.file_size,
                            "compress_size": member.compress_size,
                            "date_time": member.date_time,
                        }
                    ),
                )
            )

    assert members_with_signatures == sorted(expected_members, key=lambda item: item[0])


def test_zip_slide_source_uses_read_mode_for_archive_and_member_access(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    destination = tmp_path / "extract"
    source = ZipSlideSource(zip_path)

    archive_context = MagicMock()
    archive = archive_context.__enter__.return_value
    member_context = MagicMock()
    member_stream = member_context.__enter__.return_value

    archive.open.return_value = member_context

    with patch(
        "helpers.artifact.zip.zipfile.ZipFile",
        return_value=archive_context,
    ) as zip_file_cls:
        with patch("helpers.artifact.zip.shutil.copyfileobj") as copyfileobj:
            extracted_path = source.extract_member("nested/case_1.tif", destination)

    zip_file_cls.assert_called_once_with(zip_path, "r")
    archive.open.assert_called_once_with("nested/case_1.tif", "r")
    copyfileobj.assert_called_once()
    source_stream, target_stream = copyfileobj.call_args.args
    assert source_stream is member_stream
    assert Path(target_stream.name) == extracted_path
    assert extracted_path == destination / "case_1.tif"


def test_zip_slide_source_uses_read_mode_for_signature_listing(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    source = ZipSlideSource(zip_path)

    archive_context = MagicMock()
    archive = archive_context.__enter__.return_value
    archive.infolist.return_value = []

    with patch(
        "helpers.artifact.zip.zipfile.ZipFile",
        return_value=archive_context,
    ) as zip_file_cls:
        members = source.list_slide_members_with_signatures()

    zip_file_cls.assert_called_once_with(zip_path, "r")
    assert members == []
