from pathlib import Path

import pytest

from helpers.extraction.staging import stage_wsi_locally


def test_stage_wsi_locally_copies_slide_to_cache_and_cleans_up(tmp_path: Path) -> None:
    source_path = tmp_path / "external" / "slide.svs"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("slide-bytes")
    cache_dir = tmp_path / "local_cache"

    with stage_wsi_locally(
        source_path=source_path,
        cache_dir=cache_dir,
        patient_id="100001",
    ) as staged_path:
        assert staged_path != source_path
        assert staged_path.exists()
        assert staged_path.read_text() == "slide-bytes"
        assert staged_path.parent.parent == cache_dir

    assert not staged_path.exists()
    assert list(cache_dir.glob("**/*")) == []


def test_stage_wsi_locally_cleans_up_when_processing_fails(tmp_path: Path) -> None:
    source_path = tmp_path / "external" / "slide.svs"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("slide-bytes")
    cache_dir = tmp_path / "local_cache"

    with pytest.raises(RuntimeError, match="boom"):
        with stage_wsi_locally(
            source_path=source_path,
            cache_dir=cache_dir,
            patient_id="100001",
        ) as staged_path:
            assert staged_path.exists()
            raise RuntimeError("boom")

    assert not staged_path.exists()
    assert list(cache_dir.glob("**/*")) == []


def test_stage_wsi_locally_returns_original_path_when_cache_disabled(tmp_path: Path) -> None:
    source_path = tmp_path / "external" / "slide.svs"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("slide-bytes")

    with stage_wsi_locally(
        source_path=source_path,
        cache_dir=None,
        patient_id="100001",
    ) as staged_path:
        assert staged_path == source_path
        assert staged_path.exists()
