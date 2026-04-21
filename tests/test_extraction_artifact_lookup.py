from pathlib import Path
from typing import Any

import pytest

from helpers.extraction.artifact_lookup import (
    GeoJsonLookup,
    ProcessingSignatureConfig,
    build_processing_signature,
    resolve_geojson_for_slide,
)


def _signature_config(
    image_path: Path,
    annotation_path: Path,
    *,
    artifacts_geojson_path: Path | None,
    use_advanced_artifact_filtering: bool,
    hiseg_xml_coord_level: int,
) -> ProcessingSignatureConfig:
    return ProcessingSignatureConfig(
        image_path=image_path,
        annotation_path=annotation_path,
        artifacts_geojson_path=artifacts_geojson_path,
        window_size=224,
        stride=112,
        match_percentage=1.0,
        tissue_percentage=0.3,
        target_level=0,
        use_advanced_artifact_filtering=use_advanced_artifact_filtering,
        hiseg_xml_coord_level=hiseg_xml_coord_level,
    )


def test_resolve_geojson_for_slide_accepts_single_collision_safe_match(tmp_path: Path) -> None:
    geojson_dir = tmp_path / "geojson"
    geojson_dir.mkdir()
    candidate = geojson_dir / "case_a__abcdef123456.geojson"
    candidate.write_text("{}")

    resolved = resolve_geojson_for_slide(geojson_dir, Path("/tmp/case_a.svs"))

    assert resolved == candidate


def test_resolve_geojson_for_slide_rejects_ambiguous_collision_safe_matches(tmp_path: Path) -> None:
    geojson_dir = tmp_path / "geojson"
    geojson_dir.mkdir()
    (geojson_dir / "case_a__abcdef123456.geojson").write_text("{}")
    (geojson_dir / "case_a__fedcba654321.geojson").write_text("{}")

    with pytest.raises(ValueError, match="Ambiguous artifact GeoJSON mapping"):
        resolve_geojson_for_slide(geojson_dir, Path("/tmp/case_a.svs"))


def test_resolve_geojson_for_slide_uses_prebuilt_lookup(tmp_path: Path) -> None:
    geojson_dir = tmp_path / "geojson"
    geojson_dir.mkdir()
    candidate = geojson_dir / "case_a__abcdef123456.geojson"
    candidate.write_text("{}")

    lookup = GeoJsonLookup.from_directory(geojson_dir)
    resolved = resolve_geojson_for_slide(
        geojson_dir,
        Path("/tmp/case_a.svs"),
        lookup=lookup,
    )

    assert resolved == candidate


def test_build_processing_signature_changes_when_geojson_metadata_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "case_a.svs"
    annotation_path = tmp_path / "case_a.xml"
    geojson_path = tmp_path / "case_a__abcdef.geojson"
    image_path.write_text("slide")
    annotation_path.write_text("annotation")
    geojson_path.write_text('{"version": 1}')

    signature_before = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=geojson_path,
            use_advanced_artifact_filtering=True,
            hiseg_xml_coord_level=6,
        )
    )

    geojson_path.write_text('{"version": 200}')
    signature_after = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=geojson_path,
            use_advanced_artifact_filtering=True,
            hiseg_xml_coord_level=6,
        )
    )

    assert signature_before != signature_after


def test_build_processing_signature_changes_when_hiseg_coord_level_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "case_a.svs"
    annotation_path = tmp_path / "case_a.xml"
    image_path.write_text("slide")
    annotation_path.write_text("annotation")

    signature_before = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=None,
            use_advanced_artifact_filtering=False,
            hiseg_xml_coord_level=5,
        )
    )
    signature_after = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=None,
            use_advanced_artifact_filtering=False,
            hiseg_xml_coord_level=6,
        )
    )

    assert signature_before != signature_after


def test_build_processing_signature_avoids_opening_raw_input_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "case_a.svs"
    annotation_path = tmp_path / "case_a.xml"
    geojson_path = tmp_path / "case_a__abcdef.geojson"
    image_path.write_text("slide")
    annotation_path.write_text("annotation")
    geojson_path.write_text('{"version": 1}')

    original_open = Path.open

    def fail_if_raw_input_opened(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self in {image_path, annotation_path, geojson_path}:
            raise AssertionError("Raw input files should not be opened for processing signatures.")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_if_raw_input_opened)

    build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=geojson_path,
            use_advanced_artifact_filtering=True,
            hiseg_xml_coord_level=6,
        )
    )


def test_build_processing_signature_changes_when_image_metadata_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "case_a.svs"
    annotation_path = tmp_path / "case_a.xml"
    image_path.write_text("slide")
    annotation_path.write_text("annotation")

    signature_before = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=None,
            use_advanced_artifact_filtering=False,
            hiseg_xml_coord_level=6,
        )
    )

    image_path.write_text("slide-expanded")
    signature_after = build_processing_signature(
        _signature_config(
            image_path,
            annotation_path,
            artifacts_geojson_path=None,
            use_advanced_artifact_filtering=False,
            hiseg_xml_coord_level=6,
        )
    )

    assert signature_before != signature_after
