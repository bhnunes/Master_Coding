from pathlib import Path

import pytest

from helpers.extraction.artifact_lookup import build_processing_signature, resolve_geojson_for_slide


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


def test_build_processing_signature_changes_when_geojson_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "case_a.svs"
    annotation_path = tmp_path / "case_a.xml"
    geojson_path = tmp_path / "case_a__abcdef.geojson"
    image_path.write_text("slide")
    annotation_path.write_text("annotation")
    geojson_path.write_text('{"version": 1}')

    signature_before = build_processing_signature(
        image_path=image_path,
        annotation_path=annotation_path,
        artifacts_geojson_path=geojson_path,
        window_size=224,
        stride=112,
        match_percentage=1.0,
        tissue_percentage=0.3,
        target_level=0,
        use_advanced_artifact_filtering=True,
    )

    geojson_path.write_text('{"version": 2}')
    signature_after = build_processing_signature(
        image_path=image_path,
        annotation_path=annotation_path,
        artifacts_geojson_path=geojson_path,
        window_size=224,
        stride=112,
        match_percentage=1.0,
        tissue_percentage=0.3,
        target_level=0,
        use_advanced_artifact_filtering=True,
    )

    assert signature_before != signature_after
