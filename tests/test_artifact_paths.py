from pathlib import Path

from helpers.artifact.paths import cleanup_directory, create_slide_temp_dir, geojson_output_path


def test_create_and_cleanup_slide_temp_dir(tmp_path: Path) -> None:
    temp_dir = create_slide_temp_dir(tmp_path, "nested/case_1.svs")
    payload = temp_dir / "artifact.txt"
    payload.write_text("content")

    cleanup_directory(temp_dir)

    assert not temp_dir.exists()


def test_geojson_output_path_uses_slide_basename(tmp_path: Path) -> None:
    output_path = geojson_output_path(tmp_path, "nested/case_1.svs")

    assert output_path == tmp_path / "case_1.geojson"
