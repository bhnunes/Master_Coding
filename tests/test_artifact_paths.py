from pathlib import Path

from helpers.artifact.paths import cleanup_directory, create_slide_temp_dir, geojson_output_path

DIGEST_LENGTH = 12
EXPECTED_CASE_1_DIGEST = "092479f39fa8"


def test_create_and_cleanup_slide_temp_dir(tmp_path: Path) -> None:
    temp_dir = create_slide_temp_dir(tmp_path, "nested/case_1.svs")
    payload = temp_dir / "artifact.txt"
    payload.write_text("content")

    cleanup_directory(temp_dir)

    assert not temp_dir.exists()


def test_create_slide_temp_dir_creates_missing_temp_root(tmp_path: Path) -> None:
    temp_root = tmp_path / "missing" / "artifact_temp"

    temp_dir = create_slide_temp_dir(temp_root, "nested/case_1.svs")

    assert temp_dir == temp_root / "case_1"
    assert temp_dir.is_dir()


def test_create_slide_temp_dir_sanitizes_spaces_and_cleans_existing_directory(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "artifact_temp"
    existing_dir = temp_root / "case_1"
    existing_dir.mkdir(parents=True)
    stale_file = existing_dir / "stale.txt"
    stale_file.write_text("stale")

    temp_dir = create_slide_temp_dir(temp_root, "nested/case 1.svs")

    assert temp_dir == existing_dir
    assert temp_dir.is_dir()
    assert not stale_file.exists()


def test_geojson_output_path_uses_slide_basename(tmp_path: Path) -> None:
    output_path = geojson_output_path(tmp_path, "nested/case_1.svs")

    assert output_path.parent == tmp_path
    assert output_path.suffix == ".geojson"
    assert output_path.name.startswith("case_1__")


def test_geojson_output_path_avoids_collisions_for_duplicate_basenames(tmp_path: Path) -> None:
    first = geojson_output_path(tmp_path, "nested_a/case_1.svs")
    second = geojson_output_path(tmp_path, "nested_b/case_1.svs")

    assert first != second


def test_geojson_output_path_uses_twelve_character_digest(tmp_path: Path) -> None:
    output_path = geojson_output_path(tmp_path, "nested/case_1.svs")
    digest = output_path.stem.split("__", 1)[1]

    assert len(digest) == DIGEST_LENGTH


def test_cleanup_directory_is_noop_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    cleanup_directory(missing_path)

    assert not missing_path.exists()


def test_geojson_output_path_uses_expected_digest_value(tmp_path: Path) -> None:
    output_path = geojson_output_path(tmp_path, "nested/case_1.svs")

    assert output_path == tmp_path / f"case_1__{EXPECTED_CASE_1_DIGEST}.geojson"
