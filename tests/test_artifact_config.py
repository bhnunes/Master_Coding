from pathlib import Path

import pytest

from helpers.artifact.config import load_artifact_detection_config


def test_load_artifact_detection_config_reads_stage_one_environment(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    zip_path.write_bytes(b"zip")
    output_dir = tmp_path / "geojson"
    database_dir = tmp_path / "database"
    temp_dir = tmp_path / "temp"
    log_dir = tmp_path / "logs"

    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": str(zip_path),
            "ARTIFACT_GEOJSON_OUTPUT": str(output_dir),
            "ARTIFACT_DATABASE_FOLDER": str(database_dir),
            "ARTIFACT_TEMP_FOLDER": str(temp_dir),
            "ARTIFACT_LOG_FOLDER": str(log_dir),
            "ARTIFACT_DEVICE": "cpu",
        }
    )

    assert config.images_zip == zip_path
    assert config.geojson_output == output_dir
    assert config.database_path == database_dir / "artifact_detection.db"
    assert config.temp_root == temp_dir
    assert config.log_folder == log_dir
    assert config.device == "cpu"


def test_load_artifact_detection_config_requires_images_zip() -> None:
    with pytest.raises(ValueError, match="ARTIFACT_IMAGES_ZIP"):
        load_artifact_detection_config({})
