from pathlib import Path

import pytest

from helpers.extraction_config import load_database_manager_config


def test_load_database_manager_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
            "WINDOW_SIZE": "256",
            "STRIDE": "128",
            "MATCH_PERCENTAGE": "0.75",
            "TISSUE_PERCENTAGE": "0.4",
            "LOADCASES": "true",
            "USE_ADVANCED_ARTIFACT_FILTERING": "1",
            "ACTIVATE_SANITY_CHECK_GEOJSON": "t",
            "GEOJSON_PATH": str(tmp_path / "geojson"),
        }
    )

    assert config.tag == "TCGA"
    assert config.database_path == tmp_path / "databases" / "cases.db"
    assert config.base_path == tmp_path / "projects"
    assert config.window_size == 256
    assert config.stride == 128
    assert config.match_percentage == 0.75
    assert config.tissue_percentage == 0.4
    assert config.load_cases is True
    assert config.use_advanced_artifact_filtering is True
    assert config.activate_sanity_check_geojson is True
    assert config.geojson_path == tmp_path / "geojson"


def test_load_database_manager_config_defaults_artifact_feature_to_true(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
        }
    )

    assert config.use_advanced_artifact_filtering is True


def test_load_database_manager_config_requires_tag() -> None:
    with pytest.raises(ValueError, match="TAG"):
        load_database_manager_config({})
