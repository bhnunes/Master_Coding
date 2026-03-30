from pathlib import Path

import pytest

from helpers.extraction.config import load_database_manager_config


def test_load_database_manager_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SOURCE_FOLDER": str(tmp_path / "source"),
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
            "WINDOW_SIZE": "256",
            "STRIDE": "128",
            "MATCH_PERCENTAGE": "0.75",
            "TISSUE_PERCENTAGE": "0.4",
            "LOADCASES": "true",
            "USE_ADVANCED_ARTIFACT_FILTERING": "1",
            "ACTIVATE_SANITY_CHECK_GEOJSON": "t",
        }
    )

    assert config.tag == "TCGA"
    assert config.source_folder == tmp_path / "source"
    assert config.database_path == tmp_path / "databases" / "cases.db"
    assert config.base_path == tmp_path / "projects"
    assert config.window_size == 256
    assert config.stride == 128
    assert config.match_percentage == 0.75
    assert config.tissue_percentage == 0.4
    assert config.load_cases is True
    assert config.use_advanced_artifact_filtering is True
    assert config.activate_sanity_check_geojson is True
    assert config.geojson_path == tmp_path / "source" / "GEOJSON"
    assert config.log_path == Path("logs/database_manager.log")


def test_load_database_manager_config_defaults_artifact_feature_to_true(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SOURCE_FOLDER": str(tmp_path / "source"),
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
        }
    )

    assert config.use_advanced_artifact_filtering is True


def test_load_database_manager_config_prefers_global_log_folder(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SOURCE_FOLDER": str(tmp_path / "source"),
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
            "LOG_FOLDER": str(tmp_path / "shared_logs"),
            "EXTRACTION_LOG_FOLDER": str(tmp_path / "legacy_logs"),
            "EXTRACTION_LOG_FILE": "extract.log",
        }
    )

    assert config.log_folder == tmp_path / "shared_logs"
    assert config.log_path == tmp_path / "shared_logs" / "extract.log"


def test_load_database_manager_config_reads_local_slide_cache_settings(tmp_path: Path) -> None:
    config = load_database_manager_config(
        {
            "TAG": "TCGA",
            "SOURCE_FOLDER": str(tmp_path / "source"),
            "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
            "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
            "STAGE2_COPY_WSI_TO_LOCAL_CACHE": "true",
            "STAGE2_LOCAL_SLIDE_CACHE_DIR": str(tmp_path / "local_cache"),
        }
    )

    assert config.copy_wsi_to_local_cache is True
    assert config.local_slide_cache_dir == tmp_path / "local_cache"


def test_load_database_manager_config_requires_cache_dir_when_staging_enabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="STAGE2_LOCAL_SLIDE_CACHE_DIR"):
        load_database_manager_config(
            {
                "TAG": "TCGA",
                "SOURCE_FOLDER": str(tmp_path / "source"),
                "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
                "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
                "STAGE2_COPY_WSI_TO_LOCAL_CACHE": "true",
            }
        )


def test_load_database_manager_config_requires_tag() -> None:
    with pytest.raises(ValueError, match="TAG"):
        load_database_manager_config({})


def test_load_database_manager_config_requires_source_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SOURCE_FOLDER"):
        load_database_manager_config(
            {
                "TAG": "TCGA",
                "SQLITE_DB_PATH": str(tmp_path / "databases" / "cases.db"),
                "PROJECTS_BASE_PATH": str(tmp_path / "projects"),
            }
        )
