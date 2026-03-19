from __future__ import annotations

import pytest

from helpers.artifact_config import load_artifact_detection_config
from helpers.extraction_config import load_database_manager_config


def test_load_database_manager_config_rejects_windows_path_on_linux() -> None:
    with pytest.raises(ValueError, match="Windows path"):
        load_database_manager_config(
            {
                "TAG": "TCGA",
                "SQLITE_DB_PATH": r"D:\data\database.db",
                "PROJECTS_BASE_PATH": r"D:\data\projects",
            },
            system_name="Linux",
        )


def test_load_artifact_detection_config_rejects_windows_path_on_linux() -> None:
    with pytest.raises(ValueError, match="Windows path"):
        load_artifact_detection_config(
            {
                "ARTIFACT_IMAGES_ZIP": r"D:\slides\archive.zip",
            },
            system_name="Linux",
        )
