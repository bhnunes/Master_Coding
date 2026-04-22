from pathlib import Path
from unittest.mock import patch

import pytest

from helpers.artifact.config import (
    _float_with_default,
    _int_with_default,
    _path_with_default,
    _required_path,
    _string_with_default,
    load_artifact_detection_config,
)

DEFAULT_MPP_MODEL = 1.5
DEFAULT_OVERLAY_FACTOR = 10
CUSTOM_MPP_MODEL = 2.0
CUSTOM_OVERLAY_FACTOR = 12
DEFAULT_DEVICE = "cuda"


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
    assert config.log_file_name == "artifact_detection.log"
    assert config.log_path == log_dir / "artifact_detection.log"
    assert config.device == "cpu"


def test_load_artifact_detection_config_prefers_global_log_folder(tmp_path: Path) -> None:
    zip_path = tmp_path / "slides.zip"
    zip_path.write_bytes(b"zip")
    shared_logs = tmp_path / "shared_logs"

    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": str(zip_path),
            "LOG_FOLDER": str(shared_logs),
            "ARTIFACT_LOG_FOLDER": str(tmp_path / "legacy_logs"),
        }
    )

    assert config.log_folder == shared_logs


def test_load_artifact_detection_config_populates_default_model_and_runtime_settings(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "slides.zip"
    zip_path.write_bytes(b"zip")

    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": str(zip_path),
        }
    )

    assert config.device == "cuda"
    assert config.tissue_detector_model_dir == Path("models/td")
    assert config.tissue_detector_model_name == "Tissue_Detection_MPP10.pth"
    assert config.qc_model_dir == Path("models/qc")
    assert config.mpp_model == DEFAULT_MPP_MODEL
    assert config.overlay_factor == DEFAULT_OVERLAY_FACTOR
    assert config.geojson_output == Path("artifacts/geojson")
    assert config.database_folder == Path("databases")
    assert config.database_path == Path("databases/artifact_detection.db")
    assert config.temp_root == Path("temp/artifact_detection")
    assert config.log_file_name == "artifact_detection.log"
    assert config.log_path == Path("logs") / "artifact_detection.log"


def test_load_artifact_detection_config_parses_custom_numeric_and_file_settings(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "slides.zip"
    zip_path.write_bytes(b"zip")

    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": str(zip_path),
            "ARTIFACT_DATABASE_NAME": "custom_artifact.db",
            "ARTIFACT_LOG_FILE": "custom_artifact.log",
            "ARTIFACT_MPP_MODEL": "2.0",
            "ARTIFACT_OVERLAY_FACTOR": "12",
        }
    )

    assert config.mpp_model == CUSTOM_MPP_MODEL
    assert config.qc_model_name == "GrandQC_MPP2.pth"
    assert config.overlay_factor == CUSTOM_OVERLAY_FACTOR
    assert config.log_file_name == "custom_artifact.log"
    assert config.database_path == Path("databases/custom_artifact.db")


def test_load_artifact_detection_config_reads_custom_tissue_detector_model_name(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "slides.zip"
    zip_path.write_bytes(b"zip")

    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": str(zip_path),
            "ARTIFACT_TD_MODEL_NAME": "custom_td_model.pth",
        }
    )

    assert config.tissue_detector_model_name == "custom_td_model.pth"


def test_load_artifact_detection_config_propagates_system_name_to_all_path_fields() -> None:
    config = load_artifact_detection_config(
        {
            "ARTIFACT_IMAGES_ZIP": r"C:\slides\slides.zip",
            "ARTIFACT_GEOJSON_OUTPUT": r"C:\artifacts\geojson",
            "ARTIFACT_DATABASE_FOLDER": r"C:\databases",
            "ARTIFACT_TEMP_FOLDER": r"C:\temp\artifact_detection",
            "LOG_FOLDER": r"C:\logs",
            "ARTIFACT_TD_MODEL_DIR": r"C:\models\td",
            "ARTIFACT_QC_MODEL_DIR": r"C:\models\qc",
        },
        system_name="Windows",
    )

    assert config.images_zip == Path("C:/slides/slides.zip")
    assert config.geojson_output == Path("C:/artifacts/geojson")
    assert config.database_folder == Path("C:/databases")
    assert config.temp_root == Path("C:/temp/artifact_detection")
    assert config.log_folder == Path("C:/logs")
    assert config.tissue_detector_model_dir == Path("C:/models/td")
    assert config.qc_model_dir == Path("C:/models/qc")


def test_load_artifact_detection_config_requires_images_zip() -> None:
    with pytest.raises(ValueError, match="ARTIFACT_IMAGES_ZIP"):
        load_artifact_detection_config({})


def test_load_artifact_detection_config_keeps_path_variable_name_in_validation_errors() -> None:
    with pytest.raises(ValueError, match="ARTIFACT_GEOJSON_OUTPUT"):
        load_artifact_detection_config(
            {
                "ARTIFACT_IMAGES_ZIP": "/tmp/slides.zip",
                "ARTIFACT_GEOJSON_OUTPUT": "bad\x07path",
            },
            system_name="Linux",
        )


def test_string_with_default_prefers_non_empty_environment_value() -> None:
    assert (
        _string_with_default({"ARTIFACT_DEVICE": "cpu"}, "ARTIFACT_DEVICE", DEFAULT_DEVICE)
        == "cpu"
    )


def test_string_with_default_falls_back_for_missing_or_empty_value() -> None:
    assert _string_with_default({}, "ARTIFACT_DEVICE", DEFAULT_DEVICE) == DEFAULT_DEVICE
    assert (
        _string_with_default({"ARTIFACT_DEVICE": ""}, "ARTIFACT_DEVICE", DEFAULT_DEVICE)
        == DEFAULT_DEVICE
    )


def test_float_with_default_prefers_non_empty_environment_value() -> None:
    assert (
        _float_with_default(
            {"ARTIFACT_MPP_MODEL": str(CUSTOM_MPP_MODEL)},
            "ARTIFACT_MPP_MODEL",
            DEFAULT_MPP_MODEL,
        )
        == CUSTOM_MPP_MODEL
    )


def test_float_with_default_falls_back_for_missing_or_empty_value() -> None:
    assert _float_with_default({}, "ARTIFACT_MPP_MODEL", DEFAULT_MPP_MODEL) == DEFAULT_MPP_MODEL
    assert (
        _float_with_default(
            {"ARTIFACT_MPP_MODEL": ""},
            "ARTIFACT_MPP_MODEL",
            DEFAULT_MPP_MODEL,
        )
        == DEFAULT_MPP_MODEL
    )


def test_int_with_default_prefers_non_empty_environment_value() -> None:
    assert (
        _int_with_default(
            {"ARTIFACT_OVERLAY_FACTOR": str(CUSTOM_OVERLAY_FACTOR)},
            "ARTIFACT_OVERLAY_FACTOR",
            DEFAULT_OVERLAY_FACTOR,
        )
        == CUSTOM_OVERLAY_FACTOR
    )


def test_int_with_default_falls_back_for_missing_or_empty_value() -> None:
    assert (
        _int_with_default({}, "ARTIFACT_OVERLAY_FACTOR", DEFAULT_OVERLAY_FACTOR)
        == DEFAULT_OVERLAY_FACTOR
    )
    assert (
        _int_with_default(
            {"ARTIFACT_OVERLAY_FACTOR": ""},
            "ARTIFACT_OVERLAY_FACTOR",
            DEFAULT_OVERLAY_FACTOR,
        )
        == DEFAULT_OVERLAY_FACTOR
    )


def test_required_path_returns_resolved_path() -> None:
    resolved = _required_path({"ARTIFACT_IMAGES_ZIP": "/tmp/slides.zip"}, "ARTIFACT_IMAGES_ZIP")

    assert resolved == Path("/tmp/slides.zip")


def test_path_with_default_uses_default_and_system_name() -> None:
    resolved = _path_with_default(
        {},
        "ARTIFACT_TEMP_FOLDER",
        r"C:\temp\artifact_detection",
        system_name="Windows",
    )

    assert resolved == Path("C:/temp/artifact_detection")


def test_path_with_default_requires_resolved_path() -> None:
    with patch(
        "helpers.artifact.config.resolve_env_path",
        return_value=Path("/tmp/output"),
    ) as resolve:
        resolved = _path_with_default({}, "ARTIFACT_GEOJSON_OUTPUT", "./artifacts/geojson")

    assert resolved == Path("/tmp/output")
    resolve.assert_called_once_with(
        "./artifacts/geojson",
        "ARTIFACT_GEOJSON_OUTPUT",
        system_name=None,
        required=True,
    )
