from pathlib import Path

import pytest

from helpers.patient_shards.config import load_patient_shards_config


def test_load_patient_shards_config_reads_defaults(tmp_path: Path) -> None:
    config = load_patient_shards_config(
        {
            "PATIENT_SHARDS_STAGE5_BASE_DIR": str(tmp_path / "stage5"),
        }
    )

    assert config.stage5_base_dir == tmp_path / "stage5"
    assert config.output_base_dir == tmp_path / "stage5"
    assert config.overwrite_output is False
    assert config.hdf5_compression == "NONE"
    assert config.copy_batch_size == 256
    assert config.log_path == Path("logs/patient_shards.log")


def test_load_patient_shards_config_reads_explicit_values(tmp_path: Path) -> None:
    config = load_patient_shards_config(
        {
            "PATIENT_SHARDS_STAGE5_BASE_DIR": str(tmp_path / "stage5"),
            "PATIENT_SHARDS_OUTPUT_BASE_DIR": str(tmp_path / "out"),
            "PATIENT_SHARDS_OVERWRITE_OUTPUT": "true",
            "PATIENT_SHARDS_HDF5_COMPRESSION": "lzf",
            "PATIENT_SHARDS_COPY_BATCH_SIZE": "32",
            "LOG_FOLDER": str(tmp_path / "logs"),
            "PATIENT_SHARDS_LOG_FILE": "stage6_5.log",
        }
    )

    assert config.output_base_dir == tmp_path / "out"
    assert config.overwrite_output is True
    assert config.hdf5_compression == "LZF"
    assert config.copy_batch_size == 32
    assert config.log_path == tmp_path / "logs" / "stage6_5.log"


def test_load_patient_shards_config_rejects_invalid_compression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PATIENT_SHARDS_HDF5_COMPRESSION"):
        load_patient_shards_config(
            {
                "PATIENT_SHARDS_STAGE5_BASE_DIR": str(tmp_path / "stage5"),
                "PATIENT_SHARDS_HDF5_COMPRESSION": "brotli",
            }
        )
