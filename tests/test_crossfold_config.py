from pathlib import Path

import pytest

from helpers.crossfold.config import load_crossfold_config

TEST_RANDOM_STATE = 7
DEFAULT_TEST_PATIENT_COUNT = 20
EXPLICIT_TEST_PATIENT_COUNT = 24
DEFAULT_VALIDATION_PATIENT_COUNT = 20
EXPLICIT_VALIDATION_PATIENT_COUNT = 21
DEFAULT_OPTUNA_TRIALS = 1000
EXPLICIT_OPTUNA_TRIALS = 99
ENTROPY_WORKERS = 3
ENTROPY_CHUNKSIZE = 64
ENTROPY_THUMBNAIL = 256
COPY_BATCH_SIZE = 32


def test_load_crossfold_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "CROSSFOLD_NORMALIZATION_METHOD": "MACENKO",
            "CROSSFOLD_OVERWRITE_OUTPUT_DIR": "false",
            "CROSSFOLD_RANDOM_STATE": "7",
            "CROSSFOLD_TEST_PATIENT_COUNT": "24",
            "CROSSFOLD_VALIDATION_PATIENT_COUNT": "21",
            "CROSSFOLD_SPLIT_OPTUNA_TRIALS": "99",
            "CROSSFOLD_ENTROPY_NUM_WORKERS": "3",
            "CROSSFOLD_ENTROPY_CHUNKSIZE": "64",
            "CROSSFOLD_ENTROPY_THUMBNAIL": "256",
            "CROSSFOLD_HDF5_COMPRESSION": "lzf",
            "CROSSFOLD_COPY_BATCH_SIZE": "32",
            "CROSSFOLD_CALC_CHECKSUMS": "true",
            "CROSSFOLD_SAVE_ENTROPY_CACHE_CSV": "false",
        }
    )

    assert config.source_path == tmp_path / "SOURCE_DATASET.h5"
    assert config.normalization_method == "MACENKO"
    assert config.overwrite_output_dir is False
    assert config.random_state == TEST_RANDOM_STATE
    assert config.constraints.test_patient_count == EXPLICIT_TEST_PATIENT_COUNT
    assert config.constraints.validation_patient_count == EXPLICIT_VALIDATION_PATIENT_COUNT
    assert config.objective.optuna_trials == EXPLICIT_OPTUNA_TRIALS
    assert config.objective.num_workers == ENTROPY_WORKERS
    assert config.objective.chunksize == ENTROPY_CHUNKSIZE
    assert config.objective.entropy_thumbnail == ENTROPY_THUMBNAIL
    assert config.hdf5_compression == "LZF"
    assert config.copy_batch_size == COPY_BATCH_SIZE
    assert config.calc_checksums is True
    assert config.save_entropy_cache_csv is False
    assert config.log_path == Path("logs/data_preparation.log")


def test_load_crossfold_config_prefers_global_log_folder(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "LOG_FOLDER": str(tmp_path / "shared_logs"),
            "CROSSFOLD_LOG_FOLDER": str(tmp_path / "legacy_logs"),
            "CROSSFOLD_LOG_FILE": "crossfold.log",
        }
    )

    assert config.log_folder == tmp_path / "shared_logs"
    assert config.log_path == tmp_path / "shared_logs" / "crossfold.log"


def test_load_crossfold_config_allows_explicit_destructive_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE"):
        load_crossfold_config(
            {
                "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
                "CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE": "true",
            }
        )


def test_load_crossfold_config_requires_source_hdf5_path() -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_SOURCE_HDF5_PATH"):
        load_crossfold_config({})


def test_load_crossfold_config_accepts_sqlite_source_path(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "master_manifest.sqlite"),
        }
    )

    assert config.source_path == tmp_path / "master_manifest.sqlite"


def test_load_crossfold_config_rejects_invalid_normalization_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_NORMALIZATION_METHOD"):
        load_crossfold_config(
            {
                "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
                "CROSSFOLD_NORMALIZATION_METHOD": "INVALID",
            }
        )


def test_load_crossfold_config_defaults_new_split_surface(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
        }
    )

    assert config.constraints.test_patient_count == DEFAULT_TEST_PATIENT_COUNT
    assert config.constraints.validation_patient_count == DEFAULT_VALIDATION_PATIENT_COUNT
    assert config.objective.optuna_trials == DEFAULT_OPTUNA_TRIALS


def test_load_crossfold_config_ignores_removed_legacy_split_env_vars(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "CROSSFOLD_TEST_RATIO": "0.5",
            "CROSSFOLD_MIN_TEST_PATIENTS": "3",
            "CROSSFOLD_ENABLE_OBJECTIVE": "false",
        }
    )

    assert config.constraints.test_patient_count == DEFAULT_TEST_PATIENT_COUNT
    assert config.objective.optuna_trials == DEFAULT_OPTUNA_TRIALS


def test_load_crossfold_config_clamps_optuna_trials_to_positive_value(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "CROSSFOLD_SPLIT_OPTUNA_TRIALS": "0",
        }
    )

    assert config.objective.optuna_trials == 1
