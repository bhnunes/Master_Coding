from pathlib import Path

import pytest

from helpers.crossfold.config import load_crossfold_config


def test_load_crossfold_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "CROSSFOLD_NORMALIZATION_METHOD": "MACENKO",
            "CROSSFOLD_OVERWRITE_OUTPUT_DIR": "false",
            "CROSSFOLD_RANDOM_STATE": "7",
            "CROSSFOLD_TEST_RATIO": "0.2",
            "CROSSFOLD_VAL_RATIO": "0.15",
            "CROSSFOLD_MIN_TRAIN_PATIENTS": "4",
            "CROSSFOLD_MIN_VAL_PATIENTS": "2",
            "CROSSFOLD_MIN_TEST_PATIENTS": "3",
            "CROSSFOLD_ENFORCE_STAGE11_VALIDATION_SIZING": "true",
            "CROSSFOLD_MIN_VALIDATION_PATIENTS_FOR_ENSEMBLE": "30",
            "CROSSFOLD_MIN_VALIDATION_POSITIVE_PATIENTS_FOR_ENSEMBLE": "15",
            "CROSSFOLD_MIN_VALIDATION_NEGATIVE_PATIENTS_FOR_ENSEMBLE": "15",
            "CROSSFOLD_ADAPTIVE": "false",
            "CROSSFOLD_REQUIRE_TRAIN_IMAGE_DOMINANCE": "false",
            "CROSSFOLD_REQUIRE_BOTH_CLASSES_IF_POSSIBLE": "false",
            "CROSSFOLD_MAX_TRIES": "99",
            "CROSSFOLD_ENABLE_OBJECTIVE": "false",
            "CROSSFOLD_OBJECTIVE_SCORE_SPLIT": "VALIDATION",
            "CROSSFOLD_OBJECTIVE_MAXIMIZE": "false",
            "CROSSFOLD_ENTROPY_NUM_WORKERS": "3",
            "CROSSFOLD_ENTROPY_CHUNKSIZE": "64",
            "CROSSFOLD_ENTROPY_THUMBNAIL": "256",
            "CROSSFOLD_CALC_CHECKSUMS": "true",
            "CROSSFOLD_SAVE_ENTROPY_CACHE_CSV": "false",
        }
    )

    assert config.source_hdf5_path == tmp_path / "SOURCE_DATASET.h5"
    assert config.normalization_method == "MACENKO"
    assert config.overwrite_output_dir is False
    assert config.random_state == 7
    assert config.allow_destructive_move is False
    assert config.constraints.test_ratio == 0.2
    assert config.constraints.val_ratio == 0.15
    assert config.constraints.min_train_patients == 4
    assert config.constraints.min_val_patients == 2
    assert config.constraints.min_test_patients == 3
    assert config.constraints.enforce_stage11_validation_sizing is True
    assert config.constraints.min_validation_patients_for_ensemble == 30
    assert config.constraints.min_validation_positive_patients_for_ensemble == 15
    assert config.constraints.min_validation_negative_patients_for_ensemble == 15
    assert config.constraints.adaptive is False
    assert config.constraints.require_train_image_dominance is False
    assert config.constraints.require_both_classes_if_possible is False
    assert config.constraints.max_tries == 99
    assert config.objective.enable_objective is False
    assert config.objective.score_split == "VALIDATION"
    assert config.objective.maximize is False
    assert config.objective.num_workers == 3
    assert config.objective.chunksize == 64
    assert config.objective.entropy_thumbnail == 256
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
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
            "CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE": "true",
        }
    )

    assert config.allow_destructive_move is True


def test_load_crossfold_config_requires_data_directory() -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_SOURCE_HDF5_PATH"):
        load_crossfold_config({})


def test_load_crossfold_config_rejects_invalid_normalization_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_NORMALIZATION_METHOD"):
        load_crossfold_config(
            {
                "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
                "CROSSFOLD_NORMALIZATION_METHOD": "INVALID",
            }
        )


def test_load_crossfold_config_rejects_test_objective_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CROSSFOLD_OBJECTIVE_SCORE_SPLIT"):
        load_crossfold_config(
            {
                "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
                "CROSSFOLD_OBJECTIVE_SCORE_SPLIT": "TEST",
            }
        )


def test_load_crossfold_config_defaults_stage11_validation_guardrails(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_SOURCE_HDF5_PATH": str(tmp_path / "SOURCE_DATASET.h5"),
        }
    )

    assert config.constraints.enforce_stage11_validation_sizing is True
    assert config.constraints.min_validation_patients_for_ensemble == 30
    assert config.constraints.min_validation_positive_patients_for_ensemble == 15
    assert config.constraints.min_validation_negative_patients_for_ensemble == 15


def test_load_crossfold_config_still_accepts_legacy_directory_variable(tmp_path: Path) -> None:
    config = load_crossfold_config(
        {
            "CROSSFOLD_DATA_DIRECTORY": str(tmp_path / "legacy_patches"),
        }
    )

    assert config.source_hdf5_path == tmp_path / "legacy_patches"
