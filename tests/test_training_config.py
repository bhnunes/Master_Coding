from pathlib import Path

import pytest

from helpers.training.config import load_training_ensemble_config


def test_load_training_ensemble_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_HDF5_DRIVE_DIR": str(tmp_path / "projects" / "camelyon16"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
            "TRAINING_LOCAL_DATA_DIR": str(tmp_path / "cache"),
            "TRAINING_EMAIL_SENDER": "train@example.com",
            "TRAINING_EMAIL_RECIPIENTS": "a@example.com, b@example.com",
            "TRAINING_EMAIL_PASSWORD": "secret",
            "TRAINING_ARCHITECTURE": "segformer",
            "TRAINING_ENCODER": "mit_b5",
            "TRAINING_RESUME_CHECKPOINT": "resume/checkpoint.pth",
            "TRAINING_WORKERS": "6",
            "TRAINING_BATCH_SIZE": "12",
            "TRAINING_VAL_BATCH_MULTIPLIER": "4",
            "TRAINING_AMP_PRECISION": "bf16",
            "TRAINING_EXECUTION_MODE": "PAPER",
            "TRAINING_SMART_SAMPLING": "false",
            "TRAINING_UNLEASHED": "true",
            "TRAINING_USE_ARTIFACT_AWARE_LOSS": "true",
            "TRAINING_ARTIFACT_INDEX_PATH": str(tmp_path / "artifact_patch_index.parquet"),
        }
    )

    assert config.hdf5_drive_dir == tmp_path / "projects" / "camelyon16"
    assert config.metadata_dir == tmp_path / "metadata"
    assert config.checkpoint_path == tmp_path / "checkpoints"
    assert config.aim_repo_path == tmp_path / "aim"
    assert config.local_data_dir == tmp_path / "cache"
    assert config.email_sender == "train@example.com"
    assert config.email_recipients == ("a@example.com", "b@example.com")
    assert config.email_password == "secret"
    assert config.architecture == "SEGFORMER"
    assert config.encoder == "mit_b5"
    assert config.resume_checkpoint == Path("resume/checkpoint.pth")
    assert config.workers == 6
    assert config.batch_size == 12
    assert config.val_batch_size == 48
    assert config.amp_precision == "bf16"
    assert config.execution_mode == "PAPER"
    assert config.smart_sampling is False
    assert config.unleashed is True
    assert config.use_artifact_aware_loss is True
    assert config.artifact_index_path == tmp_path / "artifact_patch_index.parquet"
    assert config.log_path == Path("logs/training_ensemble.log")


def test_load_training_ensemble_config_uses_portable_defaults(tmp_path: Path) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
        }
    )

    assert config.local_data_dir == Path("temp/training_ensemble")
    assert config.architecture == "FPN"
    assert config.encoder == "senet154"
    assert config.resume_checkpoint is None
    assert config.email_recipients == ()
    assert config.amp_precision == "fp16"
    assert config.execution_mode == "PAPER"
    assert config.use_artifact_aware_loss is False
    assert config.artifact_index_path is None
    assert config.log_path == Path("logs/training_ensemble.log")


def test_load_training_ensemble_config_rejects_invalid_execution_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TRAINING_EXECUTION_MODE"):
        load_training_ensemble_config(
            {
                "TRAINING_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
                "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
                "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
                "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
                "TRAINING_EXECUTION_MODE": "turbo",
            }
        )


def test_load_training_ensemble_config_requires_dataset_path() -> None:
    with pytest.raises(ValueError, match="TRAINING_HDF5_DRIVE_DIR"):
        load_training_ensemble_config({})


def test_load_training_ensemble_config_rejects_unapproved_encoder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TRAINING_ENCODER"):
        load_training_ensemble_config(
            {
                "TRAINING_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
                "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
                "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
                "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
                "TRAINING_ARCHITECTURE": "FPN",
                "TRAINING_ENCODER": "resnet34",
            }
        )


def test_load_training_ensemble_config_accepts_research_approved_pair(tmp_path: Path) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
            "TRAINING_ARCHITECTURE": "SWIN",
            "TRAINING_ENCODER": "tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        }
    )

    assert config.architecture == "SWIN"
    assert config.encoder == "tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k"
