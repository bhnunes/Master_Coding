import json
from pathlib import Path

import pytest

from helpers.training.config import load_training_ensemble_config

TRAINING_WORKERS = 6
TRAINING_BATCH_SIZE = 12
VALIDATION_BATCH_SIZE = 48
OHEM_START_EPOCH = 3
OHEM_RATIO = 0.4
OHEM_MIN_KEPT = 2048
DEFAULT_OHEM_START_EPOCH = 2
DEFAULT_OHEM_RATIO = 0.25
DEFAULT_OHEM_MIN_KEPT = 1024


@pytest.fixture(autouse=True)
def _training_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "training_model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "FPN": {
                    "lr": 1e-4,
                    "wd": 1e-4,
                    "encoders": ["senet154"],
                    "loss": {"alpha_bce": 0.1, "beta_dice_bg": 0.2, "gamma_dice_fg": 0.3},
                },
                "SEGFORMER": {
                    "lr": 1e-4,
                    "wd": 1e-4,
                    "encoders": ["mit_b5"],
                    "loss": {"alpha_bce": 0.1, "beta_dice_bg": 0.2, "gamma_dice_fg": 0.3},
                },
                "SWIN": {
                    "lr": 1e-4,
                    "wd": 1e-4,
                    "encoders": ["tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k"],
                    "loss": {"alpha_bce": 0.1, "beta_dice_bg": 0.2, "gamma_dice_fg": 0.3},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))


def test_load_training_ensemble_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_MASTER_MANIFEST_PATH": str(
                tmp_path / "projects" / "camelyon16" / "master_manifest.sqlite"
            ),
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
            "TRAINING_USE_COMPACT_TRAIN_SELECTED": "false",
            "TRAIN_SELECTED_COMPACT_DIR": str(tmp_path / "compact"),
            "TRAINING_RUN_OHEM": "true",
            "TRAINING_OHEM_START_EPOCH": "3",
            "TRAINING_OHEM_RATIO": "0.4",
            "TRAINING_OHEM_MIN_KEPT": "2048",
        }
    )

    assert config.master_manifest_path == (
        tmp_path / "projects" / "camelyon16" / "master_manifest.sqlite"
    )
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
    assert config.workers == TRAINING_WORKERS
    assert config.batch_size == TRAINING_BATCH_SIZE
    assert config.val_batch_size == VALIDATION_BATCH_SIZE
    assert config.amp_precision == "bf16"
    assert config.execution_mode == "PAPER"
    assert config.smart_sampling is False
    assert config.unleashed is True
    assert config.run_ohem is True
    assert config.ohem_start_epoch == OHEM_START_EPOCH
    assert config.ohem_ratio == OHEM_RATIO
    assert config.ohem_min_kept == OHEM_MIN_KEPT
    assert config.use_artifact_aware_loss is True
    assert config.use_compact_train_selected is False
    assert config.compact_train_selected_dir == tmp_path / "compact"
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"
    assert config.log_path == Path("logs/training_ensemble.log")


def test_load_training_ensemble_config_uses_portable_defaults(tmp_path: Path) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_MASTER_MANIFEST_PATH": str(tmp_path / "dataset" / "master_manifest.sqlite"),
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
    assert config.run_ohem is False
    assert config.ohem_start_epoch == DEFAULT_OHEM_START_EPOCH
    assert config.ohem_ratio == DEFAULT_OHEM_RATIO
    assert config.ohem_min_kept == DEFAULT_OHEM_MIN_KEPT
    assert config.use_artifact_aware_loss is False
    assert config.use_compact_train_selected is True
    assert config.compact_train_selected_dir == Path("temp/train_selected_compact")
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"
    assert config.master_manifest_path == tmp_path / "dataset" / "master_manifest.sqlite"
    assert config.log_path == Path("logs/training_ensemble.log")


def test_load_training_ensemble_config_reads_shared_runtime_normalization_method(
    tmp_path: Path,
) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_MASTER_MANIFEST_PATH": str(tmp_path / "dataset" / "master_manifest.sqlite"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
            "RUNTIME_NORMALIZATION_METHOD": "macenko",
        }
    )

    assert config.runtime_normalization_method == "MACENKO"


def test_load_training_ensemble_config_reads_runtime_vahadane_backend(
    tmp_path: Path,
) -> None:
    config = load_training_ensemble_config(
        {
            "TRAINING_MASTER_MANIFEST_PATH": str(tmp_path / "dataset" / "master_manifest.sqlite"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
            "RUNTIME_VAHADANE_BACKEND": "torch_staintools_exact",
        }
    )

    assert config.runtime_vahadane_backend == "torch_staintools_exact"


def test_load_training_ensemble_config_rejects_invalid_execution_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TRAINING_EXECUTION_MODE"):
        load_training_ensemble_config(
            {
                "TRAINING_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
                "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
                "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
                "TRAINING_EXECUTION_MODE": "turbo",
            }
        )


def test_load_training_ensemble_config_requires_dataset_path() -> None:
    with pytest.raises(ValueError, match="TRAINING_MASTER_MANIFEST_PATH"):
        load_training_ensemble_config({})


def test_load_training_ensemble_config_rejects_unapproved_encoder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TRAINING_ENCODER"):
        load_training_ensemble_config(
            {
                "TRAINING_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
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
            "TRAINING_MASTER_MANIFEST_PATH": str(tmp_path / "dataset" / "master_manifest.sqlite"),
            "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
            "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
            "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
            "TRAINING_ARCHITECTURE": "SWIN",
            "TRAINING_ENCODER": "tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        }
    )

    assert config.architecture == "SWIN"
    assert config.encoder == "tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k"


def test_load_training_ensemble_config_rejects_invalid_ohem_ratio(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TRAINING_OHEM_RATIO"):
        load_training_ensemble_config(
            {
                "TRAINING_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "TRAINING_METADATA_DIR": str(tmp_path / "metadata"),
                "TRAINING_CHECKPOINT_PATH": str(tmp_path / "checkpoints"),
                "TRAINING_AIM_REPO_PATH": str(tmp_path / "aim"),
                "TRAINING_OHEM_RATIO": "0",
            }
        )
