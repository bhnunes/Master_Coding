from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path
from helpers.training.registry import validate_architecture_encoder_pair

VALID_EXECUTION_MODES = {"FAST_DEV", "PAPER"}
VALID_AMP_PRECISIONS = {"fp16", "bf16", "fp32", "auto"}
VALID_OPTIMIZERS = {"AdamWScheduleFree", "AdamW"}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "t", "yes", "y"}


def _parse_int(value: str | None, variable_name: str, default: int) -> int:
    _ = variable_name
    candidate = str(default) if value is None or value == "" else value
    return int(candidate)


def _parse_float(value: str | None, variable_name: str, default: float) -> float:
    _ = variable_name
    candidate = str(default) if value is None or value == "" else value
    return float(candidate)


def _parse_choice(
    value: str | None,
    variable_name: str,
    *,
    default: str,
    valid_values: set[str],
    normalize: bool = False,
) -> str:
    raw_candidate = default if value is None or value == "" else value
    candidate = raw_candidate.upper() if normalize else raw_candidate
    if candidate not in valid_values:
        valid_display = ", ".join(sorted(valid_values))
        raise ValueError(
            f"The '{variable_name}' environment variable must be one of: {valid_display}."
        )
    return candidate


def _parse_required_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
    default: str | None = None,
) -> Path:
    resolved = resolve_env_path(
        value if value not in {None, ""} else default,
        variable_name,
        system_name=system_name,
        required=True,
    )
    assert resolved is not None
    return resolved


def _parse_optional_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
) -> Path | None:
    return resolve_env_path(value, variable_name, system_name=system_name)


def _parse_email_recipients(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    recipients = tuple(recipient.strip() for recipient in value.split(",") if recipient.strip())
    return recipients


@dataclass(frozen=True)
class TrainingEnsembleConfig:
    """Runtime configuration for `10_training_ensemble.py`."""

    num_epochs: int
    patience: int
    unleashed: bool
    seed: int
    batch_size: int
    workers: int
    accumulation_steps: int
    val_batch_size: int
    optimizer_name: str
    sensitivity_target: float
    hdf5_drive_dir: Path
    metadata_dir: Path
    identifier: str
    local_data_dir: Path
    checkpoint_path: Path
    aim_repo_path: Path
    email_sender: str | None
    email_recipients: tuple[str, ...]
    email_password: str | None
    execution_mode: str
    smart_sampling: bool
    amp_precision: str
    architecture: str
    encoder: str
    resume_checkpoint: Path | None
    use_artifact_aware_loss: bool
    artifact_index_path: Path | None
    log_folder: Path = Path("logs")
    log_file_name: str = "training_ensemble.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_training_ensemble_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> TrainingEnsembleConfig:
    """Load and validate runtime configuration for `10_training_ensemble.py`."""

    values = env if env is not None else os.environ

    batch_size = _parse_int(values.get("TRAINING_BATCH_SIZE"), "TRAINING_BATCH_SIZE", default=32)
    val_batch_multiplier = _parse_int(
        values.get("TRAINING_VAL_BATCH_MULTIPLIER"),
        "TRAINING_VAL_BATCH_MULTIPLIER",
        default=3,
    )
    workers_default = os.cpu_count() or 1
    architecture = (values.get("TRAINING_ARCHITECTURE") or "FPN").strip().upper()
    encoder = (values.get("TRAINING_ENCODER") or "senet154").strip()
    validate_architecture_encoder_pair(architecture, encoder)
    resume_checkpoint = _parse_optional_path(
        values.get("TRAINING_RESUME_CHECKPOINT"),
        "TRAINING_RESUME_CHECKPOINT",
        system_name=system_name,
    )
    use_artifact_aware_loss = _parse_bool(
        values.get("TRAINING_USE_ARTIFACT_AWARE_LOSS"),
        default=False,
    )
    artifact_index_path = _parse_optional_path(
        values.get("TRAINING_ARTIFACT_INDEX_PATH"),
        "TRAINING_ARTIFACT_INDEX_PATH",
        system_name=system_name,
    )
    if use_artifact_aware_loss and artifact_index_path is None:
        raise ValueError(
            "The 'TRAINING_ARTIFACT_INDEX_PATH' environment variable is required when "
            "TRAINING_USE_ARTIFACT_AWARE_LOSS is enabled."
        )

    execution_mode = _parse_choice(
        values.get("TRAINING_EXECUTION_MODE"),
        "TRAINING_EXECUTION_MODE",
        default="PAPER",
        valid_values=VALID_EXECUTION_MODES,
        normalize=True,
    )
    amp_precision = _parse_choice(
        values.get("TRAINING_AMP_PRECISION"),
        "TRAINING_AMP_PRECISION",
        default="fp16",
        valid_values=VALID_AMP_PRECISIONS,
    )
    optimizer_name = _parse_choice(
        values.get("TRAINING_OPTIMIZER_NAME"),
        "TRAINING_OPTIMIZER_NAME",
        default="AdamWScheduleFree",
        valid_values=VALID_OPTIMIZERS,
    )

    return TrainingEnsembleConfig(
        num_epochs=_parse_int(values.get("TRAINING_NUM_EPOCHS"), "TRAINING_NUM_EPOCHS", 100),
        patience=_parse_int(values.get("TRAINING_PATIENCE"), "TRAINING_PATIENCE", 5),
        unleashed=_parse_bool(values.get("TRAINING_UNLEASHED"), default=False),
        seed=_parse_int(values.get("TRAINING_SEED"), "TRAINING_SEED", 24),
        batch_size=batch_size,
        workers=max(
            1,
            _parse_int(values.get("TRAINING_WORKERS"), "TRAINING_WORKERS", workers_default),
        ),
        accumulation_steps=_parse_int(
            values.get("TRAINING_ACCUMULATION_STEPS"),
            "TRAINING_ACCUMULATION_STEPS",
            4,
        ),
        val_batch_size=batch_size * val_batch_multiplier,
        optimizer_name=optimizer_name,
        sensitivity_target=_parse_float(
            values.get("TRAINING_SENSITIVITY_TARGET"),
            "TRAINING_SENSITIVITY_TARGET",
            0.95,
        ),
        hdf5_drive_dir=_parse_required_path(
            values.get("TRAINING_HDF5_DRIVE_DIR"),
            "TRAINING_HDF5_DRIVE_DIR",
            system_name=system_name,
        ),
        metadata_dir=_parse_required_path(
            values.get("TRAINING_METADATA_DIR"),
            "TRAINING_METADATA_DIR",
            system_name=system_name,
        ),
        identifier=(values.get("TRAINING_IDENTIFIER") or "CAMELYON16_HDF5_OPTIMIZED").strip(),
        local_data_dir=_parse_required_path(
            values.get("TRAINING_LOCAL_DATA_DIR"),
            "TRAINING_LOCAL_DATA_DIR",
            system_name=system_name,
            default="./temp/training_ensemble",
        ),
        checkpoint_path=_parse_required_path(
            values.get("TRAINING_CHECKPOINT_PATH"),
            "TRAINING_CHECKPOINT_PATH",
            system_name=system_name,
        ),
        aim_repo_path=_parse_required_path(
            values.get("TRAINING_AIM_REPO_PATH"),
            "TRAINING_AIM_REPO_PATH",
            system_name=system_name,
        ),
        email_sender=(values.get("TRAINING_EMAIL_SENDER") or "").strip() or None,
        email_recipients=_parse_email_recipients(values.get("TRAINING_EMAIL_RECIPIENTS")),
        email_password=(values.get("TRAINING_EMAIL_PASSWORD") or "").strip() or None,
        execution_mode=execution_mode,
        smart_sampling=_parse_bool(values.get("TRAINING_SMART_SAMPLING"), default=True),
        amp_precision=amp_precision,
        architecture=architecture,
        encoder=encoder,
        resume_checkpoint=resume_checkpoint,
        use_artifact_aware_loss=use_artifact_aware_loss,
        artifact_index_path=artifact_index_path,
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("TRAINING_LOG_FOLDER",),
        ),
        log_file_name=(values.get("TRAINING_LOG_FILE") or "training_ensemble.log").strip(),
    )
