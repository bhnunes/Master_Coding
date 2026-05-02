from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_normalization import (
    RUNTIME_NORMALIZATION_METHOD_ENV_VAR,
    RUNTIME_VAHADANE_BACKEND_ENV_VAR,
    parse_runtime_normalization_method,
    parse_runtime_vahadane_backend,
)
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
    """Runtime configuration for `10_training_ensemble.py`.

    Stage 9 now supports two independent loss modifiers:

    - ``use_artifact_aware_loss`` applies a sample-level discount using the
      Stage 2 filename-keyed artifact coverage metadata.
    - ``run_ohem`` applies pixel-level Online Hard Example Mining during
      training only, after the configured warm-up epoch.
    """

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
    master_manifest_path: Path
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
    run_ohem: bool
    ohem_start_epoch: int
    ohem_ratio: float
    ohem_min_kept: int
    architecture: str
    encoder: str
    resume_checkpoint: Path | None
    use_artifact_aware_loss: bool
    runtime_normalization_method: str = "NOT_NORMALIZED"
    runtime_vahadane_backend: str = "fixed_source"
    use_compact_train_selected: bool = True
    compact_train_selected_dir: Path = Path("temp/train_selected_compact")
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
    master_manifest_path = _parse_required_path(
        values.get("TRAINING_MASTER_MANIFEST_PATH"),
        "TRAINING_MASTER_MANIFEST_PATH",
        system_name=system_name,
    )
    compact_train_selected_dir = _parse_required_path(
        values.get("TRAIN_SELECTED_COMPACT_DIR"),
        "TRAIN_SELECTED_COMPACT_DIR",
        system_name=system_name,
        default="./temp/train_selected_compact",
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
    run_ohem = _parse_bool(values.get("TRAINING_RUN_OHEM"), default=False)
    ohem_start_epoch = _parse_int(
        values.get("TRAINING_OHEM_START_EPOCH"),
        "TRAINING_OHEM_START_EPOCH",
        2,
    )
    ohem_ratio = _parse_float(values.get("TRAINING_OHEM_RATIO"), "TRAINING_OHEM_RATIO", 0.25)
    ohem_min_kept = _parse_int(
        values.get("TRAINING_OHEM_MIN_KEPT"),
        "TRAINING_OHEM_MIN_KEPT",
        1024,
    )
    if ohem_ratio <= 0.0 or ohem_ratio > 1.0:
        raise ValueError("The 'TRAINING_OHEM_RATIO' environment variable must be in (0, 1].")
    if ohem_min_kept < 1:
        raise ValueError(
            "The 'TRAINING_OHEM_MIN_KEPT' environment variable must be greater than 0."
        )
    if ohem_start_epoch < 0:
        raise ValueError("The 'TRAINING_OHEM_START_EPOCH' environment variable must be >= 0.")
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
        master_manifest_path=master_manifest_path,
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
        run_ohem=run_ohem,
        ohem_start_epoch=ohem_start_epoch,
        ohem_ratio=ohem_ratio,
        ohem_min_kept=ohem_min_kept,
        architecture=architecture,
        encoder=encoder,
        resume_checkpoint=resume_checkpoint,
        use_artifact_aware_loss=use_artifact_aware_loss,
        runtime_normalization_method=parse_runtime_normalization_method(
            values.get(RUNTIME_NORMALIZATION_METHOD_ENV_VAR)
        ),
        runtime_vahadane_backend=parse_runtime_vahadane_backend(
            values.get(RUNTIME_VAHADANE_BACKEND_ENV_VAR)
        ),
        use_compact_train_selected=_parse_bool(
            values.get("TRAINING_USE_COMPACT_TRAIN_SELECTED"),
            default=True,
        ),
        compact_train_selected_dir=compact_train_selected_dir,
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("TRAINING_LOG_FOLDER",),
        ),
        log_file_name=(values.get("TRAINING_LOG_FILE") or "training_ensemble.log").strip(),
    )
