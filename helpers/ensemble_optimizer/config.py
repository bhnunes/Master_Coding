from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path
from helpers.training.registry import load_training_model_registry

VALID_SORT_METRICS = {"best_validation_DICE", "best_val_auprc_pixel_score"}
VALID_SPATIAL_PATIENT_POLICIES = {"positive_only", "all"}


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def _parse_int(value: str | None, *, default: int) -> int:
    return int(str(default) if value is None or value.strip() == "" else value)


def _parse_float(value: str | None, *, default: float) -> float:
    return float(str(default) if value is None or value.strip() == "" else value)


def _parse_choice(value: str | None, *, default: str, valid: set[str]) -> str:
    candidate = default if value is None or value.strip() == "" else value.strip()
    if candidate not in valid:
        raise ValueError(f"Expected one of {sorted(valid)}, got '{candidate}'.")
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


def _parse_architecture_group(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        candidates = default
    else:
        candidates = tuple(part.strip().upper() for part in value.split(",") if part.strip())

    registry = load_training_model_registry()
    available = set(registry.keys())
    missing = sorted(set(candidates) - available)
    if missing:
        raise ValueError(f"Unknown ensemble optimizer architectures: {', '.join(missing)}")
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return tuple(ordered)


@dataclass(frozen=True)
class EnsembleOptimizerConfig:
    master_manifest_path: Path
    metadata_dir: Path
    output_dir: Path
    local_data_dir: Path
    pred_cache_dir: Path
    stage_input_locally: bool
    overwrite_output: bool
    seed: int
    batch_size: int
    workers: int
    sort_metric: str
    val_calibration_frac: float
    val_holdout_frac: float
    semantic_architectures: tuple[str, ...]
    spatial_architectures: tuple[str, ...]
    roi_context_scale: int
    roi_max_median: float
    roi_empty_max: float
    roi_min_pos_recall: float
    spill_penalty_lambda: float
    spatial_patient_policy: str
    num_trials_semantic: int
    num_trials_spatial: int
    log_folder: Path = Path("logs")
    log_file_name: str = "ensemble_optimizer.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_ensemble_optimizer_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> EnsembleOptimizerConfig:
    values = env if env is not None else os.environ
    workers_default = os.cpu_count() or 1
    semantic_default = ("SWIN", "DPT", "SEGFORMER", "UPERNET")
    spatial_default = ("DEEPLABV3PLUS", "UNET++", "FPN", "MANET")

    return EnsembleOptimizerConfig(
        master_manifest_path=_parse_required_path(
            values.get("ENSEMBLE_OPT_MASTER_MANIFEST_PATH"),
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH",
            system_name=system_name,
        ),
        metadata_dir=_parse_required_path(
            values.get("ENSEMBLE_OPT_METADATA_DIR"),
            "ENSEMBLE_OPT_METADATA_DIR",
            system_name=system_name,
        ),
        output_dir=_parse_required_path(
            values.get("ENSEMBLE_OPT_OUTPUT_DIR"),
            "ENSEMBLE_OPT_OUTPUT_DIR",
            system_name=system_name,
            default="./reports/ensemble_optimizer",
        ),
        local_data_dir=_parse_required_path(
            values.get("ENSEMBLE_OPT_LOCAL_DATA_DIR"),
            "ENSEMBLE_OPT_LOCAL_DATA_DIR",
            system_name=system_name,
            default="./temp/ensemble_optimizer",
        ),
        pred_cache_dir=_parse_required_path(
            values.get("ENSEMBLE_OPT_PRED_CACHE_DIR"),
            "ENSEMBLE_OPT_PRED_CACHE_DIR",
            system_name=system_name,
            default="./temp/ensemble_optimizer_cache",
        ),
        stage_input_locally=_parse_bool(
            values.get("ENSEMBLE_OPT_STAGE_INPUT_LOCALLY"),
            default=True,
        ),
        overwrite_output=_parse_bool(values.get("ENSEMBLE_OPT_OVERWRITE_OUTPUT"), default=True),
        seed=_parse_int(values.get("ENSEMBLE_OPT_SEED"), default=24),
        batch_size=_parse_int(values.get("ENSEMBLE_OPT_BATCH_SIZE"), default=32),
        workers=max(1, _parse_int(values.get("ENSEMBLE_OPT_WORKERS"), default=workers_default)),
        sort_metric=_parse_choice(
            values.get("ENSEMBLE_OPT_SORT_METRIC"),
            default="best_val_auprc_pixel_score",
            valid=VALID_SORT_METRICS,
        ),
        val_calibration_frac=_parse_float(
            values.get("ENSEMBLE_OPT_VAL_CALIBRATION_FRAC"), default=0.25
        ),
        val_holdout_frac=_parse_float(values.get("ENSEMBLE_OPT_VAL_HOLDOUT_FRAC"), default=0.20),
        semantic_architectures=_parse_architecture_group(
            values.get("ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES"),
            default=semantic_default,
        ),
        spatial_architectures=_parse_architecture_group(
            values.get("ENSEMBLE_OPT_SPATIAL_ARCHITECTURES"),
            default=spatial_default,
        ),
        roi_context_scale=max(
            1, _parse_int(values.get("ENSEMBLE_OPT_ROI_CONTEXT_SCALE"), default=4)
        ),
        roi_max_median=_parse_float(values.get("ENSEMBLE_OPT_ROI_MAX_MEDIAN"), default=0.60),
        roi_empty_max=_parse_float(values.get("ENSEMBLE_OPT_ROI_EMPTY_MAX"), default=0.50),
        roi_min_pos_recall=_parse_float(
            values.get("ENSEMBLE_OPT_ROI_MIN_POS_RECALL"), default=0.80
        ),
        spill_penalty_lambda=_parse_float(
            values.get("ENSEMBLE_OPT_SPILL_PENALTY_LAMBDA"),
            default=0.10,
        ),
        spatial_patient_policy=_parse_choice(
            values.get("ENSEMBLE_OPT_SPATIAL_PATIENT_POLICY"),
            default="all",
            valid=VALID_SPATIAL_PATIENT_POLICIES,
        ),
        num_trials_semantic=_parse_int(values.get("ENSEMBLE_OPT_NUM_TRIALS_SEMANTIC"), default=50),
        num_trials_spatial=_parse_int(values.get("ENSEMBLE_OPT_NUM_TRIALS_SPATIAL"), default=50),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("ENSEMBLE_OPT_LOG_FOLDER",),
        ),
        log_file_name=(values.get("ENSEMBLE_OPT_LOG_FILE") or "ensemble_optimizer.log").strip(),
    )
