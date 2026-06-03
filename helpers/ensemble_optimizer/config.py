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
from helpers.training.registry import load_training_model_registry

VALID_SORT_METRICS = {"best_validation_DICE", "best_val_auprc_pixel_score"}
VALID_SPATIAL_PATIENT_POLICIES = {"positive_only", "all"}
DEFAULT_OPTIMIZATION_CACHE_MAX_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_DECISION_THRESHOLD_MIN = 0.50
DEFAULT_DECISION_THRESHOLD_MAX = 0.99
DEFAULT_DECISION_THRESHOLD_STEP = 0.01
DEFAULT_POS_DICE_DROP_TOLERANCE = 0.02
DEFAULT_POS_TPR_DROP_TOLERANCE = 0.02
DEFAULT_MIN_MICRO_DICE = 0.80
DEFAULT_NEGATIVE_CLEAN_TARGET = 0.50
DEFAULT_MIN_MACRO_PRECISION = 0.50
DEFAULT_MIN_MACRO_TPR = 0.80
DEFAULT_NEGATIVE_FP_PENALTY_LAMBDA = 0.50
DEFAULT_ROI_NEGATIVE_AREA_PENALTY_LAMBDA = 0.25
DEFAULT_MIN_COMPONENT_AREA_PX_CANDIDATES = (
    0,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
DEFAULT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES = (1, 2, 3, 5, 8, 13, 21)
DEFAULT_MIN_PATIENT_POSITIVE_AREA_FRACTION_CANDIDATES = (
    0.0,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
)
DEFAULT_MIN_COMPONENT_AREA_FRACTION_PATCH_CANDIDATES = (0.0, 0.001, 0.0025, 0.005, 0.01)


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def _parse_int(value: str | None, *, default: int) -> int:
    return int(str(default) if value is None or value.strip() == "" else value)


def _parse_non_negative_int(value: str | None, *, default: int, variable_name: str) -> int:
    parsed = _parse_int(value, default=default)
    if parsed < 0:
        raise ValueError(f"{variable_name} must be greater than or equal to 0.")
    return parsed


def _parse_float(value: str | None, *, default: float) -> float:
    return float(str(default) if value is None or value.strip() == "" else value)


def _parse_int_tuple(
    value: str | None, *, default: tuple[int, ...], variable_name: str
) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        parsed = default
    else:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{variable_name} must contain at least one integer candidate.")
    return tuple(dict.fromkeys(parsed))


def _parse_float_tuple(
    value: str | None, *, default: tuple[float, ...], variable_name: str
) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        parsed = default
    else:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{variable_name} must contain at least one float candidate.")
    return tuple(dict.fromkeys(parsed))


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


def _validate_disjoint_architecture_groups(
    semantic_architectures: tuple[str, ...],
    spatial_architectures: tuple[str, ...],
) -> None:
    overlap = sorted(set(semantic_architectures) & set(spatial_architectures))
    if overlap:
        raise ValueError(
            "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES and ENSEMBLE_OPT_SPATIAL_ARCHITECTURES "
            "must be disjoint; overlapping architectures: "
            f"{', '.join(overlap)}"
        )


def _validate_rule6_drop_tolerances(
    *,
    pos_dice_drop_tolerance: float,
    pos_tpr_drop_tolerance: float,
) -> None:
    if pos_dice_drop_tolerance < 0.0:
        raise ValueError("pos_dice_drop_tolerance must be greater than or equal to 0.")
    if pos_tpr_drop_tolerance < 0.0:
        raise ValueError("pos_tpr_drop_tolerance must be greater than or equal to 0.")


def _validate_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1].")


def _validate_rule6_targets(
    *,
    min_micro_dice: float,
    negative_clean_target: float,
    min_macro_precision: float,
    min_macro_tpr: float,
) -> None:
    for value, name in (
        (min_micro_dice, "min_micro_dice"),
        (negative_clean_target, "negative_clean_target"),
        (min_macro_precision, "min_macro_precision"),
        (min_macro_tpr, "min_macro_tpr"),
    ):
        _validate_unit_interval(value, name)


def _validate_non_negative_float(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be greater than or equal to 0.")


def _validate_rule6_integer_candidates(
    *,
    min_component_area_px_candidates: tuple[int, ...],
    min_patient_positive_patches_candidates: tuple[int, ...],
) -> None:
    if any(candidate < 0 for candidate in min_component_area_px_candidates):
        raise ValueError("min_component_area_px_candidates cannot contain negative values.")
    if any(candidate < 1 for candidate in min_patient_positive_patches_candidates):
        raise ValueError("min_patient_positive_patches_candidates must contain positive integers.")
    if 0 not in min_component_area_px_candidates:
        raise ValueError("min_component_area_px_candidates must include 0 for the baseline.")
    if 1 not in min_patient_positive_patches_candidates:
        raise ValueError("min_patient_positive_patches_candidates must include 1 for the baseline.")


def _validate_rule6_fraction_candidates(
    *,
    min_patient_positive_area_fraction_candidates: tuple[float, ...],
    min_component_area_fraction_patch_candidates: tuple[float, ...],
) -> None:
    for candidate in min_patient_positive_area_fraction_candidates:
        _validate_unit_interval(candidate, "min_patient_positive_area_fraction_candidates")
    for candidate in min_component_area_fraction_patch_candidates:
        _validate_unit_interval(candidate, "min_component_area_fraction_patch_candidates")
    if 0.0 not in min_patient_positive_area_fraction_candidates:
        raise ValueError(
            "min_patient_positive_area_fraction_candidates must include 0.0 for the baseline."
        )
    if 0.0 not in min_component_area_fraction_patch_candidates:
        raise ValueError(
            "min_component_area_fraction_patch_candidates must include 0.0 for the baseline."
        )


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
    optimization_cache_max_bytes: int = DEFAULT_OPTIMIZATION_CACHE_MAX_BYTES
    local_shard_cache_max_bytes: int = 0
    runtime_normalization_method: str = "NOT_NORMALIZED"
    runtime_vahadane_backend: str = "fixed_source"
    log_folder: Path = Path("logs")
    log_file_name: str = "ensemble_optimizer.log"
    decision_threshold_min: float = DEFAULT_DECISION_THRESHOLD_MIN
    decision_threshold_max: float = DEFAULT_DECISION_THRESHOLD_MAX
    decision_threshold_step: float = DEFAULT_DECISION_THRESHOLD_STEP
    pos_dice_drop_tolerance: float = DEFAULT_POS_DICE_DROP_TOLERANCE
    pos_tpr_drop_tolerance: float = DEFAULT_POS_TPR_DROP_TOLERANCE
    min_micro_dice: float = DEFAULT_MIN_MICRO_DICE
    negative_clean_target: float = DEFAULT_NEGATIVE_CLEAN_TARGET
    min_macro_precision: float = DEFAULT_MIN_MACRO_PRECISION
    min_macro_tpr: float = DEFAULT_MIN_MACRO_TPR
    min_component_area_px_candidates: tuple[int, ...] = DEFAULT_MIN_COMPONENT_AREA_PX_CANDIDATES
    min_patient_positive_patches_candidates: tuple[int, ...] = (
        DEFAULT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES
    )
    min_patient_positive_area_fraction_candidates: tuple[float, ...] = (
        DEFAULT_MIN_PATIENT_POSITIVE_AREA_FRACTION_CANDIDATES
    )
    min_component_area_fraction_patch_candidates: tuple[float, ...] = (
        DEFAULT_MIN_COMPONENT_AREA_FRACTION_PATCH_CANDIDATES
    )
    negative_fp_penalty_lambda: float = DEFAULT_NEGATIVE_FP_PENALTY_LAMBDA
    roi_negative_area_penalty_lambda: float = DEFAULT_ROI_NEGATIVE_AREA_PENALTY_LAMBDA

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name

    def __post_init__(self) -> None:
        _validate_disjoint_architecture_groups(
            self.semantic_architectures,
            self.spatial_architectures,
        )
        if not 0.0 <= self.decision_threshold_min <= 1.0:
            raise ValueError("decision_threshold_min must be within [0, 1].")
        if not 0.0 <= self.decision_threshold_max <= 1.0:
            raise ValueError("decision_threshold_max must be within [0, 1].")
        if self.decision_threshold_min > self.decision_threshold_max:
            raise ValueError("decision_threshold_min cannot exceed decision_threshold_max.")
        if self.decision_threshold_step <= 0.0:
            raise ValueError("decision_threshold_step must be greater than 0.")
        _validate_non_negative_float(self.spill_penalty_lambda, "spill_penalty_lambda")
        _validate_non_negative_float(
            self.negative_fp_penalty_lambda,
            "negative_fp_penalty_lambda",
        )
        _validate_non_negative_float(
            self.roi_negative_area_penalty_lambda,
            "roi_negative_area_penalty_lambda",
        )
        _validate_rule6_drop_tolerances(
            pos_dice_drop_tolerance=self.pos_dice_drop_tolerance,
            pos_tpr_drop_tolerance=self.pos_tpr_drop_tolerance,
        )
        _validate_rule6_targets(
            min_micro_dice=self.min_micro_dice,
            negative_clean_target=self.negative_clean_target,
            min_macro_precision=self.min_macro_precision,
            min_macro_tpr=self.min_macro_tpr,
        )
        _validate_rule6_integer_candidates(
            min_component_area_px_candidates=self.min_component_area_px_candidates,
            min_patient_positive_patches_candidates=self.min_patient_positive_patches_candidates,
        )
        _validate_rule6_fraction_candidates(
            min_patient_positive_area_fraction_candidates=(
                self.min_patient_positive_area_fraction_candidates
            ),
            min_component_area_fraction_patch_candidates=(
                self.min_component_area_fraction_patch_candidates
            ),
        )


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
        local_shard_cache_max_bytes=_parse_non_negative_int(
            values.get("ENSEMBLE_OPT_LOCAL_SHARD_CACHE_MAX_BYTES"),
            default=0,
            variable_name="ENSEMBLE_OPT_LOCAL_SHARD_CACHE_MAX_BYTES",
        ),
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
        negative_fp_penalty_lambda=_parse_float(
            values.get("ENSEMBLE_OPT_NEGATIVE_FP_PENALTY_LAMBDA"),
            default=DEFAULT_NEGATIVE_FP_PENALTY_LAMBDA,
        ),
        roi_negative_area_penalty_lambda=_parse_float(
            values.get("ENSEMBLE_OPT_ROI_NEGATIVE_AREA_PENALTY_LAMBDA"),
            default=DEFAULT_ROI_NEGATIVE_AREA_PENALTY_LAMBDA,
        ),
        spatial_patient_policy=_parse_choice(
            values.get("ENSEMBLE_OPT_SPATIAL_PATIENT_POLICY"),
            default="all",
            valid=VALID_SPATIAL_PATIENT_POLICIES,
        ),
        num_trials_semantic=_parse_int(values.get("ENSEMBLE_OPT_NUM_TRIALS_SEMANTIC"), default=50),
        num_trials_spatial=_parse_int(values.get("ENSEMBLE_OPT_NUM_TRIALS_SPATIAL"), default=50),
        decision_threshold_min=_parse_float(
            values.get("ENSEMBLE_OPT_DECISION_THRESHOLD_MIN"),
            default=DEFAULT_DECISION_THRESHOLD_MIN,
        ),
        decision_threshold_max=_parse_float(
            values.get("ENSEMBLE_OPT_DECISION_THRESHOLD_MAX"),
            default=DEFAULT_DECISION_THRESHOLD_MAX,
        ),
        decision_threshold_step=_parse_float(
            values.get("ENSEMBLE_OPT_DECISION_THRESHOLD_STEP"),
            default=DEFAULT_DECISION_THRESHOLD_STEP,
        ),
        pos_dice_drop_tolerance=_parse_float(
            values.get("ENSEMBLE_OPT_POS_DICE_DROP_TOLERANCE"),
            default=DEFAULT_POS_DICE_DROP_TOLERANCE,
        ),
        pos_tpr_drop_tolerance=_parse_float(
            values.get("ENSEMBLE_OPT_POS_TPR_DROP_TOLERANCE"),
            default=DEFAULT_POS_TPR_DROP_TOLERANCE,
        ),
        min_micro_dice=_parse_float(
            values.get("ENSEMBLE_OPT_MIN_MICRO_DICE"),
            default=DEFAULT_MIN_MICRO_DICE,
        ),
        negative_clean_target=_parse_float(
            values.get("ENSEMBLE_OPT_NEGATIVE_CLEAN_TARGET"),
            default=DEFAULT_NEGATIVE_CLEAN_TARGET,
        ),
        min_macro_precision=_parse_float(
            values.get("ENSEMBLE_OPT_MIN_MACRO_PRECISION"),
            default=DEFAULT_MIN_MACRO_PRECISION,
        ),
        min_macro_tpr=_parse_float(
            values.get("ENSEMBLE_OPT_MIN_MACRO_TPR"),
            default=DEFAULT_MIN_MACRO_TPR,
        ),
        min_component_area_px_candidates=_parse_int_tuple(
            values.get("ENSEMBLE_OPT_MIN_COMPONENT_AREA_PX_CANDIDATES"),
            default=DEFAULT_MIN_COMPONENT_AREA_PX_CANDIDATES,
            variable_name="ENSEMBLE_OPT_MIN_COMPONENT_AREA_PX_CANDIDATES",
        ),
        min_patient_positive_patches_candidates=_parse_int_tuple(
            values.get("ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES"),
            default=DEFAULT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES,
            variable_name="ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES",
        ),
        min_patient_positive_area_fraction_candidates=_parse_float_tuple(
            values.get("ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_AREA_FRACTION_CANDIDATES"),
            default=DEFAULT_MIN_PATIENT_POSITIVE_AREA_FRACTION_CANDIDATES,
            variable_name="ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_AREA_FRACTION_CANDIDATES",
        ),
        min_component_area_fraction_patch_candidates=_parse_float_tuple(
            values.get("ENSEMBLE_OPT_MIN_COMPONENT_AREA_FRACTION_PATCH_CANDIDATES"),
            default=DEFAULT_MIN_COMPONENT_AREA_FRACTION_PATCH_CANDIDATES,
            variable_name="ENSEMBLE_OPT_MIN_COMPONENT_AREA_FRACTION_PATCH_CANDIDATES",
        ),
        optimization_cache_max_bytes=_parse_non_negative_int(
            values.get("ENSEMBLE_OPT_OPTIMIZATION_CACHE_MAX_BYTES"),
            default=DEFAULT_OPTIMIZATION_CACHE_MAX_BYTES,
            variable_name="ENSEMBLE_OPT_OPTIMIZATION_CACHE_MAX_BYTES",
        ),
        runtime_normalization_method=parse_runtime_normalization_method(
            values.get(RUNTIME_NORMALIZATION_METHOD_ENV_VAR)
        ),
        runtime_vahadane_backend=parse_runtime_vahadane_backend(
            values.get(RUNTIME_VAHADANE_BACKEND_ENV_VAR)
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("ENSEMBLE_OPT_LOG_FOLDER",),
        ),
        log_file_name=(values.get("ENSEMBLE_OPT_LOG_FILE") or "ensemble_optimizer.log").strip(),
    )
