from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


def _parse_bool(value: str | None, variable_name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"The '{variable_name}' environment variable must be a boolean value.")


def _parse_positive_int(value: str | None, variable_name: str, default: int) -> int:
    candidate = default if value is None or value == "" else int(value)
    if candidate <= 0:
        raise ValueError(f"The '{variable_name}' environment variable must be greater than zero.")
    return candidate


def _parse_positive_float(value: str | None, variable_name: str, default: float) -> float:
    candidate = default if value is None or value == "" else float(value)
    if candidate <= 0:
        raise ValueError(f"The '{variable_name}' environment variable must be greater than zero.")
    return candidate


def _parse_probability(value: str | None, variable_name: str, default: float) -> float:
    candidate = default if value is None or value == "" else float(value)
    if not 0.0 <= candidate <= 1.0:
        raise ValueError(f"The '{variable_name}' environment variable must be between 0 and 1.")
    return candidate


def _required_path(value: str | None, variable_name: str) -> Path:
    path = resolve_env_path(value, variable_name, required=True)
    assert path is not None
    return path


@dataclass(frozen=True)
class SmartSamplerConfig:
    source_h5_path: Path
    output_dir: Path
    output_filename: str
    local_work_dir: Path | None
    stage_input_locally: bool
    stage_outputs_locally: bool
    clean_local_work_dir: bool
    write_sidecars: bool
    overwrite_output: bool
    model_name: str
    batch_size: int
    device: str
    n_start: int
    n_max: int
    growth_factor: float
    stability_threshold: float
    stability_repeats: int
    max_steps: int
    intersection_ratio_threshold: float
    k_min: int
    k_max: int
    adaptive_keep_enabled: bool
    keep_min: int
    keep_step: int
    keep_improvement_threshold: float
    keep_patience: int
    m_max: int
    seed: int
    num_workers: int
    use_gist: bool = False
    protect_positive_labels: bool = True
    protect_mask_positive: bool = True
    positive_mask_fraction_threshold: float = 0.0
    log_folder: Path = Path("logs")
    log_file_name: str = "smart_sampler.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_smart_sampler_config(
    env: Mapping[str, str] | os._Environ[str] | None = None,
) -> SmartSamplerConfig:
    values = env if env is not None else os.environ
    use_gist_value = values.get("SMART_SAMPLER_USE_GIST")
    if use_gist_value is None:
        use_gist_value = values.get("USE_GIST_SCRIPT")
    source_h5_path = _required_path(
        values.get("SMART_SAMPLER_SOURCE_H5"), "SMART_SAMPLER_SOURCE_H5"
    )
    output_dir = _required_path(values.get("SMART_SAMPLER_OUTPUT_DIR"), "SMART_SAMPLER_OUTPUT_DIR")
    local_work_dir = resolve_env_path(
        values.get("SMART_SAMPLER_LOCAL_WORK_DIR"), "SMART_SAMPLER_LOCAL_WORK_DIR"
    )
    device = (
        values.get("SMART_SAMPLER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    ).strip()
    growth_factor = _parse_positive_float(
        values.get("SMART_SAMPLER_GROWTH_FACTOR"),
        "SMART_SAMPLER_GROWTH_FACTOR",
        2.0,
    )
    if growth_factor <= 1.0:
        raise ValueError(
            "The 'SMART_SAMPLER_GROWTH_FACTOR' environment variable must be greater than 1.0."
        )

    return SmartSamplerConfig(
        source_h5_path=source_h5_path,
        output_dir=output_dir,
        output_filename=(
            values.get("SMART_SAMPLER_OUTPUT_FILENAME") or "TRAIN_FILTERED.h5"
        ).strip(),
        local_work_dir=local_work_dir,
        stage_input_locally=_parse_bool(
            values.get("SMART_SAMPLER_STAGE_INPUT_LOCALLY"),
            "SMART_SAMPLER_STAGE_INPUT_LOCALLY",
            False,
        ),
        stage_outputs_locally=_parse_bool(
            values.get("SMART_SAMPLER_STAGE_OUTPUTS_LOCALLY"),
            "SMART_SAMPLER_STAGE_OUTPUTS_LOCALLY",
            False,
        ),
        clean_local_work_dir=_parse_bool(
            values.get("SMART_SAMPLER_CLEAN_LOCAL_WORK_DIR"),
            "SMART_SAMPLER_CLEAN_LOCAL_WORK_DIR",
            True,
        ),
        write_sidecars=_parse_bool(
            values.get("SMART_SAMPLER_WRITE_SIDECARS"),
            "SMART_SAMPLER_WRITE_SIDECARS",
            True,
        ),
        overwrite_output=_parse_bool(
            values.get("SMART_SAMPLER_OVERWRITE_OUTPUT"),
            "SMART_SAMPLER_OVERWRITE_OUTPUT",
            False,
        ),
        model_name=(values.get("SMART_SAMPLER_MODEL_NAME") or "owkin/phikon-v2").strip(),
        batch_size=_parse_positive_int(
            values.get("SMART_SAMPLER_BATCH_SIZE"),
            "SMART_SAMPLER_BATCH_SIZE",
            128,
        ),
        device=device,
        n_start=_parse_positive_int(
            values.get("SMART_SAMPLER_N_START"), "SMART_SAMPLER_N_START", 512
        ),
        n_max=_parse_positive_int(values.get("SMART_SAMPLER_N_MAX"), "SMART_SAMPLER_N_MAX", 15000),
        growth_factor=growth_factor,
        stability_threshold=_parse_probability(
            values.get("SMART_SAMPLER_STABILITY_THRESHOLD"),
            "SMART_SAMPLER_STABILITY_THRESHOLD",
            0.85,
        ),
        stability_repeats=_parse_positive_int(
            values.get("SMART_SAMPLER_STABILITY_REPEATS"),
            "SMART_SAMPLER_STABILITY_REPEATS",
            3,
        ),
        max_steps=_parse_positive_int(
            values.get("SMART_SAMPLER_MAX_STEPS"),
            "SMART_SAMPLER_MAX_STEPS",
            7,
        ),
        intersection_ratio_threshold=_parse_probability(
            values.get("SMART_SAMPLER_INTERSECTION_RATIO_THRESHOLD"),
            "SMART_SAMPLER_INTERSECTION_RATIO_THRESHOLD",
            0.2,
        ),
        k_min=_parse_positive_int(values.get("SMART_SAMPLER_K_MIN"), "SMART_SAMPLER_K_MIN", 20),
        k_max=_parse_positive_int(values.get("SMART_SAMPLER_K_MAX"), "SMART_SAMPLER_K_MAX", 80),
        adaptive_keep_enabled=_parse_bool(
            values.get("SMART_SAMPLER_ADAPTIVE_KEEP_ENABLED"),
            "SMART_SAMPLER_ADAPTIVE_KEEP_ENABLED",
            True,
        ),
        keep_min=_parse_positive_int(
            values.get("SMART_SAMPLER_KEEP_MIN"),
            "SMART_SAMPLER_KEEP_MIN",
            64,
        ),
        keep_step=_parse_positive_int(
            values.get("SMART_SAMPLER_KEEP_STEP"),
            "SMART_SAMPLER_KEEP_STEP",
            64,
        ),
        keep_improvement_threshold=_parse_probability(
            values.get("SMART_SAMPLER_KEEP_IMPROVEMENT_THRESHOLD"),
            "SMART_SAMPLER_KEEP_IMPROVEMENT_THRESHOLD",
            0.02,
        ),
        keep_patience=_parse_positive_int(
            values.get("SMART_SAMPLER_KEEP_PATIENCE"),
            "SMART_SAMPLER_KEEP_PATIENCE",
            2,
        ),
        m_max=_parse_positive_int(values.get("SMART_SAMPLER_M_MAX"), "SMART_SAMPLER_M_MAX", 2000),
        seed=_parse_positive_int(values.get("SMART_SAMPLER_SEED"), "SMART_SAMPLER_SEED", 42),
        num_workers=max(
            0,
            int(values.get("SMART_SAMPLER_NUM_WORKERS") or 2),
        ),
        use_gist=_parse_bool(
            use_gist_value,
            "SMART_SAMPLER_USE_GIST",
            False,
        ),
        protect_positive_labels=_parse_bool(
            values.get("SMART_SAMPLER_PROTECT_POSITIVE_LABELS"),
            "SMART_SAMPLER_PROTECT_POSITIVE_LABELS",
            True,
        ),
        protect_mask_positive=_parse_bool(
            values.get("SMART_SAMPLER_PROTECT_MASK_POSITIVE"),
            "SMART_SAMPLER_PROTECT_MASK_POSITIVE",
            True,
        ),
        positive_mask_fraction_threshold=_parse_probability(
            values.get("SMART_SAMPLER_POSITIVE_MASK_FRACTION_THRESHOLD"),
            "SMART_SAMPLER_POSITIVE_MASK_FRACTION_THRESHOLD",
            0.0,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=None,
            fallback_names=("SMART_SAMPLER_LOG_FOLDER",),
        ),
        log_file_name=(values.get("SMART_SAMPLER_LOG_FILE") or "smart_sampler.log").strip(),
    )
