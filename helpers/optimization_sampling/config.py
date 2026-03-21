from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


def _parse_int(value: str | None, variable_name: str, default: int | None = None) -> int:
    candidate = value if value not in {None, ""} else default
    if candidate is None:
        raise ValueError(f"The '{variable_name}' environment variable is required.")
    return int(candidate)


def _parse_float(value: str | None, variable_name: str, default: float | None = None) -> float:
    candidate = value if value not in {None, ""} else default
    if candidate is None:
        raise ValueError(f"The '{variable_name}' environment variable is required.")
    return float(candidate)


def _required_path(
    environment: Mapping[str, str | None],
    variable_name: str,
    *,
    system_name: str | None = None,
) -> Path:
    path = resolve_env_path(
        environment.get(variable_name),
        variable_name,
        system_name=system_name,
        required=True,
    )
    assert path is not None
    return path


def _path_with_default(
    environment: Mapping[str, str | None],
    variable_name: str,
    default: str,
    *,
    system_name: str | None = None,
) -> Path:
    path = resolve_env_path(
        environment.get(variable_name) or default,
        variable_name,
        system_name=system_name,
        required=True,
    )
    assert path is not None
    return path


def _string_with_default(
    environment: Mapping[str, str | None], variable_name: str, default: str
) -> str:
    value = environment.get(variable_name)
    return value if value else default


@dataclass(frozen=True)
class OptimizationSamplingConfig:
    """Runtime configuration for `4_1_optimization_sampling.py`."""

    image_folder: Path
    mask_folder: Path
    output_base: Path
    log_folder: Path
    log_file_name: str
    confidence_level: float
    margin_of_error: float
    proportion: float
    pilot_sample_size: int
    master_pool_fraction: float
    overlay_color: tuple[int, int, int]
    overlay_thickness: int
    overlay_alpha: float
    num_processes: int

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_optimization_sampling_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> OptimizationSamplingConfig:
    """Load and validate Stage 4 optimization sampling configuration."""

    values = env if env is not None else os.environ
    num_processes = max(
        1,
        _parse_int(
            values.get("OPTIMIZATION_SAMPLING_NUM_PROCESSES"),
            "OPTIMIZATION_SAMPLING_NUM_PROCESSES",
            default=os.cpu_count() or 1,
        ),
    )

    return OptimizationSamplingConfig(
        image_folder=_required_path(
            values,
            "OPTIMIZATION_SAMPLING_IMAGE_FOLDER",
            system_name=system_name,
        ),
        mask_folder=_required_path(
            values,
            "OPTIMIZATION_SAMPLING_MASK_FOLDER",
            system_name=system_name,
        ),
        output_base=_required_path(
            values,
            "OPTIMIZATION_SAMPLING_OUTPUT_BASE",
            system_name=system_name,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("OPTIMIZATION_SAMPLING_LOG_FOLDER",),
        ),
        log_file_name=_string_with_default(
            values,
            "OPTIMIZATION_SAMPLING_LOG_FILE",
            "optimization_sampling.log",
        ),
        confidence_level=_parse_float(
            values.get("OPTIMIZATION_SAMPLING_CONFIDENCE_LEVEL"),
            "OPTIMIZATION_SAMPLING_CONFIDENCE_LEVEL",
            default=0.95,
        ),
        margin_of_error=_parse_float(
            values.get("OPTIMIZATION_SAMPLING_MARGIN_OF_ERROR"),
            "OPTIMIZATION_SAMPLING_MARGIN_OF_ERROR",
            default=0.05,
        ),
        proportion=_parse_float(
            values.get("OPTIMIZATION_SAMPLING_PROPORTION"),
            "OPTIMIZATION_SAMPLING_PROPORTION",
            default=0.5,
        ),
        pilot_sample_size=_parse_int(
            values.get("OPTIMIZATION_SAMPLING_PILOT_SAMPLE_SIZE"),
            "OPTIMIZATION_SAMPLING_PILOT_SAMPLE_SIZE",
            default=100,
        ),
        master_pool_fraction=_parse_float(
            values.get("OPTIMIZATION_SAMPLING_MASTER_POOL_FRACTION"),
            "OPTIMIZATION_SAMPLING_MASTER_POOL_FRACTION",
            default=0.10,
        ),
        overlay_color=(
            _parse_int(
                values.get("OPTIMIZATION_SAMPLING_OVERLAY_COLOR_B"),
                "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_B",
                default=0,
            ),
            _parse_int(
                values.get("OPTIMIZATION_SAMPLING_OVERLAY_COLOR_G"),
                "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_G",
                default=0,
            ),
            _parse_int(
                values.get("OPTIMIZATION_SAMPLING_OVERLAY_COLOR_R"),
                "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_R",
                default=255,
            ),
        ),
        overlay_thickness=_parse_int(
            values.get("OPTIMIZATION_SAMPLING_OVERLAY_THICKNESS"),
            "OPTIMIZATION_SAMPLING_OVERLAY_THICKNESS",
            default=2,
        ),
        overlay_alpha=_parse_float(
            values.get("OPTIMIZATION_SAMPLING_OVERLAY_ALPHA"),
            "OPTIMIZATION_SAMPLING_OVERLAY_ALPHA",
            default=1.0,
        ),
        num_processes=num_processes,
    )
