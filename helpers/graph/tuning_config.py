from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
class GraphTuningConfig:
    """Runtime configuration for `4_2_tune_graph_method.py`."""

    source_image_folder: Path
    source_mask_folder: Path
    review_base_dir: Path
    log_folder: Path
    log_file_name: str
    output_params_path: Path
    test_set_size: float
    n_splits_inner_cv: int
    n_bayesian_calls: int
    n_initial_points: int
    random_state: int
    bg_intensity_range: tuple[int, int]
    k_range: tuple[int, int]
    min_size_range: tuple[int, int]
    erosion_range: tuple[int, int]

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_graph_tuning_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> GraphTuningConfig:
    """Load and validate Stage 4.2 graph tuning configuration."""

    values = env if env is not None else os.environ
    return GraphTuningConfig(
        source_image_folder=_required_path(
            values,
            "GRAPH_TUNING_SOURCE_IMAGE_FOLDER",
            system_name=system_name,
        ),
        source_mask_folder=_required_path(
            values,
            "GRAPH_TUNING_SOURCE_MASK_FOLDER",
            system_name=system_name,
        ),
        review_base_dir=_required_path(values, "GRAPH_TUNING_BASE_DIR", system_name=system_name),
        log_folder=_path_with_default(
            values,
            "GRAPH_TUNING_LOG_FOLDER",
            "./logs",
            system_name=system_name,
        ),
        log_file_name=_string_with_default(
            values,
            "GRAPH_TUNING_LOG_FILE",
            "bayesian_optimization.log",
        ),
        output_params_path=_path_with_default(
            values,
            "GRAPH_TUNING_OUTPUT_PARAMS_PATH",
            "./graph_cleaning_params.json",
            system_name=system_name,
        ),
        test_set_size=_parse_float(
            values.get("GRAPH_TUNING_TEST_SET_SIZE"),
            "GRAPH_TUNING_TEST_SET_SIZE",
            default=0.2,
        ),
        n_splits_inner_cv=_parse_int(
            values.get("GRAPH_TUNING_N_SPLITS_INNER_CV"),
            "GRAPH_TUNING_N_SPLITS_INNER_CV",
            default=3,
        ),
        n_bayesian_calls=_parse_int(
            values.get("GRAPH_TUNING_N_BAYESIAN_CALLS"),
            "GRAPH_TUNING_N_BAYESIAN_CALLS",
            default=50,
        ),
        n_initial_points=_parse_int(
            values.get("GRAPH_TUNING_N_INITIAL_POINTS"),
            "GRAPH_TUNING_N_INITIAL_POINTS",
            default=10,
        ),
        random_state=_parse_int(
            values.get("GRAPH_TUNING_RANDOM_STATE"),
            "GRAPH_TUNING_RANDOM_STATE",
            default=42,
        ),
        bg_intensity_range=(
            _parse_int(
                values.get("GRAPH_TUNING_BG_INTENSITY_MIN"),
                "GRAPH_TUNING_BG_INTENSITY_MIN",
                default=100,
            ),
            _parse_int(
                values.get("GRAPH_TUNING_BG_INTENSITY_MAX"),
                "GRAPH_TUNING_BG_INTENSITY_MAX",
                default=250,
            ),
        ),
        k_range=(
            _parse_int(values.get("GRAPH_TUNING_K_MIN"), "GRAPH_TUNING_K_MIN", default=100),
            _parse_int(values.get("GRAPH_TUNING_K_MAX"), "GRAPH_TUNING_K_MAX", default=500),
        ),
        min_size_range=(
            _parse_int(
                values.get("GRAPH_TUNING_MIN_SIZE_MIN"),
                "GRAPH_TUNING_MIN_SIZE_MIN",
                default=10,
            ),
            _parse_int(
                values.get("GRAPH_TUNING_MIN_SIZE_MAX"),
                "GRAPH_TUNING_MIN_SIZE_MAX",
                default=200,
            ),
        ),
        erosion_range=(
            _parse_int(
                values.get("GRAPH_TUNING_EROSION_MIN"),
                "GRAPH_TUNING_EROSION_MIN",
                default=0,
            ),
            _parse_int(
                values.get("GRAPH_TUNING_EROSION_MAX"),
                "GRAPH_TUNING_EROSION_MAX",
                default=10,
            ),
        ),
    )
