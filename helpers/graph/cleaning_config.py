from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.graph.contamination import GraphContaminationParameters
from helpers.graph.parameter_store import load_graph_cleaning_parameter_artifact
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
class GraphCleaningConfig:
    """Runtime configuration for `4_3_cleaner_script.py`."""

    source_image_dir: Path
    source_mask_dir: Path
    output_base_dir: Path
    log_folder: Path
    log_file_name: str
    params_path: Path | None
    num_workers: int
    graph_params: GraphContaminationParameters
    tau: float

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_graph_cleaning_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> GraphCleaningConfig:
    """Load and validate Stage 4.3 graph cleaning configuration."""

    values = env if env is not None else os.environ
    params_path = resolve_env_path(
        values.get("GRAPH_CLEANING_PARAMS_PATH"),
        "GRAPH_CLEANING_PARAMS_PATH",
        system_name=system_name,
    )
    artifact = (
        load_graph_cleaning_parameter_artifact(params_path) if params_path is not None else None
    )
    return GraphCleaningConfig(
        source_image_dir=_required_path(
            values,
            "GRAPH_CLEANING_SOURCE_IMAGE_DIR",
            system_name=system_name,
        ),
        source_mask_dir=_required_path(
            values,
            "GRAPH_CLEANING_SOURCE_MASK_DIR",
            system_name=system_name,
        ),
        output_base_dir=_required_path(
            values,
            "GRAPH_CLEANING_OUTPUT_BASE_DIR",
            system_name=system_name,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("GRAPH_CLEANING_LOG_FOLDER",),
        ),
        log_file_name=_string_with_default(
            values,
            "GRAPH_CLEANING_LOG_FILE",
            "production_filtering.log",
        ),
        params_path=params_path,
        num_workers=max(
            1,
            _parse_int(
                values.get("GRAPH_CLEANING_NUM_WORKERS"),
                "GRAPH_CLEANING_NUM_WORKERS",
                default=os.cpu_count() or 1,
            ),
        ),
        graph_params=(
            artifact.graph_params
            if artifact is not None
            else GraphContaminationParameters(
                bg_intensity_thresh=_parse_int(
                    values.get("GRAPH_CLEANING_BG_INTENSITY_THRESH"),
                    "GRAPH_CLEANING_BG_INTENSITY_THRESH",
                    default=198,
                ),
                k=_parse_float(values.get("GRAPH_CLEANING_K"), "GRAPH_CLEANING_K", default=386.0),
                min_size=_parse_int(
                    values.get("GRAPH_CLEANING_MIN_SIZE"),
                    "GRAPH_CLEANING_MIN_SIZE",
                    default=200,
                ),
                erosion_px=_parse_int(
                    values.get("GRAPH_CLEANING_EROSION_PX"),
                    "GRAPH_CLEANING_EROSION_PX",
                    default=0,
                ),
            )
        ),
        tau=(
            artifact.tau
            if artifact is not None
            else _parse_float(values.get("GRAPH_CLEANING_TAU"), "GRAPH_CLEANING_TAU", default=0.24)
        ),
    )
