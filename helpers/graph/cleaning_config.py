from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.graph.contamination import GraphContaminationParameters
from helpers.graph.parameter_store import load_graph_cleaning_parameter_artifact
from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


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

    source_hdf5_path: Path
    output_base_dir: Path
    master_manifest_path: Path
    log_folder: Path
    log_file_name: str
    params_path: Path
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
    source_hdf5_path = _required_path(
        values,
        "GRAPH_CLEANING_SOURCE_HDF5_PATH",
        system_name=system_name,
    )
    params_path = _required_path(
        values,
        "GRAPH_CLEANING_PARAMS_PATH",
        system_name=system_name,
    )
    artifact = load_graph_cleaning_parameter_artifact(params_path)
    master_manifest_path = resolve_env_path(
        values.get("GRAPH_CLEANING_MASTER_MANIFEST_PATH"),
        "GRAPH_CLEANING_MASTER_MANIFEST_PATH",
        system_name=system_name,
    )
    if master_manifest_path is None:
        master_manifest_path = source_hdf5_path.parents[2] / "master_manifest.sqlite"
    return GraphCleaningConfig(
        source_hdf5_path=source_hdf5_path,
        output_base_dir=_required_path(
            values,
            "GRAPH_CLEANING_OUTPUT_BASE_DIR",
            system_name=system_name,
        ),
        master_manifest_path=master_manifest_path,
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
            int(values.get("GRAPH_CLEANING_NUM_WORKERS") or (os.cpu_count() or 1)),
        ),
        graph_params=artifact.graph_params,
        tau=artifact.tau,
    )
