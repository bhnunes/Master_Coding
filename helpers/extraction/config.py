from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "t", "yes", "y"}


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


def _parse_optional_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
) -> Path | None:
    return resolve_env_path(value, variable_name, system_name=system_name)


@dataclass(frozen=True)
class DatabaseManagerConfig:
    """Runtime configuration for `2_database_manager.py`."""

    tag: str
    source_folder: Path
    database_path: Path
    base_path: Path
    window_size: int
    stride: int
    match_percentage: float
    tissue_percentage: float
    target_level: int
    num_workers: int
    load_cases: bool
    use_advanced_artifact_filtering: bool
    activate_sanity_check_geojson: bool
    geojson_path: Path | None
    copy_wsi_to_local_cache: bool
    local_slide_cache_dir: Path | None
    log_folder: Path
    log_file_name: str

    @property
    def table_name(self) -> str:
        return f"DATABASE_{self.tag}"

    @property
    def patch_base_path(self) -> Path:
        return self.base_path / "PATCHES"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_database_manager_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> DatabaseManagerConfig:
    """Load and validate database manager configuration from environment values."""

    values = env if env is not None else os.environ
    tag = (values.get("TAG") or "").strip()
    if not tag:
        raise ValueError(
            "The 'TAG' environment variable is not set. Please define it in your .env file."
        )

    window_size = _parse_int(values.get("WINDOW_SIZE"), "WINDOW_SIZE", default=224)
    stride = _parse_int(values.get("STRIDE"), "STRIDE", default=window_size // 2)
    use_advanced_artifact_filtering = _parse_bool(
        values.get("USE_ADVANCED_ARTIFACT_FILTERING"),
        default=True,
    )
    activate_sanity_check_geojson = _parse_bool(values.get("ACTIVATE_SANITY_CHECK_GEOJSON"))
    copy_wsi_to_local_cache = _parse_bool(values.get("STAGE2_COPY_WSI_TO_LOCAL_CACHE"))

    database_path = resolve_env_path(
        values.get("SQLITE_DB_PATH"),
        "SQLITE_DB_PATH",
        system_name=system_name,
        required=True,
    )
    source_folder = resolve_env_path(
        values.get("SOURCE_FOLDER"),
        "SOURCE_FOLDER",
        system_name=system_name,
        required=True,
    )
    base_path = resolve_env_path(
        values.get("PROJECTS_BASE_PATH") or "./projects",
        "PROJECTS_BASE_PATH",
        system_name=system_name,
        required=True,
    )
    assert database_path is not None
    assert source_folder is not None
    assert base_path is not None
    local_slide_cache_dir = _parse_optional_path(
        values.get("STAGE2_LOCAL_SLIDE_CACHE_DIR"),
        "STAGE2_LOCAL_SLIDE_CACHE_DIR",
        system_name=system_name,
    )
    if copy_wsi_to_local_cache and local_slide_cache_dir is None:
        raise ValueError(
            "The 'STAGE2_LOCAL_SLIDE_CACHE_DIR' environment variable is required when "
            "'STAGE2_COPY_WSI_TO_LOCAL_CACHE=True'."
        )
    log_folder = resolve_log_folder(
        values,
        system_name=system_name,
        fallback_names=("EXTRACTION_LOG_FOLDER",),
    )

    return DatabaseManagerConfig(
        tag=tag,
        source_folder=source_folder,
        database_path=database_path,
        base_path=base_path,
        window_size=window_size,
        stride=stride,
        match_percentage=_parse_float(
            values.get("MATCH_PERCENTAGE"), "MATCH_PERCENTAGE", default=1.0
        ),
        tissue_percentage=_parse_float(
            values.get("TISSUE_PERCENTAGE"),
            "TISSUE_PERCENTAGE",
            default=0.3,
        ),
        target_level=_parse_int(values.get("TARGET_LEVEL"), "TARGET_LEVEL", default=0),
        num_workers=max(
            1, _parse_int(values.get("NUM_WORKERS"), "NUM_WORKERS", default=os.cpu_count() or 1)
        ),
        load_cases=_parse_bool(values.get("LOADCASES")),
        use_advanced_artifact_filtering=use_advanced_artifact_filtering,
        activate_sanity_check_geojson=activate_sanity_check_geojson,
        copy_wsi_to_local_cache=copy_wsi_to_local_cache,
        local_slide_cache_dir=local_slide_cache_dir,
        log_folder=log_folder,
        log_file_name=(values.get("EXTRACTION_LOG_FILE") or "database_manager.log").strip(),
        geojson_path=(
            source_folder / "GEOJSON"
            if use_advanced_artifact_filtering or activate_sanity_check_geojson
            else None
        ),
    )
