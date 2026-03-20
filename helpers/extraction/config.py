from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def table_name(self) -> str:
        return f"DATABASE_{self.tag}"

    @property
    def patch_base_path(self) -> Path:
        return self.base_path / "PATCHES"


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

    database_path = resolve_env_path(
        values.get("SQLITE_DB_PATH"),
        "SQLITE_DB_PATH",
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
    assert base_path is not None

    return DatabaseManagerConfig(
        tag=tag,
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
        geojson_path=(
            _parse_optional_path(
                values.get("GEOJSON_PATH"),
                "GEOJSON_PATH",
                system_name=system_name,
            )
            if use_advanced_artifact_filtering or activate_sanity_check_geojson
            else None
        ),
    )
