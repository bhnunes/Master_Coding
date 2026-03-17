from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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


def _parse_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return Path(stripped).expanduser()


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
    artifact_policy_path: Path | None

    @property
    def table_name(self) -> str:
        return f"DATABASE_{self.tag}"

    @property
    def patch_base_path(self) -> Path:
        return self.base_path / "PATCHES"


def load_database_manager_config(
    env: Mapping[str, str | None] | None = None,
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

    database_path_value = values.get("SQLITE_DB_PATH")
    if not database_path_value:
        raise ValueError("The 'SQLITE_DB_PATH' environment variable is required.")

    return DatabaseManagerConfig(
        tag=tag,
        database_path=Path(database_path_value).expanduser(),
        base_path=Path(values.get("PROJECTS_BASE_PATH") or "./projects").expanduser(),
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
        use_advanced_artifact_filtering=_parse_bool(values.get("USE_ADVANCED_ARTIFACT_FILTERING")),
        activate_sanity_check_geojson=_parse_bool(values.get("ACTIVATE_SANITY_CHECK_GEOJSON")),
        geojson_path=_parse_optional_path(values.get("GEOJSON_PATH")),
        artifact_policy_path=_parse_optional_path(values.get("ARTIFACT_POLICY_PATH")),
    )
