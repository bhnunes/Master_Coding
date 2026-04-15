from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path

VALID_HDF5_COMPRESSION = ("NONE", "LZF", "GZIP")


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


def _parse_choice(value: str | None, variable_name: str, default: str) -> str:
    candidate = (value or default).strip().upper()
    if candidate not in VALID_HDF5_COMPRESSION:
        choices = ", ".join(VALID_HDF5_COMPRESSION)
        raise ValueError(f"The '{variable_name}' environment variable must be one of: {choices}.")
    return candidate


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


@dataclass(frozen=True)
class PatientShardsConfig:
    stage5_base_dir: Path
    output_base_dir: Path
    overwrite_output: bool
    hdf5_compression: str
    copy_batch_size: int
    log_folder: Path = Path("logs")
    log_file_name: str = "patient_shards.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_patient_shards_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> PatientShardsConfig:
    values = env if env is not None else os.environ
    stage5_base_dir = _required_path(
        values,
        "PATIENT_SHARDS_STAGE5_BASE_DIR",
        system_name=system_name,
    )
    output_base_dir = resolve_env_path(
        values.get("PATIENT_SHARDS_OUTPUT_BASE_DIR"),
        "PATIENT_SHARDS_OUTPUT_BASE_DIR",
        system_name=system_name,
    )
    return PatientShardsConfig(
        stage5_base_dir=stage5_base_dir,
        output_base_dir=output_base_dir or stage5_base_dir,
        overwrite_output=_parse_bool(
            values.get("PATIENT_SHARDS_OVERWRITE_OUTPUT"),
            "PATIENT_SHARDS_OVERWRITE_OUTPUT",
            False,
        ),
        hdf5_compression=_parse_choice(
            values.get("PATIENT_SHARDS_HDF5_COMPRESSION"),
            "PATIENT_SHARDS_HDF5_COMPRESSION",
            "NONE",
        ),
        copy_batch_size=_parse_positive_int(
            values.get("PATIENT_SHARDS_COPY_BATCH_SIZE"),
            "PATIENT_SHARDS_COPY_BATCH_SIZE",
            1024,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("PATIENT_SHARDS_LOG_FOLDER",),
        ),
        log_file_name=(values.get("PATIENT_SHARDS_LOG_FILE") or "patient_shards.log").strip(),
    )
