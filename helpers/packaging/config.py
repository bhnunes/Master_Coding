from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


def _parse_positive_int(value: str | None, variable_name: str, default: int) -> int:
    if value is None or value == "":
        candidate = default
    else:
        candidate = int(value)
    if candidate <= 0:
        raise ValueError(f"The '{variable_name}' environment variable must be greater than zero.")
    return candidate


def _parse_bool(value: str | None, variable_name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"The '{variable_name}' environment variable must be a boolean value.")


def _parse_splits(value: str | None) -> tuple[str, ...]:
    del value
    raise ValueError("Stage 7 no longer accepts split-folder packaging input.")


def _required_path(variable_value: str | None, variable_name: str) -> Path:
    path = resolve_env_path(variable_value, variable_name, required=True)
    assert path is not None
    return path


@dataclass(frozen=True)
class PackagingConfig:
    base_dir: Path
    output_dir: Path
    output_filename: str
    img_size: int
    patient_id_regex: str
    overwrite_outputs: bool
    log_folder: Path = Path("logs")
    log_file_name: str = "packaging.log"

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_packaging_config(
    env: Mapping[str, str] | os._Environ[str] | None = None,
) -> PackagingConfig:
    values = env if env is not None else os.environ
    base_dir = _required_path(values.get("PACKAGING_BASE_DIR"), "PACKAGING_BASE_DIR")
    output_dir = (
        resolve_env_path(values.get("PACKAGING_OUTPUT_DIR"), "PACKAGING_OUTPUT_DIR") or base_dir
    )
    patient_id_regex = values.get("PACKAGING_PATIENT_ID_REGEX", r"PATIENT_(\d+)_")
    return PackagingConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        output_filename=(values.get("PACKAGING_OUTPUT_FILENAME") or "SOURCE_DATASET.h5").strip(),
        img_size=_parse_positive_int(values.get("PACKAGING_IMG_SIZE"), "PACKAGING_IMG_SIZE", 224),
        patient_id_regex=patient_id_regex,
        overwrite_outputs=_parse_bool(
            values.get("PACKAGING_OVERWRITE_OUTPUTS"),
            "PACKAGING_OVERWRITE_OUTPUTS",
            False,
        ),
        log_folder=resolve_log_folder(
            values,
            fallback_names=("PACKAGING_LOG_FOLDER",),
        ),
        log_file_name=(values.get("PACKAGING_LOG_FILE") or "packaging.log").strip(),
    )
