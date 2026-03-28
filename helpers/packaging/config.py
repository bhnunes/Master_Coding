from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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


def _required_path(variable_value: str | None, variable_name: str) -> Path:
    path = resolve_env_path(variable_value, variable_name, required=True)
    assert path is not None
    return path


@dataclass(frozen=True)
class PackagingConfig:
    source_hdf5_path: Path
    output_dir: Path
    output_filename: str
    overwrite_outputs: bool
    accepted_manifest_path: Path | None = None
    log_folder: Path = Path("logs")
    log_file_name: str = "packaging.log"

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def _infer_default_accepted_manifest_path(
    *,
    source_hdf5_path: Path,
) -> Path | None:
    candidate_paths: list[Path] = []
    if source_hdf5_path.suffix.lower() == ".h5":
        candidate_paths.append(source_hdf5_path.parent / "accepted_manifest.csv")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def load_packaging_config(
    env: Mapping[str, str] | os._Environ[str] | None = None,
) -> PackagingConfig:
    values = env if env is not None else os.environ
    source_hdf5_path = _required_path(
        values.get("PACKAGING_SOURCE_HDF5_PATH"),
        "PACKAGING_SOURCE_HDF5_PATH",
    )
    accepted_manifest_path = resolve_env_path(
        values.get("PACKAGING_ACCEPTED_MANIFEST_PATH"),
        "PACKAGING_ACCEPTED_MANIFEST_PATH",
    )
    default_output_dir = source_hdf5_path.parent
    output_dir = (
        resolve_env_path(values.get("PACKAGING_OUTPUT_DIR"), "PACKAGING_OUTPUT_DIR")
        or default_output_dir
    )
    accepted_manifest_path = accepted_manifest_path or _infer_default_accepted_manifest_path(
        source_hdf5_path=source_hdf5_path,
    )
    return PackagingConfig(
        source_hdf5_path=source_hdf5_path,
        accepted_manifest_path=accepted_manifest_path,
        output_dir=output_dir,
        output_filename=(values.get("PACKAGING_OUTPUT_FILENAME") or "SOURCE_DATASET.h5").strip(),
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
