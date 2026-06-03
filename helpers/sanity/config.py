from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path

VALID_CHECKSUM_MODES = ("OFF", "SAMPLE", "FULL")


def _parse_bool(value: str | None, variable_name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"The '{variable_name}' environment variable must be a boolean value.")


def _parse_int(value: str | None, variable_name: str, default: int) -> int:
    del variable_name
    if value is None or value == "":
        return default
    return int(value)


def _parse_choice(value: str | None, variable_name: str, default: str) -> str:
    candidate = (value or default).strip().upper()
    if candidate not in VALID_CHECKSUM_MODES:
        choices = ", ".join(VALID_CHECKSUM_MODES)
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
class SanityConfig:
    base_dir: Path
    sample_pairs: int
    full_mask_scan: bool
    full_shape_scan: bool
    checksum_mode: str
    enforce_filename_uniqueness: bool
    enforce_regex_patient_id_match: bool
    enforce_split_stats_parity: bool
    fail_on_empty_cancer_mask: bool
    fail_on_positive_not_cancer_mask: bool
    log_folder: Path = Path("logs")
    log_file_name: str = "sanity_checks.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_sanity_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> SanityConfig:
    values = env if env is not None else os.environ
    return SanityConfig(
        base_dir=_required_path(values, "SANITY_BASE_DIR", system_name=system_name),
        sample_pairs=_parse_int(values.get("SANITY_SAMPLE_PAIRS"), "SANITY_SAMPLE_PAIRS", 1000),
        full_mask_scan=_parse_bool(
            values.get("SANITY_FULL_MASK_SCAN"),
            "SANITY_FULL_MASK_SCAN",
            False,
        ),
        full_shape_scan=_parse_bool(
            values.get("SANITY_FULL_SHAPE_SCAN"),
            "SANITY_FULL_SHAPE_SCAN",
            False,
        ),
        checksum_mode=_parse_choice(
            values.get("SANITY_CHECKSUM_MODE"),
            "SANITY_CHECKSUM_MODE",
            "SAMPLE",
        ),
        enforce_filename_uniqueness=_parse_bool(
            values.get("SANITY_ENFORCE_FILENAME_UNIQUENESS"),
            "SANITY_ENFORCE_FILENAME_UNIQUENESS",
            True,
        ),
        enforce_regex_patient_id_match=_parse_bool(
            values.get("SANITY_ENFORCE_REGEX_PATIENT_ID_MATCH"),
            "SANITY_ENFORCE_REGEX_PATIENT_ID_MATCH",
            True,
        ),
        enforce_split_stats_parity=_parse_bool(
            values.get("SANITY_ENFORCE_SPLIT_STATS_PARITY"),
            "SANITY_ENFORCE_SPLIT_STATS_PARITY",
            True,
        ),
        fail_on_empty_cancer_mask=_parse_bool(
            values.get("SANITY_FAIL_ON_EMPTY_CANCER_MASK"),
            "SANITY_FAIL_ON_EMPTY_CANCER_MASK",
            True,
        ),
        fail_on_positive_not_cancer_mask=_parse_bool(
            values.get("SANITY_FAIL_ON_POSITIVE_NOT_CANCER_MASK"),
            "SANITY_FAIL_ON_POSITIVE_NOT_CANCER_MASK",
            True,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("SANITY_LOG_FOLDER",),
        ),
        log_file_name=(values.get("SANITY_LOG_FILE") or "sanity_checks.log").strip(),
    )
