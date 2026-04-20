from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path

VALID_NORMALIZATION_METHODS = (
    "NOT_NORMALIZED",
    "REINHARD",
    "RUIFROK",
    "MACENKO",
    "VAHADANE",
)
VALID_HDF5_COMPRESSION = ("NONE", "LZF", "GZIP")


def _parse_bool(value: str | None, variable_name: str, default: bool) -> bool:
    if value is None or value == "":
        candidate: bool | str = default
    else:
        candidate = value.strip().lower()
    if isinstance(candidate, bool):
        return candidate
    if candidate in {"1", "true", "yes", "on"}:
        return True
    if candidate in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"The '{variable_name}' environment variable must be a boolean value.")


def _parse_int(value: str | None, variable_name: str, default: int) -> int:
    del variable_name
    if value is None or value == "":
        candidate: int | str = default
    else:
        candidate = value
    return int(candidate)


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


def _parse_choice(
    value: str | None,
    variable_name: str,
    *,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    candidate = (value or default).strip().upper()
    if candidate not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"The '{variable_name}' environment variable must be one of: {choices}.")
    return candidate


@dataclass(frozen=True)
class SplitConstraints:
    test_patient_count: int = 20
    validation_patient_count: int = 20


@dataclass(frozen=True)
class ObjectiveConfig:
    optuna_trials: int = 1000
    num_workers: int = max(1, (os.cpu_count() or 1) - 1)
    chunksize: int = 128
    entropy_thumbnail: int = 128


@dataclass(frozen=True)
class CrossfoldConfig:
    normalization_method: str
    source_path: Path
    overwrite_output_dir: bool
    random_state: int
    constraints: SplitConstraints
    objective: ObjectiveConfig
    hdf5_compression: str
    copy_batch_size: int
    calc_checksums: bool
    save_entropy_cache_csv: bool
    log_folder: Path
    log_file_name: str

    @property
    def output_base_dir(self) -> Path:
        return self.source_path.parent / self.normalization_method

    @property
    def output_run_dir(self) -> Path:
        return self.output_base_dir / f"{self.normalization_method}_seed_{self.random_state}"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_crossfold_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> CrossfoldConfig:
    """Load and validate Stage 5 crossfold configuration from `.env`."""

    values = env if env is not None else os.environ
    source_path = _required_path(
        values,
        "CROSSFOLD_SOURCE_HDF5_PATH",
        system_name=system_name,
    )
    if source_path.suffix.lower() not in {".h5", ".sqlite"}:
        raise ValueError(
            "The 'CROSSFOLD_SOURCE_HDF5_PATH' environment variable must point "
            "to a .h5 or .sqlite file."
        )
    if _parse_bool(
        values.get("CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE"),
        "CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE",
        False,
    ):
        raise ValueError(
            "The 'CROSSFOLD_ALLOW_DESTRUCTIVE_MOVE' environment variable is not "
            "supported in HDF5-native Stage 5."
        )
    constraints = SplitConstraints(
        test_patient_count=_parse_int(
            values.get("CROSSFOLD_TEST_PATIENT_COUNT"),
            "CROSSFOLD_TEST_PATIENT_COUNT",
            20,
        ),
        validation_patient_count=_parse_int(
            values.get("CROSSFOLD_VALIDATION_PATIENT_COUNT"),
            "CROSSFOLD_VALIDATION_PATIENT_COUNT",
            20,
        ),
    )
    objective = ObjectiveConfig(
        optuna_trials=max(
            1,
            _parse_int(
                values.get("CROSSFOLD_SPLIT_OPTUNA_TRIALS"),
                "CROSSFOLD_SPLIT_OPTUNA_TRIALS",
                1000,
            ),
        ),
        num_workers=max(
            1,
            _parse_int(
                values.get("CROSSFOLD_ENTROPY_NUM_WORKERS"),
                "CROSSFOLD_ENTROPY_NUM_WORKERS",
                max(1, (os.cpu_count() or 1) - 1),
            ),
        ),
        chunksize=_parse_int(
            values.get("CROSSFOLD_ENTROPY_CHUNKSIZE"),
            "CROSSFOLD_ENTROPY_CHUNKSIZE",
            128,
        ),
        entropy_thumbnail=_parse_int(
            values.get("CROSSFOLD_ENTROPY_THUMBNAIL"),
            "CROSSFOLD_ENTROPY_THUMBNAIL",
            128,
        ),
    )

    return CrossfoldConfig(
        normalization_method=_parse_choice(
            values.get("CROSSFOLD_NORMALIZATION_METHOD"),
            "CROSSFOLD_NORMALIZATION_METHOD",
            default="NOT_NORMALIZED",
            allowed=VALID_NORMALIZATION_METHODS,
        ),
        source_path=source_path,
        overwrite_output_dir=_parse_bool(
            values.get("CROSSFOLD_OVERWRITE_OUTPUT_DIR"),
            "CROSSFOLD_OVERWRITE_OUTPUT_DIR",
            True,
        ),
        random_state=_parse_int(values.get("CROSSFOLD_RANDOM_STATE"), "CROSSFOLD_RANDOM_STATE", 42),
        constraints=constraints,
        objective=objective,
        hdf5_compression=_parse_choice(
            values.get("CROSSFOLD_HDF5_COMPRESSION"),
            "CROSSFOLD_HDF5_COMPRESSION",
            default="NONE",
            allowed=VALID_HDF5_COMPRESSION,
        ),
        copy_batch_size=max(
            1,
            _parse_int(
                values.get("CROSSFOLD_COPY_BATCH_SIZE"),
                "CROSSFOLD_COPY_BATCH_SIZE",
                256,
            ),
        ),
        calc_checksums=_parse_bool(
            values.get("CROSSFOLD_CALC_CHECKSUMS"),
            "CROSSFOLD_CALC_CHECKSUMS",
            False,
        ),
        save_entropy_cache_csv=_parse_bool(
            values.get("CROSSFOLD_SAVE_ENTROPY_CACHE_CSV"),
            "CROSSFOLD_SAVE_ENTROPY_CACHE_CSV",
            True,
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("CROSSFOLD_LOG_FOLDER",),
        ),
        log_file_name=(values.get("CROSSFOLD_LOG_FILE") or "data_preparation.log").strip(),
    )
