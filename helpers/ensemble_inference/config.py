from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_normalization import (
    RUNTIME_NORMALIZATION_METHOD_ENV_VAR,
    RUNTIME_VAHADANE_BACKEND_ENV_VAR,
    parse_runtime_normalization_method,
    parse_runtime_vahadane_backend,
)
from helpers.runtime_platform import resolve_env_path


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def _parse_int(value: str | None, *, default: int) -> int:
    return int(str(default) if value is None or value.strip() == "" else value)


def _parse_float(value: str | None, *, default: float) -> float:
    return float(str(default) if value is None or value.strip() == "" else value)


def _parse_required_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
) -> Path:
    resolved = resolve_env_path(value, variable_name, system_name=system_name, required=True)
    assert resolved is not None
    return resolved


def _parse_optional_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
    default: str | None = None,
) -> Path | None:
    candidate = value if value not in {None, ""} else default
    return resolve_env_path(candidate, variable_name, system_name=system_name, required=False)


@dataclass(frozen=True)
class EnsembleInferenceConfig:
    recipe_path: Path
    master_manifest_path: Path
    output_dir: Path | None
    local_data_dir: Path
    stage_input_locally: bool
    overwrite_output: bool
    batch_size: int
    workers: int
    seed: int
    visualization_samples: int
    export_csv: bool
    export_latex: bool
    export_visualizations: bool
    patch_positive_area_fraction_threshold: float = 0.0
    runtime_normalization_method: str = "NOT_NORMALIZED"
    runtime_vahadane_backend: str = "fixed_source"
    log_folder: Path = Path("logs")
    log_file_name: str = "ensemble_inference.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def load_ensemble_inference_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> EnsembleInferenceConfig:
    values = env if env is not None else os.environ
    workers_default = os.cpu_count() or 1
    patch_positive_area_fraction_threshold = _parse_float(
        values.get("ENSEMBLE_INFER_PATCH_POSITIVE_AREA_FRACTION_THRESHOLD"),
        default=0.0,
    )
    if not 0.0 <= patch_positive_area_fraction_threshold <= 1.0:
        raise ValueError(
            "ENSEMBLE_INFER_PATCH_POSITIVE_AREA_FRACTION_THRESHOLD must be between 0.0 and 1.0."
        )

    return EnsembleInferenceConfig(
        recipe_path=_parse_required_path(
            values.get("ENSEMBLE_INFER_RECIPE_PATH"),
            "ENSEMBLE_INFER_RECIPE_PATH",
            system_name=system_name,
        ),
        master_manifest_path=_parse_required_path(
            values.get("ENSEMBLE_INFER_MASTER_MANIFEST_PATH"),
            "ENSEMBLE_INFER_MASTER_MANIFEST_PATH",
            system_name=system_name,
        ),
        output_dir=_parse_optional_path(
            values.get("ENSEMBLE_INFER_OUTPUT_DIR"),
            "ENSEMBLE_INFER_OUTPUT_DIR",
            system_name=system_name,
        ),
        local_data_dir=_parse_optional_path(
            values.get("ENSEMBLE_INFER_LOCAL_DATA_DIR"),
            "ENSEMBLE_INFER_LOCAL_DATA_DIR",
            system_name=system_name,
            default="./temp/ensemble_inference",
        )
        or Path("./temp/ensemble_inference"),
        stage_input_locally=_parse_bool(
            values.get("ENSEMBLE_INFER_STAGE_INPUT_LOCALLY"),
            default=True,
        ),
        overwrite_output=_parse_bool(
            values.get("ENSEMBLE_INFER_OVERWRITE_OUTPUT"),
            default=True,
        ),
        batch_size=max(1, _parse_int(values.get("ENSEMBLE_INFER_BATCH_SIZE"), default=32)),
        workers=max(1, _parse_int(values.get("ENSEMBLE_INFER_WORKERS"), default=workers_default)),
        seed=_parse_int(values.get("ENSEMBLE_INFER_SEED"), default=24),
        visualization_samples=max(
            0,
            _parse_int(values.get("ENSEMBLE_INFER_VIS_NUM_SAMPLES"), default=5),
        ),
        export_csv=_parse_bool(values.get("ENSEMBLE_INFER_EXPORT_CSV"), default=True),
        export_latex=_parse_bool(values.get("ENSEMBLE_INFER_EXPORT_LATEX"), default=True),
        export_visualizations=_parse_bool(
            values.get("ENSEMBLE_INFER_EXPORT_VISUALIZATIONS"),
            default=True,
        ),
        patch_positive_area_fraction_threshold=patch_positive_area_fraction_threshold,
        runtime_normalization_method=parse_runtime_normalization_method(
            values.get(RUNTIME_NORMALIZATION_METHOD_ENV_VAR)
        ),
        runtime_vahadane_backend=parse_runtime_vahadane_backend(
            values.get(RUNTIME_VAHADANE_BACKEND_ENV_VAR)
        ),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("ENSEMBLE_INFER_LOG_FOLDER",),
        ),
        log_file_name=(values.get("ENSEMBLE_INFER_LOG_FILE") or "ensemble_inference.log").strip(),
    )
