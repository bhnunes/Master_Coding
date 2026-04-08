from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.lr_finder.search_space import BCEDiceSearchSpace
from helpers.runtime_platform import resolve_env_path
from helpers.training.registry import load_training_model_registry

VALID_EXECUTION_MODES = {"FAST_DEV", "PAPER"}
VALID_AMP_PRECISIONS = {"fp16", "bf16", "fp32", "auto"}

__all__ = ["BCEDiceSearchSpace", "LRFinderConfig", "ModelPlan", "load_lr_finder_config"]


@dataclass(frozen=True)
class ModelPlan:
    architecture: str
    encoder: str


@dataclass(frozen=True)
class LRFinderConfig:
    hdf5_drive_dir: Path
    output_dir: Path
    local_data_dir: Path
    stage_input_locally: bool
    overwrite_output: bool
    smart_sampling: bool
    execution_mode: str
    amp_precision: str
    seed: int
    batch_size: int
    workers: int
    use_subset: bool
    subset_ratio: float
    num_lhs_samples: int
    end_lr: float
    num_iter: int
    num_repeats: int
    optimizer_weight_decay: float
    optimizer_start_lr: float
    pdf_name: str
    hf_token: str | None
    search_space: BCEDiceSearchSpace
    model_plans: list[ModelPlan]
    log_folder: Path = Path("logs")
    log_file_name: str = "lr_finder.log"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def _parse_int(value: str | None, *, default: int) -> int:
    return int(str(default) if value is None or value.strip() == "" else value)


def _parse_float(value: str | None, *, default: float) -> float:
    return float(str(default) if value is None or value.strip() == "" else value)


def _parse_choice(value: str | None, *, default: str, valid: set[str], upper: bool = False) -> str:
    candidate = default if value is None or value.strip() == "" else value.strip()
    normalized = candidate.upper() if upper else candidate
    if normalized not in valid:
        raise ValueError(f"Expected one of {sorted(valid)}, got '{candidate}'.")
    return normalized


def _parse_required_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
    default: str | None = None,
) -> Path:
    resolved = resolve_env_path(
        value if value not in {None, ""} else default,
        variable_name,
        system_name=system_name,
        required=True,
    )
    assert resolved is not None
    return resolved


def _parse_architecture_filter(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def _parse_optional_token(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _build_model_plans(architecture_filter: tuple[str, ...] | None) -> list[ModelPlan]:
    registry = load_training_model_registry()
    available_architectures = set(registry.keys())
    selected_architectures = sorted(available_architectures)
    if architecture_filter is not None:
        missing = sorted(set(architecture_filter) - available_architectures)
        if missing:
            raise ValueError(f"Unknown LR Finder architectures: {', '.join(missing)}")
        selected_architectures = sorted(set(architecture_filter))

    model_plans: list[ModelPlan] = []
    for architecture in selected_architectures:
        for encoder in registry[architecture].encoders:
            model_plans.append(ModelPlan(architecture=architecture, encoder=encoder))
    if not model_plans:
        raise ValueError("LR Finder model plan is empty after registry filtering.")
    return model_plans


def load_lr_finder_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> LRFinderConfig:
    values = env if env is not None else os.environ
    workers_default = os.cpu_count() or 1
    architecture_filter = _parse_architecture_filter(values.get("LR_FINDER_ARCHITECTURES"))
    pdf_name = (values.get("LR_FINDER_PDF_NAME") or "report.pdf").strip()
    if not pdf_name.endswith(".pdf"):
        raise ValueError("LR_FINDER_PDF_NAME must end with '.pdf'.")

    return LRFinderConfig(
        hdf5_drive_dir=_parse_required_path(
            values.get("LR_FINDER_HDF5_DRIVE_DIR"),
            "LR_FINDER_HDF5_DRIVE_DIR",
            system_name=system_name,
        ),
        output_dir=_parse_required_path(
            values.get("LR_FINDER_OUTPUT_DIR"),
            "LR_FINDER_OUTPUT_DIR",
            system_name=system_name,
            default="./reports/lr_finder",
        ),
        local_data_dir=_parse_required_path(
            values.get("LR_FINDER_LOCAL_DATA_DIR"),
            "LR_FINDER_LOCAL_DATA_DIR",
            system_name=system_name,
            default="./temp/lr_finder",
        ),
        stage_input_locally=_parse_bool(values.get("LR_FINDER_STAGE_INPUT_LOCALLY"), default=False),
        overwrite_output=_parse_bool(values.get("LR_FINDER_OVERWRITE_OUTPUT"), default=True),
        smart_sampling=_parse_bool(values.get("LR_FINDER_SMART_SAMPLING"), default=True),
        execution_mode=_parse_choice(
            values.get("LR_FINDER_EXECUTION_MODE"),
            default="PAPER",
            valid=VALID_EXECUTION_MODES,
            upper=True,
        ),
        amp_precision=_parse_choice(
            values.get("LR_FINDER_AMP_PRECISION"),
            default="fp32",
            valid=VALID_AMP_PRECISIONS,
        ),
        seed=_parse_int(values.get("LR_FINDER_SEED"), default=24),
        batch_size=_parse_int(values.get("LR_FINDER_BATCH_SIZE"), default=32),
        workers=max(1, _parse_int(values.get("LR_FINDER_WORKERS"), default=workers_default)),
        use_subset=_parse_bool(values.get("LR_FINDER_USE_SUBSET"), default=False),
        subset_ratio=_parse_float(values.get("LR_FINDER_SUBSET_RATIO"), default=1.0),
        num_lhs_samples=_parse_int(values.get("LR_FINDER_NUM_LHS_SAMPLES"), default=12),
        end_lr=_parse_float(values.get("LR_FINDER_END_LR"), default=1e-1),
        num_iter=_parse_int(values.get("LR_FINDER_NUM_ITER"), default=100),
        num_repeats=_parse_int(values.get("LR_FINDER_NUM_REPEATS"), default=3),
        optimizer_weight_decay=_parse_float(
            values.get("LR_FINDER_OPTIMIZER_WEIGHT_DECAY"), default=1e-4
        ),
        optimizer_start_lr=_parse_float(values.get("LR_FINDER_OPTIMIZER_START_LR"), default=1e-8),
        pdf_name=pdf_name,
        hf_token=_parse_optional_token(
            values.get("HF_TOKEN"),
            values.get("HUGGINGFACE_HUB_TOKEN"),
        ),
        search_space=BCEDiceSearchSpace(
            alpha_min=_parse_float(values.get("LR_FINDER_ALPHA_MIN"), default=0.1),
            alpha_max=_parse_float(values.get("LR_FINDER_ALPHA_MAX"), default=1.0),
            beta_min=_parse_float(values.get("LR_FINDER_BETA_MIN"), default=0.1),
            beta_max=_parse_float(values.get("LR_FINDER_BETA_MAX"), default=0.5),
            gamma_min=_parse_float(values.get("LR_FINDER_GAMMA_MIN"), default=0.1),
            gamma_max=_parse_float(values.get("LR_FINDER_GAMMA_MAX"), default=0.5),
            gamma_log=_parse_bool(values.get("LR_FINDER_GAMMA_LOG"), default=False),
        ),
        model_plans=_build_model_plans(architecture_filter),
        log_folder=resolve_log_folder(
            values,
            system_name=system_name,
            fallback_names=("LR_FINDER_LOG_FOLDER",),
        ),
        log_file_name=(values.get("LR_FINDER_LOG_FILE") or "lr_finder.log").strip(),
    )
