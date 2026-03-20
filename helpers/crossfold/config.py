from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.runtime_platform import resolve_env_path

VALID_NORMALIZATION_METHODS = (
    "NOT_NORMALIZED",
    "REINHARD",
    "RUIFROK",
    "MACENKO",
    "VAHADANE",
)
VALID_SCORE_SPLITS = ("TRAIN", "VALIDATION", "TEST")


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


def _parse_float(value: str | None, variable_name: str, default: float) -> float:
    del variable_name
    if value is None or value == "":
        candidate: float | str = default
    else:
        candidate = value
    return float(candidate)


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
    min_test_patients: int = 20
    min_val_patients: int = 5
    min_train_patients: int = 5
    test_ratio: float = 0.10
    val_ratio: float = 0.10
    require_train_image_dominance: bool = True
    require_both_classes_if_possible: bool = True
    max_tries: int = 1000
    adaptive: bool = True


@dataclass(frozen=True)
class ObjectiveConfig:
    enable_objective: bool = True
    score_split: str = "TRAIN"
    maximize: bool = True
    metric: str = "entropy_median_per_patient_median"
    num_workers: int = max(1, (os.cpu_count() or 1) - 1)
    chunksize: int = 128
    entropy_thumbnail: int = 128


@dataclass(frozen=True)
class CrossfoldConfig:
    normalization_method: str
    data_directory: Path
    overwrite_output_dir: bool
    random_state: int
    constraints: SplitConstraints
    objective: ObjectiveConfig
    calc_checksums: bool
    save_entropy_cache_csv: bool

    @property
    def output_base_dir(self) -> Path:
        return self.data_directory / self.normalization_method

    @property
    def output_run_dir(self) -> Path:
        return self.output_base_dir / f"{self.normalization_method}_seed_{self.random_state}"


def load_crossfold_config(
    env: Mapping[str, str | None] | None = None,
    *,
    system_name: str | None = None,
) -> CrossfoldConfig:
    """Load and validate Stage 5 crossfold configuration from `.env`."""

    values = env if env is not None else os.environ
    constraints = SplitConstraints(
        min_test_patients=_parse_int(
            values.get("CROSSFOLD_MIN_TEST_PATIENTS"),
            "CROSSFOLD_MIN_TEST_PATIENTS",
            20,
        ),
        min_val_patients=_parse_int(
            values.get("CROSSFOLD_MIN_VAL_PATIENTS"),
            "CROSSFOLD_MIN_VAL_PATIENTS",
            5,
        ),
        min_train_patients=_parse_int(
            values.get("CROSSFOLD_MIN_TRAIN_PATIENTS"),
            "CROSSFOLD_MIN_TRAIN_PATIENTS",
            5,
        ),
        test_ratio=_parse_float(values.get("CROSSFOLD_TEST_RATIO"), "CROSSFOLD_TEST_RATIO", 0.10),
        val_ratio=_parse_float(values.get("CROSSFOLD_VAL_RATIO"), "CROSSFOLD_VAL_RATIO", 0.10),
        require_train_image_dominance=_parse_bool(
            values.get("CROSSFOLD_REQUIRE_TRAIN_IMAGE_DOMINANCE"),
            "CROSSFOLD_REQUIRE_TRAIN_IMAGE_DOMINANCE",
            True,
        ),
        require_both_classes_if_possible=_parse_bool(
            values.get("CROSSFOLD_REQUIRE_BOTH_CLASSES_IF_POSSIBLE"),
            "CROSSFOLD_REQUIRE_BOTH_CLASSES_IF_POSSIBLE",
            True,
        ),
        max_tries=_parse_int(values.get("CROSSFOLD_MAX_TRIES"), "CROSSFOLD_MAX_TRIES", 1000),
        adaptive=_parse_bool(values.get("CROSSFOLD_ADAPTIVE"), "CROSSFOLD_ADAPTIVE", True),
    )
    objective = ObjectiveConfig(
        enable_objective=_parse_bool(
            values.get("CROSSFOLD_ENABLE_OBJECTIVE"),
            "CROSSFOLD_ENABLE_OBJECTIVE",
            True,
        ),
        score_split=_parse_choice(
            values.get("CROSSFOLD_OBJECTIVE_SCORE_SPLIT"),
            "CROSSFOLD_OBJECTIVE_SCORE_SPLIT",
            default="TRAIN",
            allowed=VALID_SCORE_SPLITS,
        ),
        maximize=_parse_bool(
            values.get("CROSSFOLD_OBJECTIVE_MAXIMIZE"),
            "CROSSFOLD_OBJECTIVE_MAXIMIZE",
            True,
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
        data_directory=_required_path(values, "CROSSFOLD_DATA_DIRECTORY", system_name=system_name),
        overwrite_output_dir=_parse_bool(
            values.get("CROSSFOLD_OVERWRITE_OUTPUT_DIR"),
            "CROSSFOLD_OVERWRITE_OUTPUT_DIR",
            True,
        ),
        random_state=_parse_int(values.get("CROSSFOLD_RANDOM_STATE"), "CROSSFOLD_RANDOM_STATE", 42),
        constraints=constraints,
        objective=objective,
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
    )
