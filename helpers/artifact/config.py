from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from helpers.logging_utils import resolve_log_folder
from helpers.runtime_platform import resolve_env_path


@dataclass(frozen=True)
class ArtifactDetectionConfig:
    """Runtime configuration for `1_artifact_detection.py`."""

    images_zip: Path
    geojson_output: Path
    database_folder: Path
    database_path: Path
    temp_root: Path
    log_folder: Path
    log_file_name: str
    device: str
    tissue_detector_model_dir: Path
    tissue_detector_model_name: str
    qc_model_dir: Path
    mpp_model: float
    mpp_model_td: float = 10.0
    model_patch_size: int = 512
    overlay_factor: int = 10
    over_image: float = 0.7
    over_mask: float = 0.3
    back_class: int = 7
    encoder_model: str = "timm-efficientnet-b0"
    encoder_weights: str = "imagenet"
    encoder_model_td: str = "timm-efficientnet-b0"
    encoder_weights_td: str = "imagenet"

    @property
    def log_path(self) -> Path:
        return self.log_folder / self.log_file_name

    @property
    def qc_model_name(self) -> str:
        model_names = {
            1.0: "GrandQC_MPP1.pth",
            1.5: "GrandQC_MPP15.pth",
            2.0: "GrandQC_MPP2.pth",
        }
        try:
            return model_names[self.mpp_model]
        except KeyError as error:
            raise ValueError("ARTIFACT_MPP_MODEL must be one of 1.0, 1.5, or 2.0") from error


def load_artifact_detection_config(
    environment: Mapping[str, str | None],
    *,
    system_name: str | None = None,
) -> ArtifactDetectionConfig:
    """Load Stage 1 artifact detection settings from environment variables."""

    images_zip = _required_path(environment, "ARTIFACT_IMAGES_ZIP", system_name=system_name)
    geojson_output = _path_with_default(
        environment,
        "ARTIFACT_GEOJSON_OUTPUT",
        "./artifacts/geojson",
        system_name=system_name,
    )
    database_folder = _path_with_default(
        environment,
        "ARTIFACT_DATABASE_FOLDER",
        "./databases",
        system_name=system_name,
    )
    temp_root = _path_with_default(
        environment,
        "ARTIFACT_TEMP_FOLDER",
        "./temp/artifact_detection",
        system_name=system_name,
    )
    log_folder = resolve_log_folder(
        environment,
        system_name=system_name,
        fallback_names=("ARTIFACT_LOG_FOLDER",),
    )
    log_file_name = _string_with_default(
        environment,
        "ARTIFACT_LOG_FILE",
        "artifact_detection.log",
    )
    device = _string_with_default(environment, "ARTIFACT_DEVICE", "cuda")
    tissue_detector_model_dir = _path_with_default(
        environment,
        "ARTIFACT_TD_MODEL_DIR",
        "./models/td",
        system_name=system_name,
    )
    tissue_detector_model_name = _string_with_default(
        environment,
        "ARTIFACT_TD_MODEL_NAME",
        "Tissue_Detection_MPP10.pth",
    )
    qc_model_dir = _path_with_default(
        environment,
        "ARTIFACT_QC_MODEL_DIR",
        "./models/qc",
        system_name=system_name,
    )
    mpp_model = _float_with_default(environment, "ARTIFACT_MPP_MODEL", 1.5)
    database_name = _string_with_default(
        environment, "ARTIFACT_DATABASE_NAME", "artifact_detection.db"
    )
    overlay_factor = _int_with_default(environment, "ARTIFACT_OVERLAY_FACTOR", 10)

    return ArtifactDetectionConfig(
        images_zip=images_zip,
        geojson_output=geojson_output,
        database_folder=database_folder,
        database_path=database_folder / database_name,
        temp_root=temp_root,
        log_folder=log_folder,
        log_file_name=log_file_name,
        device=device,
        tissue_detector_model_dir=tissue_detector_model_dir,
        tissue_detector_model_name=tissue_detector_model_name,
        qc_model_dir=qc_model_dir,
        mpp_model=mpp_model,
        overlay_factor=overlay_factor,
    )


def _required_path(
    environment: Mapping[str, str | None],
    name: str,
    *,
    system_name: str | None = None,
) -> Path:
    path = resolve_env_path(
        environment.get(name),
        name,
        system_name=system_name,
        required=True,
    )
    assert path is not None
    return path


def _path_with_default(
    environment: Mapping[str, str | None],
    name: str,
    default: str,
    *,
    system_name: str | None = None,
) -> Path:
    value = environment.get(name) or default
    path = resolve_env_path(value, name, system_name=system_name, required=True)
    assert path is not None
    return path


def _string_with_default(environment: Mapping[str, str | None], name: str, default: str) -> str:
    value = environment.get(name)
    return value if value else default


def _float_with_default(environment: Mapping[str, str | None], name: str, default: float) -> float:
    value = environment.get(name)
    return float(value) if value else default


def _int_with_default(environment: Mapping[str, str | None], name: str, default: int) -> int:
    value = environment.get(name)
    return int(value) if value else default
