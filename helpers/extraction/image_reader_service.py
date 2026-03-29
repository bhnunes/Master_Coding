from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from helpers.extraction import patch_engine
from helpers.extraction.data_handlers import (
    BaseHandler,
    JSON_Handler,
    NDPI_NDPA_Handler,
    SVS_XML_Handler,
)
from helpers.extraction.hdf5_manifest import update_stage2_shard_manifest
from helpers.extraction.hdf5_storage import write_slide_patch_dataset_hdf5
from helpers.runtime_platform import load_openslide_module

HANDLER_MAPPING: dict[tuple[str, str], type[BaseHandler]] = {
    (".svs", ".xml"): SVS_XML_Handler,
    (".ndpi", ".ndpa"): NDPI_NDPA_Handler,
    (".tif", ".json"): JSON_Handler,
    (".svs", ".json"): JSON_Handler,
}


@dataclass(frozen=True)
class SlideRuntimeSettings:
    """Runtime settings shared by CLI and database-driven slide extraction."""

    window_size: int
    stride: int
    match_percentage: float
    tissue_percentage: float
    target_level: int
    num_workers: int
    use_advanced_artifact_filtering: bool


@dataclass(frozen=True)
class SlideProcessingRequest:
    """All inputs required to process one slide."""

    image_path: Path
    annotation_path: Path
    cancer_color: str
    not_cancer_color: str
    patient: str
    window_size: int
    stride: int
    match_percentage: float
    tissue_percentage: float
    target_level: int
    num_workers: int
    use_advanced_artifact_filtering: bool
    artifacts_geojson_path: Path | None = None
    profile_output_path: Path | None = None
    hdf5_output_path: Path | None = None


@dataclass(frozen=True)
class SlideProcessingResult:
    """Outcome contract shared by CLI and database manager."""

    status: str
    comments: str
    cancer_patches_created: int
    not_cancer_patches_created: int
    artifact_patch_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "comments": self.comments,
            "cancer_patches_created": self.cancer_patches_created,
            "not_cancer_patches_created": self.not_cancer_patches_created,
            "artifact_patch_records": self.artifact_patch_records,
        }


def load_slide_runtime_settings(
    env: Mapping[str, str | None] | None = None,
    *,
    _system_name: str | None = None,
) -> SlideRuntimeSettings:
    """Load patch extraction settings from environment values."""

    values = env if env is not None else os.environ
    window_size = int(values.get("WINDOW_SIZE") or 224)
    stride = int(values.get("STRIDE") or (window_size // 2))
    use_advanced_artifact_filtering = (
        values.get("USE_ADVANCED_ARTIFACT_FILTERING") or "False"
    ).lower() in {
        "true",
        "1",
        "t",
    }

    return SlideRuntimeSettings(
        window_size=window_size,
        stride=stride,
        match_percentage=float(values.get("MATCH_PERCENTAGE") or 1.0),
        tissue_percentage=float(values.get("TISSUE_PERCENTAGE") or 0.3),
        target_level=int(values.get("TARGET_LEVEL") or 0),
        num_workers=max(1, int(values.get("NUM_WORKERS") or (os.cpu_count() or 1))),
        use_advanced_artifact_filtering=use_advanced_artifact_filtering,
    )


def get_handler_for_files(image_path: str, annotation_path: str) -> BaseHandler:
    """Select the correct annotation handler based on file extensions."""

    img_ext = Path(image_path).suffix.lower()
    ann_ext = Path(annotation_path).suffix.lower()
    handler_class = HANDLER_MAPPING.get((img_ext, ann_ext))
    if handler_class is None:
        raise ValueError(
            "No handler found for file combination: "
            f"Image ('{img_ext}') and Annotation ('{ann_ext}')"
        )
    logging.info(
        "Dispatching handler: %s for %s",
        handler_class.__name__,
        Path(image_path).name,
    )
    return handler_class()


def run_slide_processing(request: SlideProcessingRequest) -> SlideProcessingResult:
    """Run patch extraction for one slide through the shared Python API."""

    load_openslide_module()
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    patch_engine.PATCH_AREA = request.window_size * request.window_size
    patch_engine.HALF_WINDOW = request.window_size // 2

    handler = get_handler_for_files(str(request.image_path), str(request.annotation_path))
    logging.info("Script started for image: %s", request.image_path.name)

    try:
        cancer_count, not_cancer_count, artifact_patch_records = patch_engine.run_extraction(
            handler=handler,
            path_Image=str(request.image_path),
            annotation_path=str(request.annotation_path),
            target_level=request.target_level,
            window_size=request.window_size,
            stride=request.stride,
            tissue_percentage_req=request.tissue_percentage,
            match_percentage_req=request.match_percentage,
            cancer_color=request.cancer_color,
            not_cancer_color=request.not_cancer_color,
            patient=request.patient,
            path_artifacts_geojson=str(request.artifacts_geojson_path)
            if request.artifacts_geojson_path is not None
            else None,
            profile_output_path=str(request.profile_output_path)
            if request.profile_output_path is not None
            else None,
            use_artifact_filter=request.use_advanced_artifact_filtering,
            num_workers=request.num_workers,
        )
        artifact_patch_records = cast(list[dict[str, Any]], artifact_patch_records)
        if request.hdf5_output_path is not None:
            shard_output = write_slide_patch_dataset_hdf5(
                output_path=request.hdf5_output_path,
                records=artifact_patch_records,
            )
            update_stage2_shard_manifest(
                request.hdf5_output_path.parent / "manifest.json",
                request.hdf5_output_path if shard_output is not None else request.hdf5_output_path,
            )
    except Exception as error:
        logging.exception("Critical failure while processing %s", request.image_path.name)
        return SlideProcessingResult(
            status="FAILED",
            comments=f"Error processing {request.image_path.name}: {error}",
            cancer_patches_created=0,
            not_cancer_patches_created=0,
            artifact_patch_records=[],
        )

    comments = f"Successfully processed {request.image_path.name}."
    logging.info(comments)
    return SlideProcessingResult(
        status="COMPLETED",
        comments=comments,
        cancer_patches_created=cancer_count,
        not_cancer_patches_created=not_cancer_count,
        artifact_patch_records=artifact_patch_records,
    )
