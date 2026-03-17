from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from helpers.image_reader_service import (
    SlideProcessingRequest,
    load_slide_runtime_settings,
    run_slide_processing,
)

load_dotenv(override=True)
OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")

try:
    dll_context: AbstractContextManager[None] = nullcontext()
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory) and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        dll_context = cast(AbstractContextManager[None], add_dll_directory(OPENSLIDE_PATH))
    with dll_context:
        import_module("openslide")
except ImportError as error:
    logging.error("Error importing OpenSlide: %s", error)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one-slide processing."""

    parser = argparse.ArgumentParser(description="Generic WSI Patch Extractor.")
    parser.add_argument("--path_Image", type=str, required=True)
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Full path to the annotation file.",
    )
    parser.add_argument("--path_cancer_folder", type=str, required=True)
    parser.add_argument("--path_not_cancer_folder", type=str, required=True)
    parser.add_argument("--path_cancer_mask_folder", type=str, required=True)
    parser.add_argument("--path_not_cancer_mask_folder", type=str, required=True)
    parser.add_argument("--cancer_color", type=str, required=True)
    parser.add_argument("--not_cancer_color", type=str, required=True)
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--path_artifacts_geojson", type=str, required=False, default=None)
    return parser.parse_args()


def main() -> None:
    """Run one slide extraction job and emit the legacy JSON contract."""

    load_dotenv(override=True)
    args = parse_args()
    runtime_settings = load_slide_runtime_settings()

    result = run_slide_processing(
        SlideProcessingRequest(
            image_path=Path(args.path_Image),
            annotation_path=Path(args.annotation_path),
            cancer_folder=Path(args.path_cancer_folder),
            not_cancer_folder=Path(args.path_not_cancer_folder),
            cancer_mask_folder=Path(args.path_cancer_mask_folder),
            not_cancer_mask_folder=Path(args.path_not_cancer_mask_folder),
            cancer_color=args.cancer_color,
            not_cancer_color=args.not_cancer_color,
            patient=args.patient,
            window_size=runtime_settings.window_size,
            stride=runtime_settings.stride,
            match_percentage=runtime_settings.match_percentage,
            tissue_percentage=runtime_settings.tissue_percentage,
            target_level=runtime_settings.target_level,
            num_workers=runtime_settings.num_workers,
            use_advanced_artifact_filtering=runtime_settings.use_advanced_artifact_filtering,
            artifact_policy=runtime_settings.artifact_policy,
            artifacts_geojson_path=Path(args.path_artifacts_geojson)
            if args.path_artifacts_geojson
            else None,
        )
    )
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
