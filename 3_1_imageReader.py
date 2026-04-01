from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from helpers.extraction.image_reader_service import (
    SlideProcessingRequest,
    load_slide_runtime_settings,
    run_slide_processing,
)
from helpers.logging_utils import configure_root_logger, resolve_log_folder
from helpers.runtime_platform import load_openslide_module

load_dotenv(override=True)


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
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--path_artifacts_geojson", type=str, required=False, default=None)
    return parser.parse_args()


def main() -> None:
    """Run one slide extraction job and emit the CLI JSON result contract."""

    load_dotenv(override=True)
    log_folder = resolve_log_folder(os.environ, fallback_names=("EXTRACTION_LOG_FOLDER",))
    log_file_name = (os.environ.get("IMAGE_READER_LOG_FILE") or "image_reader.log").strip()
    configure_root_logger(
        log_folder / log_file_name,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.ERROR,
        file_mode="a",
        file_pattern="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        console_pattern="%(message)s",
        console_stream=sys.stderr,
    )
    load_openslide_module()
    args = parse_args()
    runtime_settings = load_slide_runtime_settings()

    result = run_slide_processing(
        SlideProcessingRequest(
            image_path=Path(args.path_Image),
            annotation_path=Path(args.annotation_path),
            dataset_tag=(os.environ.get("TAG") or "").strip(),
            patient=args.patient,
            window_size=runtime_settings.window_size,
            stride=runtime_settings.stride,
            match_percentage=runtime_settings.match_percentage,
            tissue_percentage=runtime_settings.tissue_percentage,
            target_level=runtime_settings.target_level,
            num_workers=runtime_settings.num_workers,
            use_advanced_artifact_filtering=runtime_settings.use_advanced_artifact_filtering,
            hiseg_xml_coord_level=int(os.environ.get("HISEG_XML_COORD_LEVEL") or 6),
            openslide_cache_bytes=runtime_settings.openslide_cache_bytes,
            hdf5_compression=runtime_settings.hdf5_compression,
            preload_scan_area_max_bytes=runtime_settings.preload_scan_area_max_bytes,
            artifacts_geojson_path=Path(args.path_artifacts_geojson)
            if args.path_artifacts_geojson
            else None,
        )
    )
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
