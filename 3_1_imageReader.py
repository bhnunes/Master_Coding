import argparse
import json
import logging
import os
import sys
import warnings
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from typing import cast

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

from helpers import patch_engine
from helpers.data_handlers import BaseHandler, JSON_Handler, NDPI_NDPA_Handler, SVS_XML_Handler

# --- All imports and OpenSlide initialization remain the same ---
load_dotenv(override=True)
OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")

try:
    dll_context: AbstractContextManager[None] = nullcontext()
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory) and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        dll_context = cast(AbstractContextManager[None], add_dll_directory(OPENSLIDE_PATH))
    with dll_context:
        import_module("openslide")
except ImportError as e:
    logging.error(f"Error importing OpenSlide: {e}")
    sys.exit(1)

# --- NEW: Handler dispatch mapping ---
HANDLER_MAPPING = {
    (".svs", ".xml"): SVS_XML_Handler,
    (".ndpi", ".ndpa"): NDPI_NDPA_Handler,
    (".tif", ".json"): JSON_Handler,
    (".svs", ".json"): JSON_Handler,
}


def get_handler_for_files(image_path: str, annotation_path: str) -> BaseHandler:
    """Selects the correct data handler based on file extensions."""
    img_ext = os.path.splitext(image_path)[1].lower()
    ann_ext = os.path.splitext(annotation_path)[1].lower()

    handler_class = HANDLER_MAPPING.get((img_ext, ann_ext))
    if handler_class is None:
        raise ValueError(
            "No handler found for file combination: "
            f"Image ('{img_ext}') and Annotation ('{ann_ext}')"
        )

    logging.info(
        f"Dispatching handler: {handler_class.__name__} for {os.path.basename(image_path)}"
    )
    return handler_class()  # type: ignore[abstract]


def main() -> None:
    load_dotenv(override=True)

    patch_engine.setup_logging()  # type: ignore[no-untyped-call]
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # --- Configuration from .env remains the same ---
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 224))
    STRIDE = int(os.getenv("STRIDE", WINDOW_SIZE // 2))
    MATCH_PERCENTAGE = float(os.getenv("MATCH_PERCENTAGE", 1.0))
    TISSUE_PERCENTAGE = float(os.getenv("TISSUE_PERCENTAGE", 0.3))
    TARGET_LEVEL = int(os.getenv("TARGET_LEVEL", 0))
    NUM_WORKERS = max(1, os.cpu_count() or 1)

    USE_ADVANCED_ARTIFACT_FILTERING = os.getenv(
        "USE_ADVANCED_ARTIFACT_FILTERING", "False"
    ).lower() in ("true", "1", "t")
    ARTIFACT_POLICY_PATH = os.getenv("ARTIFACT_POLICY_PATH")
    ARTIFACT_POLICY = None
    if USE_ADVANCED_ARTIFACT_FILTERING:
        try:
            if not ARTIFACT_POLICY_PATH:
                raise ValueError(
                    "ARTIFACT_POLICY_PATH is required when artifact filtering is enabled"
                )
            with open(ARTIFACT_POLICY_PATH) as f:
                ARTIFACT_POLICY = yaml.safe_load(f)
            if "DROP_THRESH" not in ARTIFACT_POLICY:
                raise ValueError("DROP_THRESH not found in artifact_policy.yaml")
        except Exception as e:
            logging.error(f"Could not load artifact policy: {e}")
            USE_ADVANCED_ARTIFACT_FILTERING = False

    patch_engine.PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
    patch_engine.HALF_WINDOW = WINDOW_SIZE // 2

    # --- Argument Parsing (MODIFIED) ---
    parser = argparse.ArgumentParser(description="Generic WSI Patch Extractor.")
    parser.add_argument("--path_Image", type=str, required=True)
    parser.add_argument(
        "--annotation_path", type=str, required=True, help="Full path to the annotation file."
    )
    parser.add_argument("--path_cancer_folder", type=str, required=True)
    parser.add_argument("--path_not_cancer_folder", type=str, required=True)
    parser.add_argument("--path_cancer_mask_folder", type=str, required=True)
    parser.add_argument("--path_not_cancer_mask_folder", type=str, required=True)
    parser.add_argument("--cancer_color", type=str, required=True)
    parser.add_argument("--not_cancer_color", type=str, required=True)
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--path_artifacts_geojson", type=str, required=False, default=None)

    status, comments = "UNKNOWN", ""
    cancer_count, not_cancer_count = 0, 0
    args = None
    try:
        args = parser.parse_args()
        # --- NEW: Automatic handler dispatch ---
        handler = get_handler_for_files(args.path_Image, args.annotation_path)
        logging.info("Script started for image: %s", os.path.basename(args.path_Image))

        kwargs = {
            "handler": handler,
            "path_Image": args.path_Image,
            "annotation_path": args.annotation_path,
            "target_level": TARGET_LEVEL,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "tissue_percentage_req": TISSUE_PERCENTAGE,
            "match_percentage_req": MATCH_PERCENTAGE,
            "path_cancer_folder": args.path_cancer_folder,
            "path_not_cancer_folder": args.path_not_cancer_folder,
            "path_cancer_mask_folder": args.path_cancer_mask_folder,
            "path_not_cancer_mask_folder": args.path_not_cancer_mask_folder,
            "cancer_color": args.cancer_color,
            "not_cancer_color": args.not_cancer_color,
            "patient": args.patient,
            "path_artifacts_geojson": args.path_artifacts_geojson,
            "use_artifact_filter": USE_ADVANCED_ARTIFACT_FILTERING,
            "artifact_policy": ARTIFACT_POLICY,
            "num_workers": NUM_WORKERS,
        }

        cancer_count, not_cancer_count = patch_engine.run_extraction(**kwargs)
        status = "COMPLETED"
        comments = f"Successfully processed {os.path.basename(args.path_Image)}."
        logging.info(comments)

    except Exception as e:
        status = "FAILED"
        img_path = os.path.basename(args.path_Image) if args and args.path_Image else "input WSI"
        logging.exception(f"Critical failure while processing {img_path}")
        comments = f"Error processing {img_path}: {e}"
    finally:
        # The final JSON output remains the same, ensuring compatibility
        print(
            json.dumps(
                {
                    "status": status,
                    "comments": comments,
                    "cancer_patches_created": cancer_count,
                    "not_cancer_patches_created": not_cancer_count,
                }
            )
        )


if __name__ == "__main__":
    main()
