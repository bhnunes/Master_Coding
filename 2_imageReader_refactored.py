import argparse
import os
import json
import logging
from dotenv import load_dotenv
import yaml
import warnings
import xml.etree.ElementTree as ET
import sys

load_dotenv(override=True)
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
# --- OpenSlide Initialization ---
try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except ImportError as e:
    logging.error(f"Error importing OpenSlide: {e}")
    sys.exit(1)

# Import the engine and the specific handlers
import patch_engine
from data_handlers import SVS_XML_Handler, NDPI_NDPA_Handler

def main():
    load_dotenv(override=True)
    
    # Configure logging and warnings
    patch_engine.setup_logging()
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # --- Configuration from .env ---
    WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 224))
    STRIDE = int(os.getenv('STRIDE', WINDOW_SIZE // 2))
    MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE', 0.9))
    TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE', 0.9))
    TARGET_LEVEL = int(os.getenv('TARGET_LEVEL', 0))
    NUM_WORKERS = os.cpu_count()
    
    # **MODIFIED**: Load artifact config here in the controller
    USE_ADVANCED_ARTIFACT_FILTERING = os.getenv('USE_ADVANCED_ARTIFACT_FILTERING', 'False').lower() in ('true', '1', 't')
    ARTIFACT_POLICY_PATH = os.getenv('ARTIFACT_POLICY_PATH')
    ARTIFACT_POLICY = None
    if USE_ADVANCED_ARTIFACT_FILTERING:
        try:
            with open(ARTIFACT_POLICY_PATH, 'r') as f:
                ARTIFACT_POLICY = yaml.safe_load(f)
                if 'DROP_THRESH' not in ARTIFACT_POLICY:
                    raise ValueError("`DROP_THRESH` not found in artifact_policy.yaml")
        except Exception as e:
            logging.error(f"Could not load artifact policy: {e}")
            USE_ADVANCED_ARTIFACT_FILTERING = False

    # Set global constants in the engine (these are safe and don't change per run)
    patch_engine.PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
    patch_engine.HALF_WINDOW = WINDOW_SIZE // 2

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Generic WSI Patch Extractor.')
    parser.add_argument('--handler', type=str, required=True, choices=['SVS', 'NDPI'], help='The handler to use for the slide format (SVS or NDPI).')
    parser.add_argument('--path_Image', type=str, required=True)
    parser.add_argument('--path_cancer_folder', type=str, required=True)
    parser.add_argument('--path_not_cancer_folder', type=str, required=True)
    parser.add_argument('--path_cancer_mask_folder', type=str, required=True)
    parser.add_argument('--path_not_cancer_mask_folder', type=str, required=True)
    parser.add_argument('--cancer_color', type=str, required=True)
    parser.add_argument('--not_cancer_color', type=str, required=True)
    parser.add_argument('--patient', type=str, required=True)
    parser.add_argument('--path_artifacts_geojson', type=str, required=False, default=None)

    status, comments = 'UNKNOWN', ''
    cancer_count, not_cancer_count = 0, 0
    args = None
    try:
        args = parser.parse_args()
        logging.info(f"Script started for image: {os.path.basename(args.path_Image)} with handler: {args.handler}")
        
        if args.handler == 'SVS':
            handler = SVS_XML_Handler()
            annotation_path = args.path_Image.replace('.svs', '.xml')
        elif args.handler == 'NDPI':
            handler = NDPI_NDPA_Handler()
            annotation_path = args.path_Image + '.ndpa'
        else:
            raise ValueError(f"Unknown handler type: {args.handler}")

        # **MODIFIED**: Pack all config into a dictionary to pass to the engine
        kwargs = {
            "handler": handler,
            "path_Image": args.path_Image,
            "annotation_path": annotation_path,
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
            "num_workers": NUM_WORKERS
        }
        cancer_count, not_cancer_count = patch_engine.run_extraction(**kwargs)
        
        status = 'COMPLETED'
        comments = f'Successfully processed {os.path.basename(args.path_Image)}.'
        logging.info(comments)
    except ET.ParseError as e:
        status = 'FAILED'
        img_path = os.path.basename(args.path_Image) if args and args.path_Image else "input WSI"
        logging.error(f"XML ParseError for {img_path}: {e}")
        comments = f"XML ParseError for {img_path}: Annotation file is corrupt."
    except Exception as e:
        status = 'FAILED'
        img_path = os.path.basename(args.path_Image) if args and args.path_Image else "input WSI"
        logging.exception(f"Critical failure while processing {img_path}")
        comments = f"Error processing {img_path}: {e}"
    finally:
        print(json.dumps({
            "status": status,
            "comments": comments,
            "cancer_patches_created": cancer_count,
            "not_cancer_patches_created": not_cancer_count
        }))

if __name__ == '__main__':
    main()