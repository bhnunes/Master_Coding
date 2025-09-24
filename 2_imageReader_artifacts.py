import numpy as np
import cv2
import os
from PIL import Image
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
import shapely.geos
from datetime import datetime
from multiprocessing import Pool
import random
import argparse
from dotenv import load_dotenv
import json
import warnings
import xml.etree.ElementTree as ET
import sys
import yaml # **NEW**: Import YAML library

load_dotenv(override=True)

# --- Configuration from .env ---
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 224))
STRIDE = int(os.getenv('STRIDE', WINDOW_SIZE // 2))
MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE', 0.9))
TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE', 0.9))
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
TARGET_LEVEL = int(os.getenv('TARGET_LEVEL', 0))
NUM_WORKERS = os.cpu_count()

# **NEW**: Artifact Filtering Configuration
USE_ADVANCED_ARTIFACT_FILTERING = os.getenv('USE_ADVANCED_ARTIFACT_FILTERING', 'False').lower() in ('true', '1', 't')
ARTIFACT_POLICY_PATH = os.getenv('ARTIFACT_POLICY_PATH')
ARTIFACT_POLICY = None
if USE_ADVANCED_ARTIFACT_FILTERING:
    try:
        with open(ARTIFACT_POLICY_PATH, 'r') as f:
            ARTIFACT_POLICY = yaml.safe_load(f)
            # Basic validation
            if 'DROP_THRESH' not in ARTIFACT_POLICY:
                raise ValueError("`DROP_THRESH` not found in artifact_policy.yaml")
    except FileNotFoundError:
        print(f"Error: Artifact filtering is ON but policy file not found at {ARTIFACT_POLICY_PATH}", file=sys.stderr)
        USE_ADVANCED_ARTIFACT_FILTERING = False # Disable if file not found
    except Exception as e:
        print(f"Error loading artifact policy YAML: {e}", file=sys.stderr)
        USE_ADVANCED_ARTIFACT_FILTERING = False

# --- OpenSlide Initialization ---
try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except ImportError as e:
    print(f"Error importing OpenSlide: {e}", file=sys.stderr)
    exit(1)
from openslide import OpenSlide

# --- Constants ---
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2


# --- Helper Functions ---
# **NEW**: This function contains the core artifact rejection logic.
def should_drop_patch_due_to_artifacts(patch_polygon, prepared_artifacts_by_class, policy):
    """
    Checks a patch polygon against artifact geometries based on a defined policy.

    Returns:
        bool: True if the patch should be dropped, False otherwise.
    """
    if not policy or not prepared_artifacts_by_class:
        return False

    drop_thresholds = policy.get('DROP_THRESH', {})

    for artifact_class, threshold in drop_thresholds.items():
        if artifact_class in prepared_artifacts_by_class:
            artifact_geom = prepared_artifacts_by_class[artifact_class]
            
            # Calculate intersection area efficiently
            intersection = patch_polygon.intersection(artifact_geom)
            if not intersection.is_empty:
                intersection_area = intersection.area
                coverage_fraction = intersection_area / PATCH_AREA

                if coverage_fraction > threshold:
                    # Optional: Log why a patch was dropped for debugging
                    # print(f"Patch dropped. Reason: {artifact_class} coverage ({coverage_fraction:.2f}) > threshold ({threshold}).")
                    return True # Drop decision
    
    return False # Keep decision


# [Existing helper functions: updateMax, check_tissue_percentage_robust, polygons_to_mask are unchanged]
def updateMax(Xmax, Xmin, Ymax, Ymin, x, y):
    """Updates the bounding box coordinates."""
    Xmax = max(Xmax, x)
    Xmin = min(Xmin, x)
    Ymax = max(Ymax, y)
    Ymin = min(Ymin, y)
    return Xmax, Xmin, Ymax, Ymin

def check_tissue_percentage_robust(patch_np, required_percentage):
    """
    Checks if the tissue content in a patch meets the required percentage
    using a robust HSV + Otsu thresholding method.
    """
    if patch_np is None or patch_np.size == 0:
        return False

    patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation_channel = patch_hsv[:, :, 1]

    # Apply Otsu's thresholding
    _, tissue_mask = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations for refinement
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    tissue_pixels = np.count_nonzero(tissue_mask)
    tissue_ratio = tissue_pixels / PATCH_AREA

    return tissue_ratio >= required_percentage

def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords_target_level):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_level0 = patch_coords_target_level[0] * scale_factor
    patch_y_level0 = patch_coords_target_level[1] * scale_factor
    window_width_level0 = mask_shape[1] * scale_factor
    window_height_level0 = mask_shape[0] * scale_factor
    window_poly_level0 = Polygon([
        (patch_x_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0 + window_height_level0),
        (patch_x_level0, patch_y_level0 + window_height_level0)
    ])
    prepared_window = prep(window_poly_level0)

    for poly_coords_level0 in polygons_level0:
        if len(poly_coords_level0) < 3: continue
        try:
            annotation_polygon_level0 = Polygon(poly_coords_level0)
        except ValueError:
            continue

        if not prepared_window.intersects(annotation_polygon_level0):
            continue
        
        intersection = window_poly_level0.intersection(annotation_polygon_level0)
        if intersection.is_empty: continue
        geoms = intersection.geoms if isinstance(intersection, MultiPolygon) else [intersection]

        for geom in geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty:
                coords = np.array(geom.exterior.coords)
                coords[:, 0] = (coords[:, 0] - patch_x_level0) / scale_factor
                coords[:, 1] = (coords[:, 1] - patch_y_level0) / scale_factor
                coords = np.round(np.clip(coords, 0, mask_shape[1] - 1)).astype(np.int32)
                coords[:, 1] = np.round(np.clip(coords[:, 1], 0, mask_shape[0] - 1)).astype(np.int32)
                if coords.shape[0] >= 3:
                    cv2.fillPoly(mask, [coords], 1)
    return mask


def process_window(args):
    """
    Processes a single window: checks artifacts, tissue, and annotations, then saves.
    """
    (path_Image, target_level, window_size,
     tissue_percentage_req, match_percentage_req,
     path_cancer_folder, path_not_cancer_folder,
     path_cancer_mask_folder, path_not_cancer_mask_folder,
     patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0,
     prepared_artifacts_by_class, artifact_policy) = args # **MODIFIED**: Unpack new args

    slide = None
    try:
        x_int, y_int = int(x), int(y)
        patch_polygon_target_level = Polygon([
            (x_int, y_int), (x_int + window_size, y_int),
            (x_int + window_size, y_int + window_size), (x_int, y_int + window_size)
        ])

        # 1. **MODIFIED**: Artifact Check (fail-fast)
        if USE_ADVANCED_ARTIFACT_FILTERING:
            if should_drop_patch_due_to_artifacts(patch_polygon_target_level, prepared_artifacts_by_class, artifact_policy):
                return True, None # Drop patch silently

        # --- If checks pass, proceed to read image ---
        slide = OpenSlide(path_Image)
        patch_pil = slide.read_region((x_int, y_int), target_level, (window_size, window_size)).convert("RGB")
        patch_np = np.array(patch_pil)

        # 2. Tissue Percentage Check
        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return True, None

        # 3. Annotation Overlap Checks and Saving Logic (Unchanged)
        scale_factor = slide.level_downsamples[target_level]
        patch_coords_target_level = (x_int, y_int)
        mask_shape = (window_size, window_size)
        cancer_mask_patch = polygons_to_mask(mask_shape, annotations_cancer_level0, scale_factor, patch_coords_target_level)
        non_cancer_mask_patch = polygons_to_mask(mask_shape, annotations_not_cancer_level0, scale_factor, patch_coords_target_level)
        cancer_overlap_ratio = np.count_nonzero(cancer_mask_patch) / PATCH_AREA
        non_cancer_overlap_ratio = np.count_nonzero(non_cancer_mask_patch) / PATCH_AREA

        meets_cancer_threshold = cancer_overlap_ratio >= match_percentage_req
        meets_non_cancer_threshold = non_cancer_overlap_ratio >= match_percentage_req
        
        patch_saved = False
        final_mask_to_save = None

        if meets_cancer_threshold and not meets_non_cancer_threshold:
            save_folder_img, save_folder_mask, label = path_cancer_folder, path_cancer_mask_folder, "CANCER"
            final_mask_to_save, patch_saved = cancer_mask_patch, True
        elif meets_non_cancer_threshold and not meets_cancer_threshold:
            save_folder_img, save_folder_mask, label = path_not_cancer_folder, path_not_cancer_mask_folder, "NOT_CANCER"
            final_mask_to_save, patch_saved = np.zeros(mask_shape, dtype=np.uint8), True

        if patch_saved:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            random_number = random.randint(1000, 9999)
            file_basename = f"{label}_PATIENT_{patient}_{x_int}_{y_int}_{random_number}_{timestamp}.png"
            patch_pil.save(os.path.join(save_folder_img, file_basename))
            Image.fromarray(final_mask_to_save * 255).save(os.path.join(save_folder_mask, file_basename))

        slide.close()
        return True, None
    except Exception as e:
        if slide: slide.close()
        return False, f"Error at ({x},{y}): {e}"


def extract_patches_for_slide(path_Image, target_level, window_size, stride,
                             tissue_percentage_req, match_percentage_req,
                             path_cancer_folder, path_not_cancer_folder,
                             path_cancer_mask_folder, path_not_cancer_mask_folder,
                             cancer_color, not_cancer_color, patient,
                             path_artifacts_geojson):
    slide = None
    try:
        slide = OpenSlide(path_Image)
        scale_factor = slide.level_downsamples[target_level]
        target_width, target_height = slide.level_dimensions[target_level]

        # **MODIFIED**: Load artifacts and group them by class
        prepared_artifacts_by_class = {}
        if USE_ADVANCED_ARTIFACT_FILTERING and path_artifacts_geojson and ARTIFACT_POLICY:
            with open(path_artifacts_geojson, 'r') as f:
                artifact_data = json.load(f)
            
            # Group polygons by their classification
            polygons_by_class_level0 = {cls: [] for cls in ARTIFACT_POLICY['DROP_THRESH']}
            for feature in artifact_data.get('features', []):
                props = feature.get('properties', {})
                cls = props.get('classification')
                if cls in polygons_by_class_level0:
                    coords = feature.get('geometry', {}).get('coordinates', [])
                    if coords:
                        polygons_by_class_level0[cls].append(coords[0])

            # Scale and prepare geometries for each class
            for cls, polygons_level0 in polygons_by_class_level0.items():
                if not polygons_level0: continue
                
                shapely_polygons = []
                for poly_coords in polygons_level0:
                    try:
                        scaled_coords = [(x / scale_factor, y / scale_factor) for x, y in poly_coords]
                        if len(scaled_coords) >= 3:
                            shapely_polygons.append(Polygon(scaled_coords))
                    except (ValueError, shapely.geos.errors.TopologicalError):
                        continue # Skip invalid polygons
                
                if shapely_polygons:
                    prepared_artifacts_by_class[cls] = prep(MultiPolygon(shapely_polygons))
            print(f"Loaded and prepared artifact geometries for {len(prepared_artifacts_by_class)} classes.")

        # Annotation loading and window filtering (Unchanged)
        path_Annotation = path_Image.replace('.svs', '.xml', 1)
        if not os.path.exists(path_Annotation): raise FileNotFoundError(f"Annotation file not found: {path_Annotation}")
        annotations_tree = ET.parse(path_Annotation)
        root = annotations_tree.getroot()
        all_polygons_level0 = []
        annotations_cancer_level0 = []
        annotations_not_cancer_level0 = []
        for ann in root.findall('.//Annotation'):
            color = ann.get('LineColor')
            is_cancer, is_non_cancer = (color == str(cancer_color)), (color == str(not_cancer_color))
            if is_cancer or is_non_cancer:
                for reg in ann.findall('.//Region'):
                    verts = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall('.//Vertex')]
                    if len(verts) >= 3:
                        all_polygons_level0.append(verts)
                        if is_cancer: annotations_cancer_level0.append(verts)
                        else: annotations_not_cancer_level0.append(verts)
        if not all_polygons_level0:
            print(f"Warning: No valid annotations for slide {path_Image}", file=sys.stderr)
            slide.close(); return

        scaled_polys = [Polygon([(x/scale_factor, y/scale_factor) for x, y in p]) for p in all_polygons_level0]
        combined_annotations = prep(MultiPolygon(scaled_polys))
        x_coords = np.arange(0, target_width - window_size + 1, stride)
        y_coords = np.arange(0, target_height - window_size + 1, stride)
        filtered_windows_coords = [(x, y) for x in x_coords for y in y_coords if combined_annotations.contains(Point(x + HALF_WINDOW, y + HALF_WINDOW))]

        if not filtered_windows_coords:
            slide.close(); return
        
        # **MODIFIED**: Pass artifact data and policy to workers
        args_list = [(path_Image, target_level, window_size,
                      tissue_percentage_req, match_percentage_req,
                      path_cancer_folder, path_not_cancer_folder,
                      path_cancer_mask_folder, path_not_cancer_mask_folder,
                      patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0,
                      prepared_artifacts_by_class, ARTIFACT_POLICY)
                     for x, y in filtered_windows_coords]

        print(f"Processing {len(filtered_windows_coords)} candidate windows for {path_Image}...", file=sys.stderr)
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process_window, args_list)

        errors = [msg for success, msg in results if not success]
        if errors:
            raise Exception("Errors occurred during processing: " + "; ".join(set(errors)))

    finally:
        if slide: slide.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
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
    args = None
    try:
        args = parser.parse_args()
        os.makedirs(args.path_cancer_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_folder, exist_ok=True)
        os.makedirs(args.path_cancer_mask_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_mask_folder, exist_ok=True)

        extract_patches_for_slide(
            path_Image=args.path_Image,
            target_level=TARGET_LEVEL,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            tissue_percentage_req=TISSUE_PERCENTAGE,
            match_percentage_req=MATCH_PERCENTAGE,
            path_cancer_folder=args.path_cancer_folder,
            path_not_cancer_folder=args.path_not_cancer_folder,
            path_cancer_mask_folder=args.path_cancer_mask_folder,
            path_not_cancer_mask_folder=args.path_not_cancer_mask_folder,
            cancer_color=args.cancer_color,
            not_cancer_color=args.not_cancer_color,
            patient=args.patient,
            path_artifacts_geojson=args.path_artifacts_geojson
        )
        status = 'COMPLETED'
        comments = f'Successfully processed {os.path.basename(args.path_Image)}.'
    except Exception as e:
        status = 'FAILED'
        img_path = os.path.basename(args.path_Image) if args and args.path_Image else "input WSI"
        comments = f"Error processing {img_path}: {e.__class__.__name__}: {e}"
    finally:
        print(json.dumps({"status": status, "comments": comments}))