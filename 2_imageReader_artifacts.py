import numpy as np
import cv2
import os
from PIL import Image
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
import shapely
from datetime import datetime
from multiprocessing import Pool
import random
import argparse
from dotenv import load_dotenv
import json
import warnings
import xml.etree.ElementTree as ET
import sys
import yaml
import logging
import traceback

# --- Configuration & Logging Setup ---
load_dotenv(override=True)

def setup_logging():
    log_format = '%(asctime)s - %(process)d - %(levelname)s - %(message)s'
    logging.basicConfig(filename='patch_extraction.log', level=logging.INFO,
                        format=log_format, filemode='a')

# --- Configuration from .env ---
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 224))
STRIDE = int(os.getenv('STRIDE', WINDOW_SIZE // 2))
MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE', 0.9))
TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE', 0.9))
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
TARGET_LEVEL = int(os.getenv('TARGET_LEVEL', 0))
NUM_WORKERS = os.cpu_count()

# --- Artifact Filtering Configuration ---
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

# --- Constants ---
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2

# --- Helper Functions ---
def check_tissue_percentage_robust(patch_np, required_percentage):
    if patch_np is None or patch_np.size == 0: return False
    patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(patch_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    return (np.count_nonzero(tissue_mask) / PATCH_AREA) >= required_percentage

def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_l0, patch_y_l0 = patch_coords[0] * scale_factor, patch_coords[1] * scale_factor
    win_poly_l0 = Polygon([(patch_x_l0, patch_y_l0), (patch_x_l0 + mask_shape[1]*scale_factor, patch_y_l0), (patch_x_l0 + mask_shape[1]*scale_factor, patch_y_l0 + mask_shape[0]*scale_factor), (patch_x_l0, patch_y_l0 + mask_shape[0]*scale_factor)])
    prep_win = prep(win_poly_l0)
    for poly_l0 in polygons_level0:
        if len(poly_l0) < 3: continue
        try:
            anno_poly_l0 = Polygon(poly_l0)
            if not anno_poly_l0.is_valid:
                anno_poly_l0 = anno_poly_l0.buffer(0)
        except Exception:
            continue
        if not prep_win.intersects(anno_poly_l0): continue
        try:
            intersection = win_poly_l0.intersection(anno_poly_l0)
        except shapely.errors.TopologicalError:
            continue
        if intersection.is_empty: continue
        geoms = intersection.geoms if isinstance(intersection, MultiPolygon) else [intersection]
        for geom in geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty:
                coords = np.array(geom.exterior.coords)
                coords[:, 0] = (coords[:, 0] - patch_x_l0) / scale_factor
                coords[:, 1] = (coords[:, 1] - patch_y_l0) / scale_factor
                coords = np.round(np.clip(coords, 0, mask_shape[1] - 1)).astype(np.int32)
                coords[:, 1] = np.round(np.clip(coords[:, 1], 0, mask_shape[0] - 1)).astype(np.int32)
                if len(coords) >= 3:
                    cv2.fillPoly(mask, [coords], 1)
    return mask

def process_window(args):
    (path_Image, target_level, window_size,
     tissue_percentage_req, match_percentage_req,
     path_cancer_folder, path_not_cancer_folder,
     path_cancer_mask_folder, path_not_cancer_mask_folder,
     patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0,
     artifact_polygons_by_class_level0, artifact_policy) = args

    slide = None
    try:
        slide = openslide.OpenSlide(path_Image)
        x_int, y_int = int(x), int(y)
        patch_coords = (x_int, y_int)
        
        if USE_ADVANCED_ARTIFACT_FILTERING and artifact_polygons_by_class_level0:
            scale_factor = slide.level_downsamples[target_level]
            patch_polygon = Polygon([(x, y), (x + window_size, y), (x + window_size, y + window_size), (x, y + window_size)])
            
            should_drop = False
            for cls, polygons_l0 in artifact_polygons_by_class_level0.items():
                if not polygons_l0: continue
                threshold = artifact_policy['DROP_THRESH'].get(cls)
                if threshold is None: continue
                
                scaled_polys = []
                for p in polygons_l0:
                    if len(p) < 3: continue
                    try:
                        poly = Polygon([(px/scale_factor, py/scale_factor) for px, py in p])
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        scaled_polys.append(poly)
                    except Exception:
                        continue
                
                if not scaled_polys: continue

                unprepared_geom = MultiPolygon(scaled_polys)
                prepared_geom = prep(unprepared_geom)
                
                if prepared_geom.intersects(patch_polygon):
                    try:
                        intersection = unprepared_geom.intersection(patch_polygon)
                        coverage = intersection.area / PATCH_AREA
                        if coverage > threshold:
                            logging.info(f"Patch at {patch_coords} DROPPED. Reason: {cls} coverage ({coverage:.2f}) > threshold ({threshold}).")
                            should_drop = True
                            break
                    except shapely.errors.TopologicalError:
                         logging.warning(f"Skipping intersection check for a problematic artifact geometry at {patch_coords}.")
                         continue

            if should_drop:
                slide.close()
                return "SKIPPED", None

        patch_pil = slide.read_region(patch_coords, target_level, (window_size, window_size)).convert("RGB")
        patch_np = np.array(patch_pil)

        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return "SKIPPED", None

        scale_factor = slide.level_downsamples[target_level]
        cancer_mask = polygons_to_mask((window_size, window_size), annotations_cancer_level0, scale_factor, patch_coords)
        non_cancer_mask = polygons_to_mask((window_size, window_size), annotations_not_cancer_level0, scale_factor, patch_coords)
        cancer_overlap = np.count_nonzero(cancer_mask) / PATCH_AREA
        non_cancer_overlap = np.count_nonzero(non_cancer_mask) / PATCH_AREA
        
        patch_saved = False
        if (cancer_overlap >= match_percentage_req) and (non_cancer_overlap < match_percentage_req):
            save_folder_img, save_folder_mask, label = path_cancer_folder, path_cancer_mask_folder, "CANCER"
            final_mask, patch_saved = cancer_mask, True
        elif (non_cancer_overlap >= match_percentage_req) and (cancer_overlap < match_percentage_req):
            save_folder_img, save_folder_mask, label = path_not_cancer_folder, path_not_cancer_mask_folder, "NOT_CANCER"
            final_mask, patch_saved = np.zeros((window_size, window_size), dtype=np.uint8), True

        if patch_saved:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            file_basename = f"{label}_PATIENT_{patient}_{x_int}_{y_int}_{random.randint(1000,9999)}_{timestamp}.png"
            patch_pil.save(os.path.join(save_folder_img, file_basename))
            Image.fromarray(final_mask * 255).save(os.path.join(save_folder_mask, file_basename))
            slide.close()
            return f"SAVED_{label}", None
        
        slide.close()
        return "SKIPPED", None
    except Exception:
        if slide: slide.close()
        return False, traceback.format_exc()

def extract_patches_for_slide(path_Image, **kwargs):
    slide = None
    try:
        slide_basename = os.path.basename(path_Image)
        logging.info(f"--- Starting processing for slide: {slide_basename} ---")
        slide = openslide.OpenSlide(path_Image)
        scale_factor = slide.level_downsamples[kwargs['target_level']]
        target_width, target_height = slide.level_dimensions[kwargs['target_level']]

        artifact_polygons_by_class_level0 = {}
        if USE_ADVANCED_ARTIFACT_FILTERING and kwargs.get('path_artifacts_geojson') and ARTIFACT_POLICY:
            with open(kwargs['path_artifacts_geojson'], 'r') as f:
                artifact_data = json.load(f)
            for cls in ARTIFACT_POLICY['DROP_THRESH']:
                artifact_polygons_by_class_level0[cls] = []
            for feature in artifact_data.get('features', []):
                cls = feature.get('properties', {}).get('classification')
                if cls in artifact_polygons_by_class_level0:
                    artifact_polygons_by_class_level0[cls].append(feature['geometry']['coordinates'][0])
            logging.info(f"Loaded artifact coordinates for {len(artifact_polygons_by_class_level0)} classes.")
        
        path_Annotation = path_Image.replace('.svs', '.xml', 1)
        if not os.path.exists(path_Annotation): raise FileNotFoundError(f"Annotation not found: {path_Annotation}")
        root = ET.parse(path_Annotation).getroot()
        all_polygons_level0, annotations_cancer_level0, annotations_not_cancer_level0 = [], [], []
        for ann in root.findall('.//Annotation'):
            is_cancer = str(ann.get('LineColor')) == str(kwargs['cancer_color'])
            is_non_cancer = str(ann.get('LineColor')) == str(kwargs['not_cancer_color'])
            if is_cancer or is_non_cancer:
                for reg in ann.findall('.//Region'):
                    verts = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall('.//Vertex')]
                    if len(verts) >= 3:
                        all_polygons_level0.append(verts)
                        if is_cancer: annotations_cancer_level0.append(verts)
                        else: annotations_not_cancer_level0.append(verts)
        if not all_polygons_level0:
            logging.warning(f"No valid annotations for slide {slide_basename}")
            slide.close(); slide = None # **THE FIX for ctypes.ArgumentError**
            return 0, 0

        # **THE FIX for ValueError**: Flatten the list of geometries after healing them.
        scaled_polys_raw = []
        for p in all_polygons_level0:
            try:
                poly = Polygon([(x/scale_factor, y/scale_factor) for x, y in p])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                scaled_polys_raw.append(poly)
            except Exception:
                continue
        
        # Flatten the list: handle cases where buffer(0) creates a MultiPolygon
        scaled_polys_flat = []
        for geom in scaled_polys_raw:
            if geom.geom_type == 'Polygon':
                scaled_polys_flat.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                scaled_polys_flat.extend(list(geom.geoms))

        if not scaled_polys_flat:
            logging.warning(f"No valid annotation polygons after scaling for slide {slide_basename}")
            slide.close(); slide = None # **THE FIX for ctypes.ArgumentError**
            return 0,0

        combined_annotations = prep(MultiPolygon(scaled_polys_flat))
        x_coords = np.arange(0, target_width - WINDOW_SIZE + 1, STRIDE)
        y_coords = np.arange(0, target_height - WINDOW_SIZE + 1, STRIDE)
        filtered_coords = [(int(x), int(y)) for x in x_coords for y in y_coords if combined_annotations.contains(Point(x + HALF_WINDOW, y + HALF_WINDOW))]
        
        logging.info(f"Found {len(filtered_coords)} candidate windows.")
        if not filtered_coords:
            slide.close(); return 0, 0

        args_list = [(path_Image, kwargs['target_level'], kwargs['window_size'], kwargs['tissue_percentage_req'], kwargs['match_percentage_req'],
                      kwargs['path_cancer_folder'], kwargs['path_not_cancer_folder'], kwargs['path_cancer_mask_folder'], kwargs['path_not_cancer_mask_folder'],
                      kwargs['patient'], x, y, annotations_cancer_level0, annotations_not_cancer_level0,
                      artifact_polygons_by_class_level0, ARTIFACT_POLICY)
                     for x, y in filtered_coords]

        logging.info(f"Starting parallel processing...")
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process_window, args_list)

        cancer_patches_created = 0
        not_cancer_patches_created = 0
        errors = []
        for status, msg in results:
            if status == "SAVED_CANCER":
                cancer_patches_created += 1
            elif status == "SAVED_NOT_CANCER":
                not_cancer_patches_created += 1
            elif status is False:
                errors.append(msg)
        
        if errors:
            logging.error(f"Encountered {len(errors)} errors during worker processing.")
            logging.error("--- BEGIN FIRST WORKER TRACEBACK ---")
            logging.error(errors[0])
            logging.error("--- END FIRST WORKER TRACEBACK ---")
            raise Exception("Errors occurred in workers. See log for full traceback.")
        
        logging.info(f"--- Finished processing slide: {slide_basename} ---")
        return cancer_patches_created, not_cancer_patches_created

    finally:
        if slide: slide.close()

if __name__ == '__main__':
    setup_logging()
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    # [Argument parsing is unchanged]
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
        logging.info(f"Script started for image: {os.path.basename(args.path_Image)}")
        
        cancer_count, not_cancer_count = extract_patches_for_slide(
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