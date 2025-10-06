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
import json
import logging
import traceback
import openslide
from data_handlers import BaseHandler

def setup_logging():
    """Configures the logger to write to a file."""
    log_format = '%(asctime)s - %(process)d - %(levelname)s - %(message)s'
    logging.basicConfig(filename='patch_extraction.log', level=logging.INFO, format=log_format, filemode='a')

# --- Constants & Generic Helpers ---
PATCH_AREA = 0
HALF_WINDOW = 0

def check_tissue_percentage_robust(patch_np, required_percentage):
    # This function is generic and correct. Unchanged.
    global PATCH_AREA
    if patch_np is None or patch_np.size == 0:
        return False
    patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(patch_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPse, (7, 7))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    return (np.count_nonzero(tissue_mask) / PATCH_AREA) >= required_percentage

def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords):
    # This function is generic and correct. Unchanged.
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_l0, patch_y_l0 = patch_coords[0] * scale_factor, patch_coords[1] * scale_factor
    win_poly_l0 = Polygon([(patch_x_l0, patch_y_l0), (patch_x_l0 + mask_shape[1]*scale_factor, patch_y_l0), (patch_x_l0 + mask_shape[1]*scale_factor, patch_y_l0 + mask_shape[0]*scale_factor), (patch_x_l0, patch_y_l0 + mask_shape[0]*scale_factor)])
    prep_win = prep(win_poly_l0)
    for poly_l0 in polygons_level0:
        if len(poly_l0) < 3:
            continue
        try:
            anno_poly_l0 = Polygon(poly_l0)
            if not anno_poly_l0.is_valid:
                anno_poly_l0 = anno_poly_l0.buffer(0)
        except Exception:
            continue
        if not prep_win.intersects(anno_poly_l0):
            continue
        try:
            intersection = win_poly_l0.intersection(anno_poly_l0)
        except shapely.errors.TopologicalError:
            continue
        if intersection.is_empty:
            continue
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
    """Generic window processor. Now returns a detailed traceback on failure."""
    # This function's logic is mostly unchanged, but the exception handling is key.
    (path_Image, target_level, window_size, tissue_percentage_req, match_percentage_req, path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder, patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0, artifact_polygons_by_class_level0, artifact_policy, use_artifact_filter) = args
    slide = None
    try:
        slide = openslide.OpenSlide(path_Image)
        x_int, y_int = int(x), int(y)
        patch_coords = (x_int, y_int)
        
        if use_artifact_filter and artifact_polygons_by_class_level0 and artifact_policy:
            scale_factor = slide.level_downsamples[target_level]
            patch_polygon = Polygon([(x, y), (x + window_size, y), (x + window_size, y + window_size), (x, y + window_size)])
            should_drop = False
            for cls, polygons_l0 in artifact_polygons_by_class_level0.items():
                if not polygons_l0:
                    continue
                threshold = artifact_policy['DROP_THRESH'].get(cls)
                if threshold is None:
                    continue
                
                scaled_polys_raw = []
                for p in polygons_l0:
                    if len(p) < 3:
                        continue
                    try:
                        poly = Polygon([(px/scale_factor, py/scale_factor) for px, py in p])
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        scaled_polys_raw.append(poly)
                    except Exception:
                        continue
                
                scaled_polys_flat = [g for geom in scaled_polys_raw for g in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]) if geom.is_valid and geom.geom_type == 'Polygon']
                if not scaled_polys_flat:
                    continue

                prepared_geom = prep(MultiPolygon(scaled_polys_flat))
                if prepared_geom.intersects(patch_polygon):
                    try:
                        intersection = prepared_geom.intersection(patch_polygon)
                        if (intersection.area / PATCH_AREA) > threshold:
                            should_drop = True
                            break
                    except shapely.errors.TopologicalError:
                        continue
            if should_drop:
                slide.close()
                return "SKIPPED_ARTIFACT", None

        patch_pil = slide.read_region(patch_coords, target_level, (window_size, window_size)).convert("RGB")
        patch_np = np.array(patch_pil)

        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return "SKIPPED_TISSUE", None

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
        return "SKIPPED_OVERLAP", None
    except Exception:
        # **MODIFIED**: On ANY exception, capture the full traceback.
        # This is the crucial change for getting detailed error messages.
        if slide:
            slide.close()
        # Return a failure status and the formatted traceback string.
        return "ERROR", traceback.format_exc()

def run_extraction(handler: BaseHandler, path_Image: str, **kwargs):
    slide = None
    try:
        slide_basename = os.path.basename(path_Image)
        logging.info(f"--- Starting processing for slide: {slide_basename} ---")
        slide = openslide.OpenSlide(path_Image)

        annotation_data = handler.load_annotations(slide, **kwargs)
        annotations_cancer_level0 = annotation_data["cancer_polygons"]
        annotations_not_cancer_level0 = annotation_data["not_cancer_polygons"]
        all_polygons_level0 = annotations_cancer_level0 + annotations_not_cancer_level0

        if not all_polygons_level0:
            logging.warning(f"No valid annotations found by handler for slide {slide_basename}")
            slide.close(); slide = None
            return 0, 0

        scale_factor = slide.level_downsamples[kwargs['target_level']]
        target_width, target_height = slide.level_dimensions[kwargs['target_level']]

        artifact_polygons_by_class_level0 = {}
        if kwargs.get('use_artifact_filter') and kwargs.get('path_artifacts_geojson') and kwargs.get('artifact_policy'):
            with open(kwargs['path_artifacts_geojson'], 'r') as f:
                artifact_data = json.load(f)
            for cls in kwargs['artifact_policy']['DROP_THRESH']:
                artifact_polygons_by_class_level0[cls] = []
                for feature in artifact_data.get('features', []):
                    prop_cls = feature.get('properties', {}).get('classification', {}).get('name')
                    if prop_cls == cls:
                        artifact_polygons_by_class_level0[cls].extend(feature['geometry']['coordinates'])
            logging.info(f"Loaded artifact coordinates for {len(artifact_polygons_by_class_level0)} classes.")
            
        scaled_polys_raw = []
        for p in all_polygons_level0:
            try:
                poly = Polygon([(x/scale_factor, y/scale_factor) for x, y in p])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                scaled_polys_raw.append(poly)
            except Exception:
                continue

        scaled_polys_flat = []
        for geom in scaled_polys_raw:
            if geom.geom_type == 'Polygon':
                scaled_polys_flat.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                scaled_polys_flat.extend(list(geom.geoms))
        
        if not scaled_polys_flat:
            logging.warning(f"No valid annotation polygons after scaling for {slide_basename}")
            slide.close(); slide = None
            return 0,0

        combined_annotations = prep(MultiPolygon(scaled_polys_flat))
        x_coords = np.arange(0, target_width - kwargs['window_size'] + 1, kwargs['stride'])
        y_coords = np.arange(0, target_height - kwargs['window_size'] + 1, kwargs['stride'])

        filtered_coords = [(int(x), int(y)) for x in x_coords for y in y_coords if combined_annotations.contains(Point(x + HALF_WINDOW, y + HALF_WINDOW))]
        logging.info(f"Found {len(filtered_coords)} candidate windows.")

        if not filtered_coords:
            slide.close(); return 0, 0

        args_list = [(path_Image, kwargs['target_level'], kwargs['window_size'], kwargs['tissue_percentage_req'], kwargs['match_percentage_req'], kwargs['path_cancer_folder'], kwargs['path_not_cancer_folder'], kwargs['path_cancer_mask_folder'], kwargs['path_not_cancer_mask_folder'], kwargs['patient'], x, y, annotations_cancer_level0, annotations_not_cancer_level0, artifact_polygons_by_class_level0, kwargs.get('artifact_policy'), kwargs.get('use_artifact_filter')) for x, y in filtered_coords]

        logging.info(f"Starting parallel processing with {kwargs['num_workers']} workers...")
        with Pool(processes=kwargs['num_workers']) as pool:
            results = pool.map(process_window, args_list)

        cancer_count = len([r for r, _ in results if r == "SAVED_CANCER"])
        not_cancer_count = len([r for r, _ in results if r == "SAVED_NOT_CANCER"])

        # **MODIFIED**: This is the new, detailed error handling block.
        # It replaces the old generic exception.
        errors = [msg for status, msg in results if status == "ERROR"]
        if errors:
            # Log each traceback received from the workers.
            logging.error(f"Encountered {len(errors)} errors during parallel processing for {slide_basename}.")
            for i, error_traceback in enumerate(errors):
                logging.error(f"--- Worker Error {i+1}/{len(errors)} ---\n{error_traceback}")
            
            # Raise a more informative exception.
            error_summary = f"{len(errors)} worker process(es) failed. See '{logging.getLogger().handlers[0].baseFilename}' for detailed tracebacks."
            raise Exception(error_summary)

        logging.info(f"--- Finished processing slide: {slide_basename} ---")
        return cancer_count, not_cancer_count
    finally:
        if slide:
            slide.close()