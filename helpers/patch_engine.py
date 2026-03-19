# mypy: ignore-errors

import json
import logging
import os
import random
import traceback
from datetime import datetime
from multiprocessing import Pool
from typing import cast

import cv2
import numpy as np
import shapely
from dotenv import load_dotenv
from PIL import Image
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.prepared import prep

from helpers.data_handlers import BaseHandler
from helpers.runtime_platform import load_openslide_module

load_dotenv(override=True)
# --- Constants ---
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 224))  # Default 224
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # For noise removal
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # For hole filling
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2


def setup_logging():
    """Configures the logger to write to a file."""
    log_format = "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
    logging.basicConfig(
        filename="patch_extraction.log",
        level=logging.INFO,
        format=log_format,
        filemode="a",
    )


def check_tissue_percentage_robust(patch_np, required_percentage):
    # This function is generic and correct. Unchanged.
    if patch_np is None or patch_np.size == 0:
        return False
    patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(patch_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations for refinement
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    return (np.count_nonzero(tissue_mask) / PATCH_AREA) >= required_percentage


def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords):
    # This function is generic and correct. Unchanged.
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_l0, patch_y_l0 = patch_coords[0] * scale_factor, patch_coords[1] * scale_factor
    win_poly_l0 = Polygon(
        [
            (patch_x_l0, patch_y_l0),
            (patch_x_l0 + mask_shape[1] * scale_factor, patch_y_l0),
            (
                patch_x_l0 + mask_shape[1] * scale_factor,
                patch_y_l0 + mask_shape[0] * scale_factor,
            ),
            (patch_x_l0, patch_y_l0 + mask_shape[0] * scale_factor),
        ]
    )
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
            if geom.geom_type == "Polygon" and not geom.is_empty:
                polygon = cast(Polygon, geom)
                coords = np.array(polygon.exterior.coords)
                coords[:, 0] = (coords[:, 0] - patch_x_l0) / scale_factor
                coords[:, 1] = (coords[:, 1] - patch_y_l0) / scale_factor
                coords = np.round(np.clip(coords, 0, mask_shape[1] - 1)).astype(np.int32)
                coords[:, 1] = np.round(np.clip(coords[:, 1], 0, mask_shape[0] - 1)).astype(
                    np.int32
                )
                if len(coords) >= 3:
                    cv2.fillPoly(mask, [coords], (1,))
    return mask


def process_window(args):
    """Generic window processor. Returns a detailed traceback on failure."""
    (
        path_Image,
        target_level,
        window_size,
        tissue_percentage_req,
        match_percentage_req,
        path_cancer_folder,
        path_not_cancer_folder,
        path_cancer_mask_folder,
        path_not_cancer_mask_folder,
        patient,
        x,
        y,
        annotations_cancer_level0,
        annotations_not_cancer_level0,
        artifact_polygons_by_class_level0,
        artifact_policy,
        use_artifact_filter,
    ) = args
    slide = None
    openslide_module = load_openslide_module()
    try:
        slide = openslide_module.OpenSlide(path_Image)
        x_int, y_int = int(x), int(y)
        patch_coords = (x_int, y_int)

        if use_artifact_filter and artifact_polygons_by_class_level0 and artifact_policy:
            scale_factor = slide.level_downsamples[target_level]
            patch_polygon = Polygon(
                [
                    (x, y),
                    (x + window_size, y),
                    (x + window_size, y + window_size),
                    (x, y + window_size),
                ]
            )
            should_drop = False
            for cls, polygons_l0 in artifact_polygons_by_class_level0.items():
                if not polygons_l0:
                    continue
                threshold = artifact_policy["DROP_THRESH"].get(cls)
                if threshold is None:
                    continue

                scaled_polys_raw = []
                for p in polygons_l0:
                    if len(p) < 3:
                        continue
                    try:
                        poly = Polygon([(px / scale_factor, py / scale_factor) for px, py in p])
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        scaled_polys_raw.append(poly)
                    except Exception:
                        continue

                scaled_polys_flat = [
                    geom_part
                    for geom in scaled_polys_raw
                    for geom_part in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom])
                    if geom.is_valid and geom.geom_type == "Polygon"
                ]
                if not scaled_polys_flat:
                    continue

                unprepared_geom = MultiPolygon(scaled_polys_flat)
                prepared_geom = prep(unprepared_geom)

                if prepared_geom.intersects(patch_polygon):
                    try:
                        intersection = unprepared_geom.intersection(patch_polygon)
                        coverage = intersection.area / PATCH_AREA
                        if coverage > threshold:
                            logging.info(
                                "Patch at %s DROPPED. Reason: %s coverage (%.2f) > threshold (%s).",
                                patch_coords,
                                cls,
                                coverage,
                                threshold,
                            )
                            should_drop = True
                            break
                    except shapely.errors.TopologicalError:
                        logging.warning(
                            "Skipping intersection check for a problematic artifact "
                            "geometry at %s.",
                            patch_coords,
                        )
                        continue
            if should_drop:
                patch_pil = slide.read_region(
                    patch_coords, target_level, (window_size, window_size)
                ).convert("RGB")
                save_folder_img_artifact = str(path_not_cancer_folder).replace(
                    "NOT_CANCER", "ARTIFACTS"
                )
                os.makedirs(save_folder_img_artifact, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                file_basename = (
                    f"ARTIFACT_PATIENT_{patient}_{x_int}_{y_int}_"
                    f"{random.randint(1000, 9999)}_{timestamp}.png"
                )
                patch_pil.save(os.path.join(save_folder_img_artifact, file_basename))
                slide.close()
                return "SKIPPED_ARTIFACT", None

        patch_pil = slide.read_region(
            patch_coords, target_level, (window_size, window_size)
        ).convert("RGB")
        patch_np = np.array(patch_pil)

        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return "SKIPPED_TISSUE", None

        scale_factor = slide.level_downsamples[target_level]
        cancer_mask = polygons_to_mask(
            (window_size, window_size), annotations_cancer_level0, scale_factor, patch_coords
        )
        non_cancer_mask = polygons_to_mask(
            (window_size, window_size), annotations_not_cancer_level0, scale_factor, patch_coords
        )

        cancer_overlap = np.count_nonzero(cancer_mask) / PATCH_AREA
        non_cancer_overlap = np.count_nonzero(non_cancer_mask) / PATCH_AREA

        patch_saved = False
        label = ""
        save_folder_img = ""
        save_folder_mask = ""
        final_mask = np.zeros((window_size, window_size), dtype=np.uint8)
        if (cancer_overlap >= match_percentage_req) and (non_cancer_overlap < match_percentage_req):
            save_folder_img, save_folder_mask, label = (
                path_cancer_folder,
                path_cancer_mask_folder,
                "CANCER",
            )
            final_mask, patch_saved = cancer_mask, True
        elif (non_cancer_overlap >= match_percentage_req) and (
            cancer_overlap < match_percentage_req
        ):
            save_folder_img, save_folder_mask, label = (
                path_not_cancer_folder,
                path_not_cancer_mask_folder,
                "NOT_CANCER",
            )
            final_mask, patch_saved = np.zeros((window_size, window_size), dtype=np.uint8), True

        if patch_saved:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_basename = (
                f"{label}_PATIENT_{patient}_{x_int}_{y_int}_"
                f"{random.randint(1000, 9999)}_{timestamp}.png"
            )
            patch_pil.save(os.path.join(save_folder_img, file_basename))
            Image.fromarray((final_mask * 255).astype(np.uint8)).save(
                os.path.join(save_folder_mask, file_basename)
            )
            slide.close()
            return f"SAVED_{label}", None

        slide.close()
        return "SKIPPED_OVERLAP", None
    except Exception:
        if slide:
            slide.close()
        return "ERROR", traceback.format_exc()


def run_extraction(handler: BaseHandler, path_Image: str, **kwargs):
    slide = None
    openslide_module = load_openslide_module()
    try:
        slide_basename = os.path.basename(path_Image)
        logging.info(f"--- Starting processing for slide: {slide_basename} ---")
        slide = openslide_module.OpenSlide(path_Image)

        annotation_data = handler.load_annotations(slide, **kwargs)
        annotations_cancer_level0 = annotation_data["cancer_polygons"]
        annotations_not_cancer_level0 = annotation_data["not_cancer_polygons"]
        all_polygons_level0 = annotations_cancer_level0 + annotations_not_cancer_level0

        if not all_polygons_level0:
            logging.warning("No valid annotations found by handler for slide %s", slide_basename)
            slide.close()
            slide = None
            return 0, 0

        # --- NEW: ROBUST ARTIFACT PARSING BLOCK ---
        artifact_polygons_by_class_level0 = {}
        if (
            kwargs.get("use_artifact_filter")
            and kwargs.get("path_artifacts_geojson")
            and kwargs.get("artifact_policy")
        ):
            logging.info(f"Advanced artifact filtering is ACTIVE for {slide_basename}.")
            try:
                with open(kwargs["path_artifacts_geojson"]) as f:
                    artifact_data = json.load(f)

                # Initialize dictionaries for all classes we care about from the policy
                for cls in kwargs["artifact_policy"]["DROP_THRESH"]:
                    artifact_polygons_by_class_level0[cls] = []

                # Safely parse the GeoJSON features
                for feature in artifact_data.get("features", []):
                    properties = feature.get("properties", {})
                    if not properties:
                        continue

                    classification_obj = properties.get("classification")
                    if not classification_obj:
                        continue

                    # --- FIX IS HERE ---
                    # Robustly get the class name whether it's a dict or a string
                    prop_cls = None
                    if isinstance(classification_obj, dict):
                        prop_cls = classification_obj.get("name")
                    elif isinstance(classification_obj, str):
                        prop_cls = classification_obj
                    # --- END FIX ---

                    if prop_cls and prop_cls in artifact_polygons_by_class_level0:
                        geometry = feature.get("geometry", {})
                        geom_type = geometry.get("type")
                        coordinates = geometry.get("coordinates")
                        if not geom_type or not coordinates:
                            continue

                        # Handle both Polygon and MultiPolygon types
                        if geom_type == "Polygon":
                            if coordinates:
                                artifact_polygons_by_class_level0[prop_cls].append(coordinates[0])
                        elif geom_type == "MultiPolygon":
                            for poly_coords in coordinates:
                                if poly_coords:
                                    artifact_polygons_by_class_level0[prop_cls].append(
                                        poly_coords[0]
                                    )

                # Log a summary of what was found
                for cls, polys in artifact_polygons_by_class_level0.items():
                    logging.info("  - Loaded %s artifact polygons for class '%s'.", len(polys), cls)

            except FileNotFoundError:
                logging.warning(
                    "Artifact GeoJSON file not found: %s. "
                    "Filtering will be skipped for this slide.",
                    kwargs["path_artifacts_geojson"],
                )
            except Exception as e:
                logging.error(
                    "Failed to parse artifact GeoJSON %s: %s. Filtering skipped.",
                    kwargs["path_artifacts_geojson"],
                    e,
                )
                artifact_polygons_by_class_level0 = {}

        scale_factor = slide.level_downsamples[kwargs["target_level"]]
        target_width, target_height = slide.level_dimensions[kwargs["target_level"]]

        scaled_polys_raw = []
        for p in all_polygons_level0:
            try:
                poly = Polygon([(x / scale_factor, y / scale_factor) for x, y in p])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                scaled_polys_raw.append(poly)
            except Exception:
                continue

        scaled_polys_flat = []
        for geom in scaled_polys_raw:
            if geom.geom_type == "Polygon":
                scaled_polys_flat.append(geom)
            elif geom.geom_type == "MultiPolygon":
                scaled_polys_flat.extend(list(geom.geoms))

        if not scaled_polys_flat:
            logging.warning("No valid annotation polygons after scaling for %s", slide_basename)
            slide.close()
            slide = None
            return 0, 0

        combined_annotations = MultiPolygon(scaled_polys_flat)
        # Use prep for optimized geometric checks
        prepared_annotations = prep(combined_annotations)

        # --- OPTIMIZATION: Bounding Box Pre-filtering ---
        # Get the bounding box of all annotations.
        # The .bounds property returns (minx, miny, maxx, maxy).
        min_x, min_y, max_x, max_y = combined_annotations.bounds

        logging.info(
            "Annotations bounding box (L%s): [(%s, %s), (%s, %s)]",
            kwargs["target_level"],
            int(min_x),
            int(min_y),
            int(max_x),
            int(max_y),
        )

        # Start the grid at the beginning of the bounding box.
        # Use max(0, ...) to ensure we don't start with negative coordinates if bounds are weird.
        x_start = max(0, int(min_x))
        y_start = max(0, int(min_y))

        # End the grid at the end of the bounding box, but ensure it does not exceed
        # the slide's actual dimensions.
        # This is where target_width and target_height are now critically important.
        x_end = min(int(max_x) + kwargs["window_size"], target_width)
        y_end = min(int(max_y) + kwargs["window_size"], target_height)

        x_coords = np.arange(x_start, x_end - kwargs["window_size"] + 1, kwargs["stride"])
        y_coords = np.arange(y_start, y_end - kwargs["window_size"] + 1, kwargs["stride"])

        logging.info(f"Optimized scan area: [({x_start}, {y_start}), ({x_end}, {y_end})]")
        # --- END OF OPTIMIZATION ---

        # The rest of the code now operates on a much smaller grid of candidate coordinates.
        # Note the use of the 'prepared_annotations' object for the faster 'contains' check.
        filtered_coords = [
            (int(x), int(y))
            for x in x_coords
            for y in y_coords
            if prepared_annotations.contains(Point(x + HALF_WINDOW, y + HALF_WINDOW))
        ]

        # The number of candidate windows will now be much more reasonable.
        logging.info(f"Found {len(filtered_coords)} candidate windows after optimization.")

        if not filtered_coords:
            slide.close()
            return 0, 0

        args_list = [
            (
                path_Image,
                kwargs["target_level"],
                kwargs["window_size"],
                kwargs["tissue_percentage_req"],
                kwargs["match_percentage_req"],
                kwargs["path_cancer_folder"],
                kwargs["path_not_cancer_folder"],
                kwargs["path_cancer_mask_folder"],
                kwargs["path_not_cancer_mask_folder"],
                kwargs["patient"],
                x,
                y,
                annotations_cancer_level0,
                annotations_not_cancer_level0,
                artifact_polygons_by_class_level0,
                kwargs.get("artifact_policy"),
                kwargs.get("use_artifact_filter"),
            )
            for x, y in filtered_coords
        ]

        logging.info(f"Starting parallel processing with {kwargs['num_workers']} workers...")
        with Pool(processes=kwargs["num_workers"]) as pool:
            results = pool.map(process_window, args_list)

        cancer_count = len([r for r, _ in results if r == "SAVED_CANCER"])
        not_cancer_count = len([r for r, _ in results if r == "SAVED_NOT_CANCER"])

        errors = [msg for status, msg in results if status == "ERROR"]
        if errors:
            logging.error(
                "Encountered %s errors during parallel processing for %s.",
                len(errors),
                slide_basename,
            )
            for i, error_traceback in enumerate(errors):
                logging.error("--- Worker Error %s/%s ---\n%s", i + 1, len(errors), error_traceback)

            first_handler = logging.getLogger().handlers[0]
            log_filename = getattr(first_handler, "baseFilename", "patch_extraction.log")
            error_summary = (
                f"{len(errors)} worker process(es) failed. See '{log_filename}' "
                "for detailed tracebacks."
            )
            raise Exception(error_summary)

        logging.info(f"--- Finished processing slide: {slide_basename} ---")
        return cancer_count, not_cancer_count
    finally:
        if slide:
            slide.close()
