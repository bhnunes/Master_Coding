# mypy: ignore-errors

import json
import logging
import os
import re
import time
import traceback
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import shapely
from dotenv import load_dotenv
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import clip_by_rect
from shapely.prepared import prep
from shapely.strtree import STRtree

from helpers.extraction.data_handlers import BaseHandler
from helpers.extraction.profiling import (
    PhaseStats,
    build_profile_summary,
    create_phase_stats,
    merge_phase_stats,
    record_phase,
    write_profile_summary,
)
from helpers.runtime_platform import load_openslide_module

load_dotenv(override=True)
# --- Constants ---
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 224))  # Default 224
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # For noise removal
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # For hole filling
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2
ARTIFACT_CLASS_TO_COLUMN = {
    "Fold": "cov_fold",
    "PenMarking": "cov_penmarking",
    "OOF": "cov_oof",
    "Darkspot & Foreign Object": "cov_darkspot_foreign",
    "Edge & Air Bubble": "cov_edge_airbubble",
}

_WORKER_CONTEXT = {}
WINDOW_PROFILE_PHASES = (
    "artifact_coverage",
    "read_region",
    "tissue_check",
    "cancer_mask",
    "not_cancer_mask",
)


def build_patch_record(
    *,
    filename: str,
    label: str,
    patient_id: object,
    slide_id: object,
    artifact_coverages: dict[str, float],
    patch_np: np.ndarray,
    final_mask: np.ndarray,
) -> dict[str, object]:
    record: dict[str, object] = {
        "filename": filename,
        "label": 1 if label == "CANCER" else 0,
        "patient_id": str(patient_id),
        "slide_id": str(slide_id),
        "_image_array": patch_np.astype(np.uint8, copy=False),
        "_mask_array": final_mask.astype(np.uint8, copy=False),
        **artifact_coverages,
    }
    return record


def sanitize_patch_filename_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    sanitized = text.strip("-._")
    return sanitized or "unknown"


def build_patch_filename(
    *,
    label: str,
    patient_id: object,
    slide_id: object,
    x_coord: int,
    y_coord: int,
) -> str:
    safe_label = sanitize_patch_filename_component(label)
    safe_patient_id = sanitize_patch_filename_component(patient_id)
    safe_slide_id = sanitize_patch_filename_component(slide_id)
    return (
        f"{safe_label}_PATIENT_{safe_patient_id}_SLIDE_{safe_slide_id}_"
        f"X_{int(x_coord)}_Y_{int(y_coord)}.png"
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
            coords_list = clip_geometry_to_patch_coords(
                anno_poly_l0,
                patch_x=patch_x_l0,
                patch_y=patch_y_l0,
                mask_width=mask_shape[1] * scale_factor,
                mask_height=mask_shape[0] * scale_factor,
            )
        except shapely.errors.TopologicalError:
            continue
        if coords_list:
            scaled_coords_list = []
            for coords in coords_list:
                scaled_coords = coords.astype(np.float64)
                scaled_coords[:, 0] = np.round(scaled_coords[:, 0] / scale_factor)
                scaled_coords[:, 1] = np.round(scaled_coords[:, 1] / scale_factor)
                scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, mask_shape[1] - 1)
                scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, mask_shape[0] - 1)
                scaled_coords_list.append(scaled_coords.astype(np.int32))
            cv2.fillPoly(mask, scaled_coords_list, (1,))
    return mask


def clip_geometry_to_patch_coords(geometry, *, patch_x, patch_y, mask_width, mask_height):
    clipped = clip_by_rect(geometry, patch_x, patch_y, patch_x + mask_width, patch_y + mask_height)
    if clipped.is_empty:
        return []
    geoms = clipped.geoms if isinstance(clipped, MultiPolygon) else [clipped]
    coords_list = []
    for geom in geoms:
        if geom.geom_type != "Polygon" or geom.is_empty:
            continue
        polygon = cast(Polygon, geom)
        coords_raw = np.asarray(polygon.exterior.coords, dtype=np.float64)
        coords = np.column_stack(
            (
                np.round(np.clip(coords_raw[:, 0] - patch_x, 0, mask_width - 1)),
                np.round(np.clip(coords_raw[:, 1] - patch_y, 0, mask_height - 1)),
            )
        ).astype(np.int32)
        if len(coords) >= 3:
            coords_list.append(coords)
    return coords_list


def get_zero_artifact_coverages():
    return {column_name: 0.0 for column_name in ARTIFACT_CLASS_TO_COLUMN.values()}


def compute_artifact_coverages_for_patch(
    artifact_polygons_by_class_level0,
    patch_polygon,
    scale_factor,
    patch_area,
):
    coverages = get_zero_artifact_coverages()
    for artifact_class, column_name in ARTIFACT_CLASS_TO_COLUMN.items():
        polygons_l0 = artifact_polygons_by_class_level0.get(artifact_class, [])
        if not polygons_l0:
            continue

        scaled_polys_raw = []
        for polygon_points in polygons_l0:
            if len(polygon_points) < 3:
                continue
            try:
                poly = Polygon(
                    [(px / scale_factor, py / scale_factor) for px, py in polygon_points]
                )
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

        try:
            artifact_geometry = MultiPolygon(scaled_polys_flat)
            if not prep(artifact_geometry).intersects(patch_polygon):
                continue
            intersection = artifact_geometry.intersection(patch_polygon)
            coverages[column_name] = float(intersection.area / patch_area)
        except shapely.errors.TopologicalError:
            logging.warning(
                "Skipping artifact coverage for a problematic geometry at patch polygon %s.",
                patch_polygon.bounds,
            )
            continue
    return coverages


def build_scaled_polygon_index(polygons_level0, scale_factor):
    scaled_polygons = []
    for polygon_points in polygons_level0:
        if len(polygon_points) < 3:
            continue
        try:
            poly = Polygon([(px / scale_factor, py / scale_factor) for px, py in polygon_points])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                geoms = list(cast(MultiPolygon, poly).geoms)
            else:
                geoms = [poly]
            scaled_polygons.extend([geom for geom in geoms if geom.geom_type == "Polygon"])
        except Exception:
            continue
    return scaled_polygons, STRtree(scaled_polygons) if scaled_polygons else None


def build_artifact_geometry_index(artifact_polygons_by_class_level0, scale_factor):
    artifact_geometries = {}
    for artifact_class, column_name in ARTIFACT_CLASS_TO_COLUMN.items():
        polygons_l0 = artifact_polygons_by_class_level0.get(artifact_class, [])
        if not polygons_l0:
            continue

        scaled_polys_raw = []
        for polygon_points in polygons_l0:
            if len(polygon_points) < 3:
                continue
            try:
                poly = Polygon(
                    [(px / scale_factor, py / scale_factor) for px, py in polygon_points]
                )
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                scaled_polys_raw.append(poly)
            except Exception:
                continue

        scaled_polys_flat = [
            geom_part
            for geom in scaled_polys_raw
            for geom_part in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom])
            if geom_part.is_valid and geom_part.geom_type == "Polygon" and not geom_part.is_empty
        ]
        if not scaled_polys_flat:
            continue

        try:
            artifact_geometry = MultiPolygon(scaled_polys_flat)
            artifact_geometries[column_name] = (artifact_geometry, prep(artifact_geometry))
        except shapely.errors.TopologicalError:
            logging.warning(
                "Skipping artifact geometry index for class '%s' due to invalid geometry.",
                artifact_class,
            )
    return artifact_geometries


def compute_artifact_coverages_from_index(artifact_geometry_index, patch_polygon, patch_area):
    coverages = get_zero_artifact_coverages()
    for column_name, (artifact_geometry, prepared_geometry) in artifact_geometry_index.items():
        if not prepared_geometry.intersects(patch_polygon):
            continue
        try:
            intersection = artifact_geometry.intersection(patch_polygon)
            coverages[column_name] = float(intersection.area / patch_area)
        except shapely.errors.TopologicalError:
            logging.warning(
                "Skipping artifact coverage for a problematic geometry at patch polygon %s.",
                patch_polygon.bounds,
            )
    return coverages


def polygons_to_mask_with_index(mask_shape, polygon_index, patch_coords):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    polygons_level, tree = polygon_index
    if not polygons_level or tree is None:
        return mask

    patch_x, patch_y = patch_coords
    win_poly = Polygon(
        [
            (patch_x, patch_y),
            (patch_x + mask_shape[1], patch_y),
            (patch_x + mask_shape[1], patch_y + mask_shape[0]),
            (patch_x, patch_y + mask_shape[0]),
        ]
    )
    prep_win = prep(win_poly)
    for polygon_idx in tree.query(win_poly):
        polygon = polygons_level[int(polygon_idx)]
        if not prep_win.intersects(polygon):
            continue
        try:
            coords_list = clip_geometry_to_patch_coords(
                polygon,
                patch_x=patch_x,
                patch_y=patch_y,
                mask_width=mask_shape[1],
                mask_height=mask_shape[0],
            )
        except shapely.errors.TopologicalError:
            continue
        if coords_list:
            cv2.fillPoly(mask, coords_list, (1,))
    return mask


def chunk_coordinates(filtered_coords, batch_size):
    iterator = iter(filtered_coords)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def _initialize_worker(worker_context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = worker_context


def _process_window_with_slide(slide, x, y):
    context = _WORKER_CONTEXT
    profile_enabled = bool(context.get("profile_output_path"))
    window_phase_stats = create_phase_stats(WINDOW_PROFILE_PHASES) if profile_enabled else None
    x_int, y_int = int(x), int(y)
    patch_coords = (x_int, y_int)
    window_size = context["window_size"]

    cancer_mask_started_at = time.perf_counter()
    cancer_mask = polygons_to_mask_with_index(
        (window_size, window_size), context["cancer_polygon_index"], patch_coords
    )
    if window_phase_stats is not None:
        record_phase(
            window_phase_stats, "cancer_mask", time.perf_counter() - cancer_mask_started_at
        )

    non_cancer_mask_started_at = time.perf_counter()
    non_cancer_mask = polygons_to_mask_with_index(
        (window_size, window_size), context["not_cancer_polygon_index"], patch_coords
    )
    if window_phase_stats is not None:
        record_phase(
            window_phase_stats,
            "not_cancer_mask",
            time.perf_counter() - non_cancer_mask_started_at,
        )

    cancer_overlap = np.count_nonzero(cancer_mask) / PATCH_AREA
    non_cancer_overlap = np.count_nonzero(non_cancer_mask) / PATCH_AREA

    patch_saved = False
    label = ""
    final_mask = np.zeros((window_size, window_size), dtype=np.uint8)
    if (cancer_overlap >= context["match_percentage_req"]) and (
        non_cancer_overlap < context["match_percentage_req"]
    ):
        label = "CANCER"
        final_mask, patch_saved = cancer_mask, True
    elif (non_cancer_overlap >= context["match_percentage_req"]) and (
        cancer_overlap < context["match_percentage_req"]
    ):
        label = "NOT_CANCER"
        final_mask, patch_saved = np.zeros((window_size, window_size), dtype=np.uint8), True

    if not patch_saved:
        return (
            ("SKIPPED_OVERLAP", None, window_phase_stats)
            if window_phase_stats is not None
            else ("SKIPPED_OVERLAP", None)
        )

    read_started_at = time.perf_counter()
    patch_pil = slide.read_region(
        patch_coords, context["target_level"], (window_size, window_size)
    ).convert("RGB")
    patch_np = np.array(patch_pil)
    if window_phase_stats is not None:
        record_phase(window_phase_stats, "read_region", time.perf_counter() - read_started_at)

    tissue_started_at = time.perf_counter()
    tissue_ok = check_tissue_percentage_robust(patch_np, context["tissue_percentage_req"])
    if window_phase_stats is not None:
        record_phase(window_phase_stats, "tissue_check", time.perf_counter() - tissue_started_at)
    if not tissue_ok:
        return (
            ("SKIPPED_TISSUE", None, window_phase_stats)
            if window_phase_stats is not None
            else ("SKIPPED_TISSUE", None)
        )

    artifact_coverages = get_zero_artifact_coverages()
    artifact_geometry_index = context.get("artifact_geometry_index")
    if context.get("use_artifact_filter") and artifact_geometry_index:
        artifact_started_at = time.perf_counter()
        artifact_coverages = compute_artifact_coverages_from_index(
            artifact_geometry_index=artifact_geometry_index,
            patch_polygon=shapely.box(x_int, y_int, x_int + window_size, y_int + window_size),
            patch_area=PATCH_AREA,
        )
        if window_phase_stats is not None:
            record_phase(
                window_phase_stats,
                "artifact_coverage",
                time.perf_counter() - artifact_started_at,
            )

    file_basename = build_patch_filename(
        label=label,
        patient_id=context["patient"],
        slide_id=context["slide_id"],
        x_coord=x_int,
        y_coord=y_int,
    )
    patch_record = build_patch_record(
        filename=file_basename,
        label=label,
        patient_id=context["patient"],
        slide_id=context["slide_id"],
        artifact_coverages=artifact_coverages,
        patch_np=patch_np,
        final_mask=final_mask,
    )
    return (
        (f"SAVED_{label}", patch_record, window_phase_stats)
        if window_phase_stats is not None
        else (f"SAVED_{label}", patch_record)
    )


def process_window_batch(coord_batch):
    slide = None
    openslide_module = load_openslide_module()
    profile_enabled = bool(_WORKER_CONTEXT.get("profile_output_path"))
    try:
        slide = openslide_module.OpenSlide(_WORKER_CONTEXT["path_Image"])
        return [_process_window_with_slide(slide, x, y) for x, y in coord_batch]
    except Exception:
        if profile_enabled:
            return [("ERROR", traceback.format_exc(), create_phase_stats(WINDOW_PROFILE_PHASES))]
        return [("ERROR", traceback.format_exc())]
    finally:
        if slide:
            slide.close()


def iter_window_results(filtered_coords, num_workers, worker_state, batch_size):
    with Pool(
        processes=num_workers,
        initializer=_initialize_worker,
        initargs=(worker_state,),
    ) as pool:
        for batch_results in pool.imap_unordered(
            process_window_batch,
            chunk_coordinates(filtered_coords, batch_size),
            chunksize=1,
        ):
            yield from batch_results


def run_extraction(handler: BaseHandler, path_Image: str, **kwargs):
    slide = None
    openslide_module = load_openslide_module()
    slide_started_at = time.perf_counter()
    slide_phase_seconds = {
        "open_slide": 0.0,
        "load_annotations": 0.0,
        "load_artifacts": 0.0,
        "candidate_filter": 0.0,
        "build_indexes": 0.0,
        "parallel_processing": 0.0,
    }
    profile_output_path = kwargs.get("profile_output_path")
    profile_enabled = bool(profile_output_path)
    try:
        slide_basename = os.path.basename(path_Image)
        logging.info(f"--- Starting processing for slide: {slide_basename} ---")
        open_slide_started_at = time.perf_counter()
        slide = openslide_module.OpenSlide(path_Image)
        slide_phase_seconds["open_slide"] = time.perf_counter() - open_slide_started_at

        load_annotations_started_at = time.perf_counter()
        annotation_data = handler.load_annotations(slide, **kwargs)
        slide_phase_seconds["load_annotations"] = time.perf_counter() - load_annotations_started_at
        annotations_cancer_level0 = annotation_data["cancer_polygons"]
        annotations_not_cancer_level0 = annotation_data["not_cancer_polygons"]
        all_polygons_level0 = annotations_cancer_level0 + annotations_not_cancer_level0

        if not all_polygons_level0:
            logging.warning("No valid annotations found by handler for slide %s", slide_basename)
            slide.close()
            slide = None
            return 0, 0, []

        # --- NEW: ROBUST ARTIFACT PARSING BLOCK ---
        artifact_polygons_by_class_level0 = {}
        load_artifacts_started_at = time.perf_counter()
        if kwargs.get("use_artifact_filter") and kwargs.get("path_artifacts_geojson"):
            logging.info(f"Advanced artifact filtering is ACTIVE for {slide_basename}.")
            try:
                with open(kwargs["path_artifacts_geojson"]) as f:
                    artifact_data = json.load(f)

                for cls in ARTIFACT_CLASS_TO_COLUMN:
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
        slide_phase_seconds["load_artifacts"] = time.perf_counter() - load_artifacts_started_at

        scale_factor = slide.level_downsamples[kwargs["target_level"]]
        target_width, target_height = slide.level_dimensions[kwargs["target_level"]]

        scaled_polys_raw = []
        for polygon_points in all_polygons_level0:
            try:
                poly = Polygon([(x / scale_factor, y / scale_factor) for x, y in polygon_points])
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
            return 0, 0, []

        combined_annotations = MultiPolygon(scaled_polys_flat)
        min_x, min_y, max_x, max_y = combined_annotations.bounds
        x_start = max(0, int(min_x))
        y_start = max(0, int(min_y))
        x_end = min(int(max_x) + kwargs["window_size"], target_width)
        y_end = min(int(max_y) + kwargs["window_size"], target_height)

        logging.info(
            "Annotations bounding box (L%s): [(%s, %s), (%s, %s)]",
            kwargs["target_level"],
            int(min_x),
            int(min_y),
            int(max_x),
            int(max_y),
        )
        logging.info(f"Optimized scan area: [({x_start}, {y_start}), ({x_end}, {y_end})]")

        candidate_filter_started_at = time.perf_counter()
        x_coords = np.arange(x_start, x_end - kwargs["window_size"] + 1, kwargs["stride"])
        y_coords = np.arange(y_start, y_end - kwargs["window_size"] + 1, kwargs["stride"])
        center_x = x_coords + (kwargs["window_size"] // 2)
        center_y = y_coords + (kwargs["window_size"] // 2)
        center_grid_x, center_grid_y = np.meshgrid(center_x, center_y, indexing="ij")
        contains_mask = shapely.contains_xy(
            combined_annotations,
            center_grid_x.ravel(),
            center_grid_y.ravel(),
        )
        coord_grid = np.column_stack(
            (
                np.repeat(x_coords, len(y_coords)),
                np.tile(y_coords, len(x_coords)),
            )
        )
        filtered_coords = [
            (int(x_coord), int(y_coord))
            for (x_coord, y_coord), keep in zip(coord_grid, contains_mask, strict=False)
            if keep
        ]
        slide_phase_seconds["candidate_filter"] = time.perf_counter() - candidate_filter_started_at

        # The number of candidate windows will now be much more reasonable.
        logging.info(f"Found {len(filtered_coords)} candidate windows after optimization.")

        if not filtered_coords:
            slide.close()
            return 0, 0, []

        build_indexes_started_at = time.perf_counter()
        cancer_polygon_index = build_scaled_polygon_index(annotations_cancer_level0, scale_factor)
        not_cancer_polygon_index = build_scaled_polygon_index(
            annotations_not_cancer_level0, scale_factor
        )
        artifact_geometry_index = build_artifact_geometry_index(
            artifact_polygons_by_class_level0,
            scale_factor,
        )
        slide_phase_seconds["build_indexes"] = time.perf_counter() - build_indexes_started_at

        batch_size = max(8, min(64, len(filtered_coords) // max(1, kwargs["num_workers"] * 4) or 8))
        worker_state = {
            "path_Image": path_Image,
            "target_level": kwargs["target_level"],
            "window_size": kwargs["window_size"],
            "tissue_percentage_req": kwargs["tissue_percentage_req"],
            "match_percentage_req": kwargs["match_percentage_req"],
            "patient": kwargs["patient"],
            "slide_id": os.path.splitext(os.path.basename(path_Image))[0],
            "use_artifact_filter": kwargs.get("use_artifact_filter"),
            "profile_output_path": profile_output_path,
            "cancer_polygon_index": cancer_polygon_index,
            "not_cancer_polygon_index": not_cancer_polygon_index,
            "artifact_geometry_index": artifact_geometry_index,
        }

        logging.info(
            "Starting parallel processing with %s workers and batch size %s...",
            kwargs["num_workers"],
            batch_size,
        )
        parallel_started_at = time.perf_counter()
        status_counts = {}
        artifact_patch_records = []
        errors = []
        window_phase_stats: dict[str, PhaseStats] | None = (
            create_phase_stats(WINDOW_PROFILE_PHASES) if profile_enabled else None
        )
        for result in iter_window_results(
            filtered_coords=filtered_coords,
            num_workers=kwargs["num_workers"],
            worker_state=worker_state,
            batch_size=batch_size,
        ):
            status = result[0]
            payload = result[1]
            if profile_enabled and len(result) == 3 and window_phase_stats is not None:
                profile_result = cast(tuple[str, object, dict[str, PhaseStats]], result)
                result_phase_stats = profile_result[2]
                merge_phase_stats(window_phase_stats, result_phase_stats)

            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ERROR":
                errors.append(payload)
            elif payload is not None:
                artifact_patch_records.append(payload)
        slide_phase_seconds["parallel_processing"] = time.perf_counter() - parallel_started_at

        cancer_count = status_counts.get("SAVED_CANCER", 0)
        not_cancer_count = status_counts.get("SAVED_NOT_CANCER", 0)
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

        if profile_enabled and window_phase_stats is not None:
            profile_summary = build_profile_summary(
                slide_name=slide_basename,
                total_runtime_seconds=time.perf_counter() - slide_started_at,
                candidate_windows=len(filtered_coords),
                status_counts=status_counts,
                phase_stats=window_phase_stats,
                slide_phase_seconds=slide_phase_seconds,
            )
            write_profile_summary(Path(profile_output_path), profile_summary)

        logging.info(f"--- Finished processing slide: {slide_basename} ---")
        return cancer_count, not_cancer_count, artifact_patch_records
    finally:
        if slide:
            slide.close()
