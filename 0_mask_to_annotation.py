# -*- coding: utf-8 -*-
# pip install imantics shapely opencv-python numpy openslide-python
import os
import sys
import json
import cv2
import logging
import multiprocessing as mp
from pathlib import Path
from glob import glob
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from imantics import Mask
import openslide


# --------------------------- Logging ---------------------------

def setup_logging(logfile: str = "data_preparation.log", level=logging.INFO) -> None:
    """
    Configures the root logger to output to both a file and the console.
    Includes process name to distinguish parallel workers.
    """
    log_format = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    # avoid duplicate handlers if called twice (main + workers)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    logging.info("Logging configured. Output will be saved to %s", logfile)


# --------------------------- Core helpers ---------------------------

def infer_cam16_mode_strict(path_Image: str, override: Optional[str] = None) -> str:
    """
    Strict CAMELYON16: must be tumor or normal. Otherwise raise.
    """
    if override is not None:
        ov = override.lower()
        if ov in ("tumor", "normal"):
            return ov
        raise ValueError(f"Invalid mode_override={override!r}; expected 'tumor' or 'normal'.")

    base = os.path.basename(path_Image).lower() if path_Image else ""
    if "tumor" in base:
        return "tumor"
    if "normal" in base:
        return "normal"
    raise ValueError(f"Cannot infer mode from filename: {path_Image!r}. "
                     f"Expected it to contain 'tumor' or 'normal'.")


def binarize_and_resize_mask(mask_path: str,
                             target_w: Optional[int],
                             target_h: Optional[int],
                             threshold: int = 128) -> np.ndarray:
    """
    Load mask as grayscale, binarize to {0,1}, resize to (target_w, target_h) with nearest-neighbor
    if target size is provided. Resize happens AFTER binarization to preserve clean regions.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

    m_bin = (m > int(threshold)).astype(np.uint8)  # 0 or 1

    if target_w is not None and target_h is not None:
        m_bin = cv2.resize(m_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return m_bin


def mask_to_polygons_imantics(mask_bin: np.ndarray,
                              min_area_px: float = 100.0,
                              simplify_tolerance: Optional[float] = 1.5) -> List[List[Tuple[float, float]]]:
    """
    Convert binary mask (0/1) to polygons using imantics.
    Returns: list of polygons, each polygon is a list of (x, y) in slide level-0 coordinates.
    """
    polys = Mask(mask_bin).polygons()
    results: List[List[Tuple[float, float]]] = []

    for pts in polys.points:  # each: Nx2
        shp = ShapelyPolygon([(float(x), float(y)) for x, y in pts])
        if not shp.is_valid:
            shp = shp.buffer(0)
        if shp.is_empty or shp.geom_type != "Polygon":
            continue
        if shp.area < min_area_px:
            continue
        if simplify_tolerance and simplify_tolerance > 0:
            shp = shp.simplify(simplify_tolerance, preserve_topology=True)
            if shp.is_empty or shp.geom_type != "Polygon":
                continue
        results.append([(float(x), float(y)) for x, y in shp.exterior.coords])
    return results


def cam16_pair_to_annotations(slide_path: str,
                              mask_path: str,
                              threshold: int = 128,
                              min_area_px: float = 100.0,
                              simplify_tolerance: Optional[float] = 1.5,
                              mode_override: Optional[str] = None) -> Dict[str, Any]:
    """
    For a single (slide, mask) pair:
      - Read level-0 (W,H)
      - Resize mask to (W,H) BEFORE polygonization (enforced)
      - Enforce strict CAMELYON16 semantics to return exclusive polygons
    """
    with openslide.OpenSlide(slide_path) as slide:
        w0, h0 = slide.level_dimensions[0]

    mask_bin = binarize_and_resize_mask(mask_path, w0, h0, threshold)
    mode = infer_cam16_mode_strict(slide_path, override=mode_override)

    if mode == "tumor":
        cancer_polys = mask_to_polygons_imantics(mask_bin, min_area_px, simplify_tolerance)
        not_cancer_polys = []
    else:  # "normal"
        not_cancer_polys = mask_to_polygons_imantics(mask_bin, min_area_px, simplify_tolerance)
        cancer_polys = []

    return {
        "cancer_polygons": cancer_polys,
        "not_cancer_polygons": not_cancer_polys
    }


# --------------------------- Batch & Multiprocessing ---------------------------

def mask_name_from_slide(slide_path: str) -> Optional[str]:
    """
    Customize this resolver to your naming scheme.
    Example layout:
        slides: /data/camelyon16/slides/tumor_001.tif
        masks:  /data/camelyon16/masks/tumor_001_mask.tif
    """
    p = Path(slide_path)
    stem = p.stem  # e.g., tumor_001
    candidate = p.parent.parent / "masks" / f"{stem}_mask.tif"
    return str(candidate) if candidate.exists() else None


def process_one(slide_path: str,
                mask_path: str,
                out_dir: str,
                threshold: int,
                min_area_px: float,
                simplify_tolerance: Optional[float],
                mode_override: Optional[str],
                save_per_slide: bool) -> Dict[str, Any]:
    """
    Worker function: processes one slide/mask pair, writes JSON (optional), returns a small summary.
    Designed to run inside a separate process.
    """
    # Each process must configure its own logger handlers
    setup_logging()

    slide_stem = Path(slide_path).stem
    logging.info("Start slide: %s", slide_stem)

    ann = cam16_pair_to_annotations(
        slide_path=slide_path,
        mask_path=mask_path,
        threshold=threshold,
        min_area_px=min_area_px,
        simplify_tolerance=simplify_tolerance,
        mode_override=mode_override
    )

    if save_per_slide:
        out_path = os.path.join(out_dir, f"{slide_stem}_polygons.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False)
        logging.info("Wrote %s", out_path)

    summary = {
        "slide": slide_path,
        "mask": mask_path,
        "num_cancer_polygons": len(ann["cancer_polygons"]),
        "num_not_cancer_polygons": len(ann["not_cancer_polygons"]),
    }
    logging.info("Done slide: %s | cancer=%d non_cancer=%d",
                 slide_stem, summary["num_cancer_polygons"], summary["num_not_cancer_polygons"])
    return summary


def build_pairs(slides_dir: str, slide_glob: str) -> List[Tuple[str, str]]:
    slides = sorted(glob(os.path.join(slides_dir, slide_glob)))
    pairs: List[Tuple[str, str]] = []
    for s in slides:
        m = mask_name_from_slide(s)
        if m is None:
            logging.warning("No mask found for slide: %s", s)
            continue
        pairs.append((s, m))
    logging.info("Prepared %d pairs", len(pairs))
    return pairs


def process_directory(slides_dir: str,
                      out_dir: str,
                      slide_glob: str = "*.tif",
                      threshold: int = 128,
                      min_area_px: float = 100.0,
                      simplify_tolerance: Optional[float] = 1.5,
                      mode_override: Optional[str] = None,
                      save_per_slide: bool = True,
                      max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Multiprocess driver over a directory of slides. Adjust mask pairing logic if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_logging()  # main process logging

    pairs = build_pairs(slides_dir, slide_glob)
    if not pairs:
        logging.error("No slide/mask pairs discovered. Check paths and naming.")
        return []

    # Reasonable default: CPU count, leave 1 core free
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    logging.info("Launching processing with %d workers", max_workers)

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [
            ex.submit(
                process_one,
                slide_path, mask_path, out_dir,
                threshold, min_area_px, simplify_tolerance, mode_override, save_per_slide
            )
            for slide_path, mask_path in pairs
        ]

        completed = 0
        total = len(futures)
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logging.info("Progress: %d/%d finished", completed, total)
            except Exception as e:
                logging.exception("Worker exception: %s", e)

    combined_path = os.path.join(out_dir, "summary_all_slides.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    logging.info("Wrote combined summary: %s", combined_path)

    return results


# --------------------------- Main ---------------------------

def main() -> None:
    # Global environment (set here, no argparse)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # ==== CONFIGURATION SECTION ====
    slides_dir = "/data/camelyon16/slides"
    out_dir = "/data/camelyon16/annotations_json"
    slide_glob = "*.tif"              # e.g., "tumor_*.tif" or "normal_*.tif"
    threshold = 128
    min_area_px = 100.0
    simplify_tolerance = 1.5
    mode_override = None              # or "tumor" / "normal" to force
    save_per_slide = True
    max_workers = None                # or an int like 8
    # ===============================

    process_directory(
        slides_dir=slides_dir,
        out_dir=out_dir,
        slide_glob=slide_glob,
        threshold=threshold,
        min_area_px=min_area_px,
        simplify_tolerance=simplify_tolerance,
        mode_override=mode_override,
        save_per_slide=save_per_slide,
        max_workers=max_workers
    )


if __name__ == "__main__":
    mp.freeze_support()  # Windows/Jupyter safety
    main()
