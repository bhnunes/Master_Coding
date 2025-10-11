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
from dotenv import load_dotenv
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from imantics import Mask

# --- All imports and OpenSlide initialization remain the same ---
load_dotenv(override=True)
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')

try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except ImportError as e:
    logging.error(f"Error importing OpenSlide: {e}")
    sys.exit(1)
# --------------------------- Logging ---------------------------

# put near your other imports, before calling TiffFile(...)
try:
    import imagecodecs  # registers LZW/Deflate/JPEG2000/etc. with tifffile
except Exception as e:
    logging.error("imagecodecs not available: %s", e)
    raise

import zarr
from tifffile import TiffFile


def to_uint8_norm(a: np.ndarray) -> np.uint8:
    vmax_local = int(a.max())
    if vmax_local <= 0:
        return (a.astype(np.uint8))
    scale = 255.0 / vmax_local
    return (np.clip(a * scale, 0, 255)).astype(np.uint8)


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

def _collect_all_arrays_zarr(root) -> list:
    """
    Robustly collect all zarr Arrays from a root that may be a Group or Array,
    handling Zarr v2/v3 API differences without using .values().
    """
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        # If it's already an Array, record it
        if isinstance(node, zarr.Array):
            out.append(node)
            continue

        # Try mapping-style iteration of keys (works in v2/v3)
        try:
            for key in node:           # iter keys
                try:
                    child = node[key]  # __getitem__
                    stack.append(child)
                except Exception:
                    pass
        except Exception:
            pass

        # Try v2-style helpers if present
        if hasattr(node, "arrays"):
            try:
                for _, arr in node.arrays():
                    out.append(arr)
            except Exception:
                pass
        if hasattr(node, "groups"):
            try:
                for _, grp in node.groups():
                    stack.append(grp)
            except Exception:
                pass

        # Last-ditch: try common names used for levels
        for key in ("0", "level0", "0/0"):
            try:
                child = node[key]
                stack.append(child)
            except Exception:
                pass

    return out

def load_mask_decimated(mask_path: str,
                        max_dim: int = 16384,
                        threshold: int = 128) -> tuple[np.ndarray, int, int]:
    """
    Load a huge TIFF mask by decimating on read using tifffile + zarr.
    Works whether aszarr() returns a Zarr Array or a Group (pyramidal TIFF), across Zarr v2/v3.
    Returns (mask_bin_small, sx, sy) where sx,sy scale coords back to level-0.
    """
    with TiffFile(mask_path) as tf:
        store = tf.series[0].aszarr()     # may be Array OR Group
        root = zarr.open(store, mode="r")

        arrs = _collect_all_arrays_zarr(root)
        if not arrs:
            raise ValueError(f"No arrays found in zarr store for {mask_path}")

        # Choose the *largest* array (assumed level-0)
        arr = max(arrs, key=lambda a: int(np.prod(a.shape)))

        # Expect (H,W) or (H,W,C) or (C,H,W)
        if arr.ndim == 2:
            z_mono = arr
            h, w = arr.shape
        elif arr.ndim == 3:
            sh = arr.shape
            # If first axis looks like channels (1/3/4) and smaller than others -> transpose to (H,W,C)
            if sh[0] in (1, 3, 4) and sh[0] < sh[1] and sh[0] < sh[2]:
                z_hw_first = arr.transpose(1, 2, 0)   # lazy
                z_mono = z_hw_first[:, :, 0]
                h, w = z_hw_first.shape[:2]
            else:
                # Assume (H,W,C)
                z_mono = arr[:, :, 0]
                h, w = sh[0], sh[1]
        else:
            raise ValueError(f"Unsupported zarr array ndim={arr.ndim} for {mask_path}")

        # Decimate so longest side ~ max_dim
        s = max(1, int(np.ceil(max(h, w) / max_dim)))

        # Lazy stepped slice; reads only ~1/s^2 pixels
        small = np.asarray(z_mono[::s, ::s])

        # --- Robust binarization (handles {0,1}, {0,2}, {0,255}, 16-bit, etc.) ---
        vmin, vmax = int(small.min()), int(small.max())

        # Sample unique values (full unique is fine at this size)
        vals = np.unique(small)
        nvals = vals.size

        if small.dtype == np.bool_:
            m_bin = small.astype(np.uint8)

        elif vmax <= 5 or nvals <= 8:
            # Low-dynamic-range integer mask (e.g., {0,1} or {0,2} or {0,255} collapsed on read):
            # Treat any non-zero as foreground.
            m_bin = (small != 0).astype(np.uint8)

        elif small.dtype == np.uint16 and vmax > 255:
            # Likely 16-bit mask {0,65535}. Scale the user threshold (default 128/255).
            thr_ref = 128 if threshold is None else int(threshold)
            thr = int(round((thr_ref / 255.0) * vmax))
            thr = max(1, min(thr, vmax))
            m_bin = (small >= thr).astype(np.uint8)

        else:
            # 8-bit or normalized case; prefer Otsu to avoid magic numbers.
            try:
                import cv2
                # Ensure uint8 input for Otsu
                small_u8 = small if small.dtype == np.uint8 else to_uint8_norm(small)
                _, m_bin = cv2.threshold(small_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                m_bin = m_bin.astype(np.uint8)
            except Exception:
                # Fallback to mid-range or provided threshold
                thr = (vmin + vmax) // 2 if threshold is None else int(threshold)
                m_bin = (small >= thr).astype(np.uint8)

        logging.info(
            "Mask stats: dtype=%s range=[%d,%d] uniques=%s -> positives=%d",
            small.dtype, vmin, vmax, (vals[:10].tolist() if nvals <= 10 else f'{nvals} values'),
            int(m_bin.sum())
        )

        logging.info("Mask %s level0 shape=%s; decimation=%dx -> approx %dx%d",
                os.path.basename(mask_path), arr.shape, s, (w + s - 1)//s, (h + s - 1)//s)
    return m_bin, s, s


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
    with openslide.OpenSlide(slide_path) as slide:
        w0, h0 = slide.level_dimensions[0]

    # Load downsampled mask and remember the scale
    mask_small, sx, sy = load_mask_decimated(mask_path, max_dim=16384, threshold=threshold)

    logging.info("After binarize: sum=%d, shape=%s, s=(%d,%d)",
             int(mask_small.sum()), mask_small.shape, sx, sy)

    mode = infer_cam16_mode_strict(slide_path, override=mode_override)

    polys_small = mask_to_polygons_imantics(mask_small, min_area_px / (sx * sy), simplify_tolerance)

    # scale polygons back to level-0 coordinates
    def scale_poly(poly):
        return [(x * sx, y * sy) for (x, y) in poly]

    if mode == "tumor":
        cancer_polys = [scale_poly(p) for p in polys_small]
        not_cancer_polys = []
    else:
        not_cancer_polys = [scale_poly(p) for p in polys_small]
        cancer_polys = []

    return {"cancer_polygons": cancer_polys, "not_cancer_polygons": not_cancer_polys}


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
    slides_dir = r"D:\Usuario\Desktop\CAMELYON16\slides"
    out_dir = r"D:\Usuario\Desktop\CAMELYON16\ANNOTATIONS"
    slide_glob = "*.tif"              # e.g., "tumor_*.tif" or "normal_*.tif"
    threshold = 128
    min_area_px = 100.0
    simplify_tolerance = 1.0
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
