from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt
from skimage.segmentation import felzenszwalb

from helpers.cv2_compat import ensure_cv2_compat

cv2 = ensure_cv2_compat(cv2)

_HDF5_HANDLES: dict[Path, h5py.File] = {}
_HDF5_CLEANUP_REGISTERED = False


@dataclass(frozen=True)
class GraphContaminationParameters:
    """Graph-segmentation parameters used to estimate ROI contamination."""

    bg_intensity_thresh: int
    k: float
    min_size: int
    erosion_px: int


def coerce_graph_contamination_parameters(
    params: GraphContaminationParameters | dict[str, object],
) -> GraphContaminationParameters:
    """Normalize graph parameters to stable Python scalar types."""

    if isinstance(params, GraphContaminationParameters):
        return params
    bg_intensity_thresh = cast("int | float | str", params["bg_intensity_thresh"])
    k_value = cast("int | float | str", params["k"])
    min_size = cast("int | float | str", params["min_size"])
    erosion_px = cast("int | float | str", params.get("erosion_px", 0))
    return GraphContaminationParameters(
        bg_intensity_thresh=int(bg_intensity_thresh),
        k=float(k_value),
        min_size=int(min_size),
        erosion_px=int(erosion_px),
    )


def calculate_roi_contamination(
    image_path: Path | str,
    mask_path: Path | str,
    params: GraphContaminationParameters | dict[str, object],
    *,
    logger: logging.Logger | None = None,
) -> float | None:
    """Calculate ROI contamination rate for one image/mask pair."""

    active_logger = logger or logging.getLogger("graph_tuning")
    normalized_params = coerce_graph_contamination_parameters(params)
    base_name = Path(str(image_path)).name

    try:
        image, roi_mask = _read_sources(image_path, mask_path)
        if image is None:
            active_logger.debug("[%s] FAILED: cv2.imread returned None for image path.", base_name)
            return None
        if roi_mask is None:
            active_logger.debug("[%s] FAILED: cv2.imread returned None for mask path.", base_name)
            return None

        return _calculate_contamination_rate(
            image=image,
            roi_mask=roi_mask,
            params=normalized_params,
            base_name=base_name,
            logger=active_logger,
        )
    except (cv2.error, ValueError, RuntimeError) as error:
        active_logger.error("[%s] CRITICAL EXCEPTION: %s", base_name, error, exc_info=True)
        return None


def calculate_roi_contamination_from_arrays(
    image: npt.NDArray[np.uint8],
    roi_mask: npt.NDArray[np.uint8],
    params: GraphContaminationParameters | dict[str, object],
    *,
    logger: logging.Logger | None = None,
    base_name: str = "preloaded_record",
) -> float | None:
    """Calculate ROI contamination from already-loaded image and mask arrays."""

    active_logger = logger or logging.getLogger("graph_tuning")
    normalized_params = coerce_graph_contamination_parameters(params)

    try:
        return _calculate_contamination_rate(
            image=image,
            roi_mask=roi_mask,
            params=normalized_params,
            base_name=base_name,
            logger=active_logger,
        )
    except (cv2.error, ValueError, RuntimeError) as error:
        active_logger.error("[%s] CRITICAL EXCEPTION: %s", base_name, error, exc_info=True)
        return None


def load_graph_sources(
    image_path: Path | str,
    mask_path: Path | str,
) -> tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.uint8] | None]:
    """Load one image/mask pair from filesystem or HDF5."""

    image, mask = _read_sources(image_path, mask_path)
    return cast("npt.NDArray[np.uint8] | None", image), cast("npt.NDArray[np.uint8] | None", mask)


def load_graph_source_batch(
    source_hdf5_path: Path | str,
    start_index: int,
    end_index: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load a contiguous HDF5 image/mask batch for graph cleaning."""

    handle = _get_cached_hdf5_handle(Path(source_hdf5_path))
    handle_any = cast(Any, handle)
    images = cast(
        npt.NDArray[np.uint8],
        np.asarray(handle_any["images"][start_index:end_index], dtype=np.uint8),
    )
    masks = cast(
        npt.NDArray[np.uint8],
        np.asarray(handle_any["masks"][start_index:end_index], dtype=np.uint8),
    )
    return images[..., ::-1], cast(npt.NDArray[np.uint8], np.asarray(masks * 255, dtype=np.uint8))


def _calculate_contamination_rate(
    *,
    image: npt.NDArray[np.uint8],
    roi_mask: npt.NDArray[np.uint8],
    params: GraphContaminationParameters,
    base_name: str,
    logger: logging.Logger,
) -> float:
    _, thresholded_mask = cv2.threshold(roi_mask, 1, 255, cv2.THRESH_BINARY)
    roi_binary = (thresholded_mask > 0).astype(np.uint8)
    initial_roi_area = int(roi_binary.sum())

    if params.erosion_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * params.erosion_px + 1, 2 * params.erosion_px + 1),
        )
        roi_binary = np.asarray(cv2.erode(roi_binary, kernel), dtype=np.uint8)

    final_roi_area = int(roi_binary.sum())
    if final_roi_area == 0:
        if initial_roi_area > 0:
            logger.debug(
                "[%s] FAILED: ROI area became zero after erosion of %spx.",
                base_name,
                params.erosion_px,
            )
        else:
            logger.debug("[%s] FAILED: Initial ROI area was zero.", base_name)
        return float("nan")

    segment_map = _segment_image(image, params)
    background_mask = _classify_background_segments(
        image=image,
        segment_map=segment_map,
        bg_intensity_thresh=params.bg_intensity_thresh,
    )
    contamination_pixels = int((background_mask & roi_binary.astype(bool)).sum())
    contamination_rate = contamination_pixels / final_roi_area
    logger.debug(
        "[%s] SUCCESS: Rate=%.4f with params k=%s, min_size=%s, erosion=%s.",
        base_name,
        contamination_rate,
        params.k,
        params.min_size,
        params.erosion_px,
    )
    return contamination_rate


def _classify_background_segments(
    *,
    image: npt.NDArray[np.uint8],
    segment_map: npt.NDArray[np.int32],
    bg_intensity_thresh: int,
) -> npt.NDArray[np.bool_]:
    flat_segments = segment_map.reshape(-1)
    flat_image = image.reshape(-1, 3).astype(np.float64, copy=False)
    counts = np.bincount(flat_segments)
    blue_sum = np.bincount(flat_segments, weights=flat_image[:, 0], minlength=counts.shape[0])
    green_sum = np.bincount(flat_segments, weights=flat_image[:, 1], minlength=counts.shape[0])
    red_sum = np.bincount(flat_segments, weights=flat_image[:, 2], minlength=counts.shape[0])
    avg_intensity = (blue_sum + green_sum + red_sum) / (3.0 * counts)
    return (avg_intensity > bg_intensity_thresh)[segment_map]


def _segment_image(
    image: npt.NDArray[np.uint8], params: GraphContaminationParameters
) -> npt.NDArray[np.int32]:
    """Return Felzenszwalb graph segments for one RGB image."""

    segment_map = felzenszwalb(
        image,
        scale=params.k,
        sigma=0.5,
        min_size=params.min_size,
    )
    return np.asarray(segment_map, dtype=np.int32)


def _parse_hdf5_ref(source: Path | str, dataset_name: str) -> tuple[Path, int] | None:
    text = str(source)
    marker = f"::{dataset_name}["
    if marker not in text or not text.endswith("]"):
        return None
    path_text, index_text = text.split(marker, maxsplit=1)
    return Path(path_text), int(index_text[:-1])


def _read_sources(
    image_path: Path | str,
    mask_path: Path | str,
) -> tuple[Any, Any]:
    image_ref = _parse_hdf5_ref(image_path, "images")
    mask_ref = _parse_hdf5_ref(mask_path, "masks")
    if image_ref is not None and mask_ref is not None and image_ref == mask_ref:
        source_hdf5_path, row_index = image_ref
        handle = _get_cached_hdf5_handle(source_hdf5_path)
        handle_any = cast(Any, handle)
        images = handle_any["images"]
        masks = handle_any["masks"]
        image = np.asarray(images[row_index], dtype=np.uint8)
        mask = np.asarray(masks[row_index], dtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (mask * 255).astype(np.uint8)
    return _read_image_source(image_path), _read_mask_source(mask_path)


def _get_cached_hdf5_handle(source_hdf5_path: Path) -> h5py.File:
    global _HDF5_CLEANUP_REGISTERED

    resolved_path = source_hdf5_path.resolve()
    handle = _HDF5_HANDLES.get(resolved_path)
    if handle is None:
        handle = h5py.File(resolved_path, "r")
        _HDF5_HANDLES[resolved_path] = handle
        if not _HDF5_CLEANUP_REGISTERED:
            atexit.register(_close_hdf5_handles)
            _HDF5_CLEANUP_REGISTERED = True
    return handle


def _close_hdf5_handles() -> None:
    for handle in _HDF5_HANDLES.values():
        handle.close()
    _HDF5_HANDLES.clear()


def _read_image_source(image_path: Path | str) -> Any:
    parsed = _parse_hdf5_ref(image_path, "images")
    if parsed is None:
        return cv2.imread(str(image_path))
    source_hdf5_path, row_index = parsed
    handle = _get_cached_hdf5_handle(source_hdf5_path)
    handle_any = cast(Any, handle)
    images = handle_any["images"]
    image = np.asarray(images[row_index], dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _read_mask_source(mask_path: Path | str) -> Any:
    parsed = _parse_hdf5_ref(mask_path, "masks")
    if parsed is None:
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    source_hdf5_path, row_index = parsed
    handle = _get_cached_hdf5_handle(source_hdf5_path)
    handle_any = cast(Any, handle)
    masks = handle_any["masks"]
    mask = np.asarray(masks[row_index], dtype=np.uint8)
    return (mask * 255).astype(np.uint8)
