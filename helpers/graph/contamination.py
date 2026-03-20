from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np


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
    image_path: Path,
    mask_path: Path,
    params: GraphContaminationParameters | dict[str, object],
    *,
    logger: logging.Logger | None = None,
) -> float | None:
    """Calculate ROI contamination rate for one image/mask pair."""

    active_logger = logger or logging.getLogger("graph_tuning")
    normalized_params = coerce_graph_contamination_parameters(params)
    base_name = image_path.name

    try:
        image: Any = cv2.imread(str(image_path))
        if image is None:
            active_logger.debug("[%s] FAILED: cv2.imread returned None for image path.", base_name)
            return None

        roi_mask: Any = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            active_logger.debug("[%s] FAILED: cv2.imread returned None for mask path.", base_name)
            return None

        _, thresholded_mask = cv2.threshold(roi_mask, 1, 255, cv2.THRESH_BINARY)
        roi_binary = (thresholded_mask > 0).astype(np.uint8)
        initial_roi_area = int(roi_binary.sum())

        if normalized_params.erosion_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * normalized_params.erosion_px + 1, 2 * normalized_params.erosion_px + 1),
            )
            roi_binary = cv2.erode(roi_binary, kernel)

        final_roi_area = int(roi_binary.sum())
        if final_roi_area == 0:
            if initial_roi_area > 0:
                active_logger.debug(
                    "[%s] FAILED: ROI area became zero after erosion of %spx.",
                    base_name,
                    normalized_params.erosion_px,
                )
            else:
                active_logger.debug("[%s] FAILED: Initial ROI area was zero.", base_name)
            return float("nan")

        ximgproc = cast(Any, cv2).ximgproc
        segmentator = ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5,
            k=normalized_params.k,
            min_size=normalized_params.min_size,
        )
        segment_map = segmentator.processImage(image)
        background_mask = np.zeros(image.shape[:2], dtype=bool)
        num_segments = int(np.max(segment_map)) + 1

        for segment_id in range(num_segments):
            segment_pixel_mask = segment_map == segment_id
            if not np.any(segment_pixel_mask):
                continue
            avg_color = cv2.mean(image, mask=segment_pixel_mask.astype(np.uint8))
            avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
            if avg_intensity > normalized_params.bg_intensity_thresh:
                background_mask[segment_pixel_mask] = True

        contamination_pixels = int((background_mask & roi_binary.astype(bool)).sum())
        contamination_rate = contamination_pixels / final_roi_area
        active_logger.debug(
            "[%s] SUCCESS: Rate=%.4f with params k=%s, min_size=%s, erosion=%s.",
            base_name,
            contamination_rate,
            normalized_params.k,
            normalized_params.min_size,
            normalized_params.erosion_px,
        )
        return contamination_rate
    except cv2.error as error:
        active_logger.error("[%s] CRITICAL EXCEPTION: %s", base_name, error, exc_info=True)
        return None
