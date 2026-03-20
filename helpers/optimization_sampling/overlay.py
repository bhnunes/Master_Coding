from __future__ import annotations

import logging
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from helpers.optimization_sampling.sampling import OverlayTask


def overlay_mask_edges(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    alpha: float = 1.0,
    logger: logging.Logger | None = None,
) -> bool:
    """Draw the boundary of a mask on top of an image and save the result."""

    active_logger = logger or logging.getLogger("optimization_sampling")
    try:
        image: Any = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            active_logger.error("Could not read image: %s", image_path)
            return False

        mask: Any = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            active_logger.error("Could not read mask: %s", mask_path)
            return False

        mask_gray = _to_grayscale_mask(mask)
        _, mask_binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
        if mask_binary.shape[:2] != image.shape[:2]:
            active_logger.error(
                "Size mismatch: image=%s vs mask=%s for %s",
                image.shape[:2],
                mask_binary.shape[:2],
                image_path.name,
            )
            return False

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        overlay = image.copy()
        cv2.drawContours(overlay, contours, contourIdx=-1, color=color, thickness=thickness)
        result = (
            cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0) if alpha < 1.0 else overlay
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), result))
    except cv2.error as error:
        active_logger.error(
            "Failed to process and save overlay for %s: %s",
            image_path.name,
            error,
        )
        return False


def generate_overlay_images(tasks: Sequence[OverlayTask], num_processes: int) -> list[bool]:
    """Generate overlay images in parallel for the selected sample tasks."""

    if not tasks:
        return []

    with Pool(processes=num_processes) as pool:
        return list(
            tqdm(
                pool.imap_unordered(_process_overlay_task, tasks),
                total=len(tasks),
                desc="Generating Samples",
            )
        )


def _process_overlay_task(task: OverlayTask) -> bool:
    return overlay_mask_edges(
        image_path=task.image_path,
        mask_path=task.mask_path,
        output_path=task.output_path,
        color=task.color,
        thickness=task.thickness,
        alpha=task.alpha,
    )


def _to_grayscale_mask(mask: Any) -> Any:
    if getattr(mask, "ndim", 0) == 3 and getattr(mask, "shape", (0, 0, 0))[2] == 4:
        return mask[:, :, 3]
    if getattr(mask, "ndim", 0) == 3:
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask
