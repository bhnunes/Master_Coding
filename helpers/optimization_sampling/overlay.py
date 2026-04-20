from __future__ import annotations

import atexit
import logging
import sys
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from helpers.cv2_compat import ensure_cv2_compat
from helpers.optimization_sampling.sampling import OverlayTask

cv2 = ensure_cv2_compat(cv2)

_WORKER_HDF5_HANDLES: dict[Path, h5py.File] = {}
_WORKER_HDF5_CLEANUP_REGISTERED = False
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5


def overlay_mask_edges(
    image_path: Path | str,
    mask_path: Path | str,
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
        image, mask = _read_overlay_sources(image_path, mask_path)
        if image is None:
            active_logger.error("Could not read image: %s", image_path)
            return False
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
                Path(str(image_path)).name,
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
            Path(str(image_path)).name,
            error,
        )
        return False


def generate_overlay_images(tasks: Sequence[OverlayTask], num_processes: int) -> list[bool]:
    """Generate overlay images in parallel for the selected sample tasks."""

    if not tasks:
        return []

    ordered_tasks = sorted(tasks, key=_overlay_task_order_key)
    with Pool(processes=num_processes) as pool:
        return list(
            tqdm(
                pool.imap_unordered(_process_overlay_task, ordered_tasks, chunksize=1),
                total=len(ordered_tasks),
                desc="Generating Samples",
                mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
                dynamic_ncols=True,
                file=_progress_file(),
                disable=_progress_disabled(),
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


def _progress_file() -> Any:
    """Use the real terminal stream so tqdm stays interactive under redirected stdout."""

    return sys.__stderr__


def _progress_disabled() -> bool:
    isatty = getattr(_progress_file(), "isatty", None)
    return not bool(isatty() if callable(isatty) else False)


def _to_grayscale_mask(mask: Any) -> Any:
    if getattr(mask, "ndim", 0) == 3 and getattr(mask, "shape", (0, 0, 0))[2] == 4:
        return mask[:, :, 3]
    if getattr(mask, "ndim", 0) == 3:
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _parse_hdf5_ref(source: Path | str, dataset_name: str) -> tuple[Path, int] | None:
    text = str(source)
    marker = f"::{dataset_name}["
    if marker not in text or not text.endswith("]"):
        return None
    path_text, index_text = text.split(marker, maxsplit=1)
    return Path(path_text), int(index_text[:-1])


def _overlay_task_order_key(task: OverlayTask) -> tuple[int, str, int, str]:
    image_ref = _parse_hdf5_ref(task.image_path, "images")
    if image_ref is None:
        return (1, str(task.image_path), 0, str(task.output_path))
    source_hdf5_path, row_index = image_ref
    return (0, str(source_hdf5_path), row_index, str(task.output_path))


def _read_overlay_sources(image_path: Path | str, mask_path: Path | str) -> tuple[Any, Any]:
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
    global _WORKER_HDF5_CLEANUP_REGISTERED

    resolved_path = source_hdf5_path.resolve()
    handle = _WORKER_HDF5_HANDLES.get(resolved_path)
    if handle is None:
        handle = h5py.File(resolved_path, "r")
        _WORKER_HDF5_HANDLES[resolved_path] = handle
        if not _WORKER_HDF5_CLEANUP_REGISTERED:
            atexit.register(_close_worker_hdf5_handles)
            _WORKER_HDF5_CLEANUP_REGISTERED = True
    return handle


def _close_worker_hdf5_handles() -> None:
    for handle in _WORKER_HDF5_HANDLES.values():
        handle.close()
    _WORKER_HDF5_HANDLES.clear()


def _read_image_source(image_path: Path | str) -> Any:
    parsed = _parse_hdf5_ref(image_path, "images")
    if parsed is None:
        return cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    source_hdf5_path, row_index = parsed
    handle = _get_cached_hdf5_handle(source_hdf5_path)
    handle_any = cast(Any, handle)
    images = handle_any["images"]
    image = np.asarray(images[row_index], dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _read_mask_source(mask_path: Path | str) -> Any:
    parsed = _parse_hdf5_ref(mask_path, "masks")
    if parsed is None:
        return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    source_hdf5_path, row_index = parsed
    handle = _get_cached_hdf5_handle(source_hdf5_path)
    handle_any = cast(Any, handle)
    masks = handle_any["masks"]
    mask = np.asarray(masks[row_index], dtype=np.uint8)
    return (mask * 255).astype(np.uint8)
