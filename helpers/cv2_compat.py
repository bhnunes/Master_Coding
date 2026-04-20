from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.draw import polygon, polygon_perimeter
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.morphology import binary_closing, binary_erosion, binary_opening


def ensure_cv2_compat(cv2: Any) -> Any:
    if not hasattr(cv2, "error"):
        class CV2Error(Exception):
            pass

        cv2.error = CV2Error

    defaults = {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "COLOR_BGR2GRAY": 6,
        "COLOR_RGB2GRAY": 7,
        "COLOR_RGB2BGR": 8,
        "COLOR_BGR2RGB": 9,
        "COLOR_RGB2HSV": 10,
        "THRESH_BINARY": 0,
        "THRESH_OTSU": 8,
        "RETR_EXTERNAL": 0,
        "CHAIN_APPROX_NONE": 1,
        "CHAIN_APPROX_SIMPLE": 2,
        "INTER_AREA": 3,
        "MORPH_ELLIPSE": 4,
        "MORPH_OPEN": 5,
        "MORPH_CLOSE": 6,
    }
    for name, value in defaults.items():
        if not hasattr(cv2, name):
            setattr(cv2, name, value)

    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda value: None
    if not hasattr(cv2, "addWeighted"):
        cv2.addWeighted = _add_weighted
    if not hasattr(cv2, "imread"):
        cv2.imread = lambda path, flag=cv2.IMREAD_COLOR: _imread(path, flag, cv2)
    if not hasattr(cv2, "imwrite"):
        cv2.imwrite = _imwrite
    if not hasattr(cv2, "cvtColor"):
        cv2.cvtColor = lambda image, code: _cvt_color(image, code, cv2)
    if not hasattr(cv2, "resize"):
        cv2.resize = lambda image, size, interpolation=cv2.INTER_AREA: _resize(
            image,
            size,
            interpolation,
        )
    if not hasattr(cv2, "threshold"):
        cv2.threshold = lambda src, thresh, maxval, threshold_type: _threshold(
            src,
            thresh,
            maxval,
            threshold_type,
            cv2,
        )
    if not hasattr(cv2, "getStructuringElement"):
        cv2.getStructuringElement = _get_structuring_element
    if not hasattr(cv2, "morphologyEx"):
        cv2.morphologyEx = lambda src, op, kernel: _morphology_ex(src, op, kernel, cv2)
    if not hasattr(cv2, "erode"):
        cv2.erode = _erode
    if not hasattr(cv2, "findContours"):
        cv2.findContours = lambda image, mode, method: _find_contours(image)
    if not hasattr(cv2, "drawContours"):
        cv2.drawContours = _draw_contours
    if not hasattr(cv2, "fillPoly"):
        cv2.fillPoly = _fill_poly
    if not hasattr(cv2, "contourArea"):
        cv2.contourArea = _contour_area
    return cv2


def _imread(path: str, flag: int, cv2: Any) -> np.ndarray[Any, Any] | None:
    try:
        image = Image.open(path)
    except (FileNotFoundError, OSError):
        return None
    if flag == cv2.IMREAD_GRAYSCALE:
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if flag == cv2.IMREAD_UNCHANGED:
        return np.asarray(image, dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1]


def _imwrite(path: str, image: np.ndarray[Any, Any]) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        array = array[:, :, ::-1]
    Image.fromarray(array).save(destination)
    return True


def _cvt_color(image: np.ndarray[Any, Any], code: int, cv2: Any) -> np.ndarray[Any, Any]:
    array = np.asarray(image, dtype=np.uint8)
    if code in {cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB}:
        return array[:, :, ::-1]
    if code == cv2.COLOR_RGB2GRAY:
        return _rgb_to_gray(array)
    if code == cv2.COLOR_BGR2GRAY:
        return _rgb_to_gray(array[:, :, ::-1])
    if code == cv2.COLOR_RGB2HSV:
        hsv = rgb2hsv(array.astype(np.float32) / 255.0)
        hsv[:, :, 0] *= 179.0
        hsv[:, :, 1:] *= 255.0
        return np.asarray(np.round(hsv), dtype=np.uint8)
    raise cv2.error(f"Unsupported cvtColor code: {code}")


def _rgb_to_gray(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    gray = (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])
    return np.asarray(np.round(gray), dtype=np.uint8)


def _resize(
    image: np.ndarray[Any, Any],
    size: tuple[int, int],
    interpolation: int,
) -> np.ndarray[Any, Any]:
    del interpolation
    width, height = size
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _threshold(
    src: np.ndarray[Any, Any],
    thresh: float,
    maxval: float,
    threshold_type: int,
    cv2: Any,
) -> tuple[float, np.ndarray[Any, Any]]:
    array = np.asarray(src, dtype=np.uint8)
    effective_threshold = thresh
    if threshold_type & cv2.THRESH_OTSU:
        effective_threshold = float(threshold_otsu(array))  # type: ignore[no-untyped-call]
    binary = np.where(array > effective_threshold, maxval, 0).astype(np.uint8)
    return effective_threshold, binary


def _get_structuring_element(shape: int, ksize: tuple[int, int]) -> np.ndarray[Any, Any]:
    del shape
    return np.ones(ksize, dtype=np.uint8)


def _morphology_ex(
    src: np.ndarray[Any, Any],
    op: int,
    kernel: np.ndarray[Any, Any],
    cv2: Any,
) -> np.ndarray[Any, Any]:
    binary = np.asarray(src) > 0
    footprint = np.asarray(kernel) > 0
    if op == cv2.MORPH_OPEN:
        opened = binary_opening(binary, footprint=footprint)
        return cast(np.ndarray[Any, Any], (opened.astype(np.uint8) * 255).astype(np.uint8))
    if op == cv2.MORPH_CLOSE:
        closed = binary_closing(binary, footprint=footprint)
        return cast(np.ndarray[Any, Any], (closed.astype(np.uint8) * 255).astype(np.uint8))
    raise cv2.error(f"Unsupported morphologyEx op: {op}")


def _erode(src: np.ndarray[Any, Any], kernel: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    binary = np.asarray(src) > 0
    footprint = np.asarray(kernel) > 0
    eroded = binary_erosion(binary, footprint=footprint)
    return cast(np.ndarray[Any, Any], (eroded.astype(np.uint8) * 255).astype(np.uint8))


def _find_contours(
    image: np.ndarray[Any, Any],
) -> tuple[list[np.ndarray[Any, Any]], None]:
    contours = find_contours(np.asarray(image) > 0, level=0.5)  # type: ignore[no-untyped-call]
    converted = [
        np.flip(np.round(contour), axis=1).astype(np.int32).reshape(-1, 1, 2)
        for contour in contours
    ]
    return converted, None


def _draw_contours(
    image: np.ndarray[Any, Any],
    contours: list[np.ndarray[Any, Any]],
    contourIdx: int,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray[Any, Any]:
    del thickness
    canvas = np.asarray(image)
    selected = contours if contourIdx < 0 else [contours[contourIdx]]
    for contour in selected:
        points = np.asarray(contour).reshape(-1, 2)
        if len(points) < 2:
            continue
        rr, cc = polygon_perimeter(points[:, 1], points[:, 0], shape=canvas.shape[:2], clip=True)
        canvas[rr, cc] = np.asarray(color, dtype=np.uint8)
    return canvas


def _fill_poly(
    image: np.ndarray[Any, Any],
    points: list[np.ndarray[Any, Any]],
    color: tuple[int, ...],
) -> np.ndarray[Any, Any]:
    canvas = np.asarray(image)
    fill_value = int(color[0]) if color else 0
    for contour in points:
        vertices = np.asarray(contour).reshape(-1, 2)
        if len(vertices) < 3:
            continue
        rr, cc = polygon(  # type: ignore[no-untyped-call]
            vertices[:, 1],
            vertices[:, 0],
            shape=canvas.shape[:2],
        )
        canvas[rr, cc] = fill_value
    return canvas


def _add_weighted(
    src1: np.ndarray[Any, Any],
    alpha: float,
    src2: np.ndarray[Any, Any],
    beta: float,
    gamma: float,
) -> np.ndarray[Any, Any]:
    result = (alpha * src1.astype(np.float32)) + (beta * src2.astype(np.float32)) + gamma
    return np.clip(result, 0, 255).astype(np.uint8)


def _contour_area(contour: np.ndarray[Any, Any]) -> float:
    points = np.asarray(contour).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    x_coords = points[:, 0].astype(np.float64)
    y_coords = points[:, 1].astype(np.float64)
    area = np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))
    return float(0.5 * abs(area))
