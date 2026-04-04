from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from helpers.graph.contamination import (
    GraphContaminationParameters,
    calculate_roi_contamination,
    coerce_graph_contamination_parameters,
)


def test_coerce_graph_contamination_parameters_casts_numpy_values() -> None:
    params = coerce_graph_contamination_parameters(
        {
            "bg_intensity_thresh": np.int64(198),
            "k": np.float64(386),
            "min_size": np.int64(200),
            "erosion_px": np.int64(0),
        }
    )

    assert params == GraphContaminationParameters(
        bg_intensity_thresh=198,
        k=386.0,
        min_size=200,
        erosion_px=0,
    )


def test_calculate_roi_contamination_returns_none_when_image_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "helpers.graph.contamination.cv2.imread",
        lambda path, flag=None: None,
    )

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 0),
    )

    assert result is None


def test_calculate_roi_contamination_returns_none_when_mask_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_imread(_path: str, flag: int | None = None) -> Any:
        if flag == cv2.IMREAD_GRAYSCALE:
            return None
        return image

    monkeypatch.setattr("helpers.graph.contamination.cv2.imread", fake_imread)

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 0),
    )

    assert result is None


def test_calculate_roi_contamination_returns_nan_for_empty_initial_roi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=np.uint8)

    def fake_imread(_path: str, flag: int | None = None) -> Any:
        return mask if flag == cv2.IMREAD_GRAYSCALE else image

    monkeypatch.setattr("helpers.graph.contamination.cv2.imread", fake_imread)

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 0),
    )

    assert result is not None
    assert np.isnan(result)


def test_calculate_roi_contamination_returns_nan_when_erosion_removes_roi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255

    def fake_imread(_path: str, flag: int | None = None) -> Any:
        return mask if flag == cv2.IMREAD_GRAYSCALE else image

    monkeypatch.setattr("helpers.graph.contamination.cv2.imread", fake_imread)

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 1),
    )

    assert result is not None
    assert np.isnan(result)


def test_calculate_roi_contamination_returns_expected_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.array(
        [
            [[255, 255, 255], [10, 10, 10]],
            [[250, 250, 250], [20, 20, 20]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[255, 255], [0, 255]], dtype=np.uint8)
    segment_map = np.array([[0, 1], [0, 1]], dtype=np.int32)

    def fake_felzenszwalb(
        image_input: npt.NDArray[np.uint8], *, scale: float, sigma: float, min_size: int
    ) -> npt.NDArray[np.int32]:
        assert np.array_equal(image_input, image)
        assert scale == 100.0
        assert sigma == 0.5
        assert min_size == 10
        return segment_map

    monkeypatch.setattr(
        "helpers.graph.contamination.cv2.imread",
        lambda path, flag=None: mask if flag == cv2.IMREAD_GRAYSCALE else image,
    )
    monkeypatch.setattr("helpers.graph.contamination.felzenszwalb", fake_felzenszwalb)

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 0),
        logger=logging.getLogger("test"),
    )

    assert result == pytest.approx(1 / 3)


def test_calculate_roi_contamination_returns_none_on_segmentation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.uint8) * 255

    def fake_imread(_path: str, flag: int | None = None) -> Any:
        return mask if flag == cv2.IMREAD_GRAYSCALE else image

    def raising_felzenszwalb(_image: npt.NDArray[np.uint8], **kwargs: Any) -> npt.NDArray[np.int32]:
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr("helpers.graph.contamination.cv2.imread", fake_imread)
    monkeypatch.setattr("helpers.graph.contamination.felzenszwalb", raising_felzenszwalb)

    result = calculate_roi_contamination(
        tmp_path / "image.png",
        tmp_path / "mask.png",
        GraphContaminationParameters(200, 100.0, 10, 0),
    )

    assert result is None
