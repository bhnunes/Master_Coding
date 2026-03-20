from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers.optimization_sampling.overlay import (
    _process_overlay_task,
    _to_grayscale_mask,
    generate_overlay_images,
    overlay_mask_edges,
)
from helpers.optimization_sampling.sampling import OverlayTask


def test_to_grayscale_mask_uses_alpha_channel_for_rgba() -> None:
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    mask[:, :, 3] = 7

    result = _to_grayscale_mask(mask)

    assert np.array_equal(result, np.full((2, 2), 7, dtype=np.uint8))


def test_to_grayscale_mask_converts_rgb_and_preserves_grayscale() -> None:
    rgb_mask = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_mask[0, 0] = [0, 0, 255]
    gray_mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    rgb_result = _to_grayscale_mask(rgb_mask)
    gray_result = _to_grayscale_mask(gray_mask)

    assert rgb_result.shape == (2, 2)
    assert np.array_equal(gray_result, gray_mask)


def test_overlay_mask_edges_returns_false_when_image_missing(tmp_path: Path) -> None:
    result = overlay_mask_edges(
        tmp_path / "missing.png", tmp_path / "mask.png", tmp_path / "out.png"
    )

    assert result is False


def test_overlay_mask_edges_returns_false_on_size_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output" / "overlay.png"
    cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((2, 2), dtype=np.uint8))

    result = overlay_mask_edges(image_path, mask_path, output_path)

    assert result is False
    assert not output_path.exists()


def test_overlay_mask_edges_writes_overlay_file(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output" / "overlay.png"
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    result = overlay_mask_edges(image_path, mask_path, output_path, alpha=0.5)

    assert result is True
    assert output_path.exists()
    overlay = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert int(overlay[:, :, 2].sum()) > 0


def test_overlay_mask_edges_returns_false_on_opencv_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    monkeypatch.setattr(
        "helpers.optimization_sampling.overlay.cv2.findContours",
        lambda *args, **kwargs: (_ for _ in ()).throw(cv2.error("findContours", "test", "boom")),
    )

    result = overlay_mask_edges(image_path, mask_path, tmp_path / "out.png")

    assert result is False


def test_generate_overlay_images_returns_empty_list_for_no_tasks() -> None:
    assert generate_overlay_images([], num_processes=2) == []


def test_process_overlay_task_forwards_parameters(tmp_path: Path) -> None:
    task = OverlayTask(
        image_path=tmp_path / "image.png",
        mask_path=tmp_path / "mask.png",
        output_path=tmp_path / "out.png",
        color=(1, 2, 3),
        thickness=4,
        alpha=0.25,
    )
    cv2.imwrite(str(task.image_path), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(task.mask_path), np.ones((4, 4), dtype=np.uint8) * 255)

    assert _process_overlay_task(task) is True
