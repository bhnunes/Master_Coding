# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from helpers.wsi.colors import colors_QC7
from helpers.wsi.maps import make_overlay
from helpers.wsi.process import (
    get_preprocessing as process_get_preprocessing,
)
from helpers.wsi.process import (
    make_1class_map_thr,
    mask_to_geojson,
    slide_process_single,
)
from helpers.wsi.process import (
    to_tensor_x as process_to_tensor_x,
)
from helpers.wsi.slide_info import slide_info
from helpers.wsi.tis_detect_helper_fx import get_preprocessing, make_class_map, to_tensor_x


class FakeSlide:
    def __init__(self) -> None:
        self.level_dimensions = [(40, 20)]
        self.properties = {
            "openslide.objective-power": "20",
            "openslide.mpp-x": "0.5",
            "openslide.vendor": "aperio",
        }
        self.level_count = 3
        self.level_downsamples = [1.0, 4.0, 16.0]
        self.read_calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []

    def get_thumbnail(self, size: tuple[float, float]) -> Image.Image:
        width, height = int(size[0]), int(size[1])
        return Image.new("RGB", (width, height), color=(10, 20, 30))

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        self.read_calls.append((location, level, size))
        return Image.new("RGB", size, color=(40, 50, 60))


def test_colors_qc7_exposes_expected_palette_contract() -> None:
    assert len(colors_QC7) == 7
    assert colors_QC7[0] == [128, 128, 128]
    assert colors_QC7[-1] == [255, 255, 255]


def test_make_overlay_resizes_heatmap_to_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    slide = FakeSlide()
    heatmap = Image.new("RGB", (5, 5), color=(200, 0, 0))
    expected = np.ones((10, 20, 3), dtype=np.uint8)

    def fake_add_weighted(
        slide_pixels: npt.NDArray[np.uint8],
        slide_weight: float,
        heatmap_pixels: npt.NDArray[np.uint8],
        heatmap_weight: float,
        gamma: int,
    ) -> npt.NDArray[np.uint8]:
        assert slide_pixels.shape == (10, 20, 3)
        assert heatmap_pixels.shape == (10, 20, 3)
        assert slide_weight == 0.7
        assert heatmap_weight == 0.3
        assert gamma == 0
        return expected

    monkeypatch.setattr("helpers.wsi.maps.cv2.addWeighted", fake_add_weighted)

    overlay = make_overlay(slide, heatmap, 512, 1, 1, 2)

    assert np.array_equal(overlay, expected)


def test_slide_info_returns_expected_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    slide = FakeSlide()
    printed: list[object] = []
    monkeypatch.setattr("builtins.print", lambda *args: printed.append(args))

    result = slide_info(slide, m_p_s=256, mpp_model=1.0)

    assert result == (512, 0, 0, 0.5, 40, 20, "20")
    assert printed


def test_slide_info_falls_back_when_objective_power_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slide = FakeSlide()
    del slide.properties["openslide.objective-power"]
    monkeypatch.setattr("builtins.print", lambda *args: None)

    result = slide_info(slide, m_p_s=256, mpp_model=1.0)

    assert result[-1] == 99


def test_tissue_detection_preprocessing_helpers_convert_to_channels_first() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    tensor = to_tensor_x(image)
    preprocessed = get_preprocessing(Image.fromarray(image), lambda array: array + 1)

    assert tensor.shape == (3, 2, 3)
    assert tensor.dtype == np.float32
    assert preprocessed.shape == (3, 2, 3)
    assert np.all(preprocessed[:, 0, 0] == np.array([1, 2, 3], dtype=np.float32))


def test_make_class_map_assigns_rgb_values_per_class() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    colors = [[1, 2, 3], [4, 5, 6]]

    class_map = make_class_map(mask, colors)

    assert class_map.tolist() == [[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3]]]


def test_process_preprocessing_resizes_images_and_reorders_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    printed: list[object] = []
    monkeypatch.setattr("builtins.print", lambda *args: printed.append(args))

    tensor = process_get_preprocessing(image, lambda array: array + 2, (2, 2))

    assert tensor.shape == (3, 2, 2)
    assert printed == [("resized",)]
    assert process_to_tensor_x(np.zeros((1, 2, 3), dtype=np.uint8)).shape == (3, 1, 2)


def test_make_1class_map_thr_assigns_palette_to_positive_classes() -> None:
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    class_map = make_1class_map_thr(mask, [[10, 20, 30], [40, 50, 60]])

    assert class_map.tolist() == [[[0, 0, 0], [10, 20, 30]], [[40, 50, 60], [0, 0, 0]]]


def test_slide_process_single_processes_tissue_and_pads_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slide = FakeSlide()

    class FakePrediction:
        def __init__(self, array: npt.NDArray[np.float32]) -> None:
            self.array = array

        def squeeze(self) -> FakePrediction:
            return self

        def cpu(self) -> FakePrediction:
            return self

        def numpy(self) -> npt.NDArray[np.float32]:
            return self.array

    class FakeModel:
        def predict(self, tensor: object) -> FakePrediction:
            del tensor
            predictions = np.zeros((2, 512, 512), dtype=np.float32)
            predictions[1, :, :] = 1.0
            return FakePrediction(predictions)

    monkeypatch.setattr(
        "helpers.wsi.process.smp.encoders.get_preprocessing_fn",
        lambda encoder, weights: lambda array: array,
    )
    monkeypatch.setattr("helpers.wsi.process.tqdm", lambda iterable, total: iterable)

    heatmap, mask = slide_process_single(
        model=FakeModel(),
        tis_det_map_mpp=np.zeros((512, 512), dtype=np.uint8),
        slide=slide,
        patch_n_w_l0=1,
        patch_n_h_l0=1,
        p_s=10,
        m_p_s=512,
        colors=[[5, 10, 15]],
        ENCODER_MODEL_1="encoder",
        ENCODER_WEIGHTS="weights",
        DEVICE="cpu",
        BACK_CLASS=0,
        MPP_MODEL_1=1.0,
        mpp=1.0,
        w_l0=11,
        h_l0=12,
    )

    assert isinstance(heatmap, Image.Image)
    assert heatmap.size == (50, 50)
    assert mask.shape == (514, 513)
    assert slide.read_calls == [((0, 0), 0, (10, 10))]
    assert mask[0, 0] == 1
    assert mask[-1, -1] == 0


def test_slide_process_single_uses_background_when_tissue_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slide = FakeSlide()

    class FakeModel:
        def predict(self, tensor: object) -> None:
            raise AssertionError("predict should not be called")

    monkeypatch.setattr(
        "helpers.wsi.process.smp.encoders.get_preprocessing_fn",
        lambda encoder, weights: lambda array: array,
    )
    monkeypatch.setattr("helpers.wsi.process.tqdm", lambda iterable, total: iterable)

    _, mask = slide_process_single(
        model=FakeModel(),
        tis_det_map_mpp=np.ones((512, 512), dtype=np.uint8),
        slide=slide,
        patch_n_w_l0=1,
        patch_n_h_l0=1,
        p_s=10,
        m_p_s=512,
        colors=[[5, 10, 15], [20, 25, 30]],
        ENCODER_MODEL_1="encoder",
        ENCODER_WEIGHTS="weights",
        DEVICE="cpu",
        BACK_CLASS=7,
        MPP_MODEL_1=1.0,
        mpp=1.0,
        w_l0=10,
        h_l0=10,
    )

    assert not slide.read_calls
    assert mask.shape == (512, 512)
    assert np.all(mask == 7)


def test_mask_to_geojson_writes_scaled_polygons_and_skips_short_contours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(b"mask")
    output_path = tmp_path / "mask.geojson"
    mask = np.array([[0, 2], [4, 6]], dtype=np.uint8)

    square = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]], dtype=np.int32)
    triangle = np.array([[[1, 1]], [[2, 1]], [[2, 2]]], dtype=np.int32)
    contours = {
        0: ([], None),
        2: ([square], None),
        3: ([], None),
        4: ([triangle], None),
        5: ([], None),
        6: ([square], None),
    }

    monkeypatch.setattr("helpers.wsi.process.cv2.imread", lambda path, flag: mask)
    monkeypatch.setattr(
        "helpers.wsi.process.cv2.findContours",
        lambda class_mask, mode, method: contours[
            int(class_mask.max() / 255 * np.max(mask[class_mask > 0]))
        ],
    )
    monkeypatch.setattr("helpers.wsi.process.cv2.contourArea", lambda contour: 4.0)

    def fake_find_contours(
        class_mask: npt.NDArray[np.uint8], mode: int, method: int
    ) -> tuple[list[npt.NDArray[np.int32]], None]:
        del mode, method
        class_value = int(mask[class_mask > 0][0]) if np.any(class_mask > 0) else 0
        return contours[class_value]

    monkeypatch.setattr("helpers.wsi.process.cv2.findContours", fake_find_contours)

    mask_to_geojson(str(mask_path), str(output_path), scale_factor=2.0)

    geojson = json.loads(output_path.read_text(encoding="utf-8"))

    assert geojson["type"] == "FeatureCollection"
    assert geojson["metadata"]["scale_factor"] == 2.0
    assert len(geojson["features"]) == 2
    assert geojson["features"][0]["properties"]["classification"] == "Fold"
    assert geojson["features"][0]["properties"]["area"] == 16.0
    assert geojson["features"][0]["geometry"]["coordinates"][0][0] == [0, 0]
    assert geojson["features"][0]["geometry"]["coordinates"][0][-1] == [0, 0]
    assert geojson["features"][1]["properties"]["classification"] == "OOF"
