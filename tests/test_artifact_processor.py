from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import helpers.artifact.processor as processor_module
from helpers.artifact.config import ArtifactDetectionConfig
from helpers.artifact.processor import (
    ArtifactProcessor,
    HorizontalTileAppend,
    TileGrid,
    VerticalTileAppend,
    _append_horizontal_tile,
    _append_vertical_tile,
    _combine_horizontal_tiles,
    _combine_vertical_tiles,
    _crop_tile,
)


def _build_config(tmp_path: Path) -> ArtifactDetectionConfig:
    return ArtifactDetectionConfig(
        images_zip=tmp_path / "slides.zip",
        geojson_output=tmp_path / "geojson",
        database_folder=tmp_path / "db",
        database_path=tmp_path / "db" / "artifact.db",
        temp_root=tmp_path / "temp",
        log_folder=tmp_path / "logs",
        log_file_name="artifact.log",
        device="cpu",
        tissue_detector_model_dir=tmp_path / "models" / "td",
        tissue_detector_model_name="td_model.pth",
        qc_model_dir=tmp_path / "models" / "qc",
        mpp_model=1.5,
    )


def test_process_slide_creates_output_dir_and_calls_internal_steps(tmp_path: Path) -> None:
    processor = ArtifactProcessor(_build_config(tmp_path), model_loader=Mock())
    run_tissue = Mock(return_value={"tissue_mask_path": tmp_path / "mask.png"})
    run_qc = Mock()
    processor._run_tissue_detection = run_tissue  # type: ignore[method-assign]
    processor._run_qc_processing = run_qc  # type: ignore[method-assign]

    slide_path = tmp_path / "slide.svs"
    slide_path.write_text("slide", encoding="utf-8")
    geojson_path = tmp_path / "geojson" / "slide.geojson"

    processor.process_slide(slide_path, "archive/slide.svs", geojson_path)

    assert (tmp_path / "artifact_output").is_dir()
    run_tissue.assert_called_once_with(slide_path, tmp_path / "artifact_output")
    run_qc.assert_called_once_with(
        slide_path,
        geojson_path,
        {"tissue_mask_path": tmp_path / "mask.png"},
    )


def test_crop_tile_handles_regular_right_bottom_and_corner_tiles() -> None:
    image = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10))

    grid = TileGrid(
        tiles_x=2,
        tiles_y=2,
        patch_size=4,
        width=10,
        height=10,
        overhang_x=2,
        overhang_y=2,
    )

    regular = _crop_tile(image, 0, 0, grid)
    right_edge = _crop_tile(image, 2, 0, grid)
    bottom_edge = _crop_tile(image, 0, 2, grid)
    corner = _crop_tile(image, 2, 2, grid)

    assert regular.size == (4, 4)
    assert regular.getbbox() == (0, 0, 4, 4)
    assert right_edge.getbbox() == (0, 0, 4, 4)
    assert bottom_edge.getbbox() == (0, 0, 4, 4)
    assert corner.getbbox() == (0, 0, 4, 4)
    assert np.array(right_edge)[0, 0] == np.array(image)[0, 6]
    assert np.array(bottom_edge)[0, 0] == np.array(image)[6, 0]
    assert np.array(corner)[0, 0] == np.array(image)[6, 6]


def test_append_horizontal_tile_handles_init_full_append_and_overhang() -> None:
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    class_mask = np.dstack([mask, mask, mask])

    first_image, first_class_map = _append_horizontal_tile(
        HorizontalTileAppend(None, None, mask, class_mask, 0, 2, 2, 1)
    )
    assert np.array_equal(first_image, mask)
    assert np.array_equal(first_class_map, class_mask)

    with pytest.raises(RuntimeError, match="Horizontal stitching state"):
        _append_horizontal_tile(
            HorizontalTileAppend(None, class_mask, mask, class_mask, 1, 2, 2, 1)
        )

    appended, appended_class_map = _append_horizontal_tile(
        HorizontalTileAppend(first_image, first_class_map, mask, class_mask, 1, 2, 2, 1)
    )
    assert appended.shape == (2, 4)
    assert appended_class_map.shape == (2, 4, 3)

    clipped, clipped_class_map = _append_horizontal_tile(
        HorizontalTileAppend(first_image, first_class_map, mask, class_mask, 2, 2, 2, 1)
    )
    assert clipped.shape == (2, 3)
    assert clipped_class_map.shape == (2, 3, 3)


def test_append_vertical_tile_handles_init_full_append_and_overhang() -> None:
    row = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    class_row = np.dstack([row, row, row])

    first_image, first_class_map = _append_vertical_tile(
        VerticalTileAppend(None, None, row, class_row, 0, 2, 2, 1)
    )
    assert np.array_equal(first_image, row)
    assert np.array_equal(first_class_map, class_row)

    with pytest.raises(RuntimeError, match="Vertical stitching state"):
        _append_vertical_tile(VerticalTileAppend(None, class_row, row, class_row, 1, 2, 2, 1))

    appended, appended_class_map = _append_vertical_tile(
        VerticalTileAppend(first_image, first_class_map, row, class_row, 1, 2, 2, 1)
    )
    assert appended.shape == (4, 2)
    assert appended_class_map.shape == (4, 2, 3)

    clipped, clipped_class_map = _append_vertical_tile(
        VerticalTileAppend(first_image, first_class_map, row, class_row, 2, 2, 2, 1)
    )
    assert clipped.shape == (3, 2)
    assert clipped_class_map.shape == (3, 2, 3)


def test_combine_tile_helpers_join_rows_and_columns_once() -> None:
    row_masks = [
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        np.array([[5, 6], [7, 8]], dtype=np.uint8),
        np.array([[9, 10], [11, 12]], dtype=np.uint8),
    ]
    row_class_masks = [np.dstack([mask, mask, mask]) for mask in row_masks]

    combined_row, combined_row_class = _combine_horizontal_tiles(
        row_masks,
        row_class_masks,
        patch_size=2,
        overhang_x=1,
    )

    assert combined_row.shape == (2, 5)
    assert combined_row_class.shape == (2, 5, 3)

    combined_image, combined_class = _combine_vertical_tiles(
        [combined_row, combined_row, combined_row],
        [combined_row_class, combined_row_class, combined_row_class],
        patch_size=2,
        overhang_y=1,
    )

    assert combined_image.shape == (5, 5)
    assert combined_class.shape == (5, 5, 3)


def test_run_qc_processing_writes_outputs_and_geojson(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_config(tmp_path)
    model_loader = Mock()
    model_loader.load.return_value = types.SimpleNamespace(qc_model="qc-model")
    processor = ArtifactProcessor(config, model_loader=model_loader)
    slide = types.SimpleNamespace()
    tissue_mask_path = tmp_path / "artifact_output" / "tis_det_mask" / "slide_MASK.png"
    tissue_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tissue_mask_path)
    geojson_path = tmp_path / "geojson" / "slide.geojson"
    slide_process_calls: list[object] = []
    imwrite_calls: list[Path] = []
    geojson_calls: list[tuple[str, str, float]] = []

    monkeypatch.setattr(
        processor_module,
        "load_openslide_module",
        lambda: types.SimpleNamespace(open_slide=lambda path: slide),
    )
    monkeypatch.setattr("helpers.wsi.slide_info.slide_info", lambda *args: (4, 2, 3, 0.5, 8, 6, 20))
    monkeypatch.setattr("helpers.wsi.colors.colors_QC7", [[1, 2, 3]])

    def fake_slide_process_single(*args: object) -> tuple[Image.Image, npt.NDArray[np.uint8]]:
        slide_process_calls.append(args)
        return Image.new("RGB", (2, 3), color=(1, 2, 3)), np.ones((3, 2), dtype=np.uint8)

    def fake_imwrite(path: str, data: object) -> bool:
        del data
        imwrite_calls.append(Path(path))
        return True

    monkeypatch.setattr("helpers.wsi.process.slide_process_single", fake_slide_process_single)
    monkeypatch.setattr(
        "helpers.wsi.maps.make_overlay",
        lambda *args: np.full((5, 5, 3), 128, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "helpers.wsi.process.mask_to_geojson",
        lambda mask_path, output_path, factor: geojson_calls.append(
            (mask_path, output_path, factor)
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "cv2",
        types.SimpleNamespace(imwrite=fake_imwrite),
    )

    processor._run_qc_processing(
        tmp_path / "slide.svs",
        geojson_path,
        {"tissue_mask_path": tissue_mask_path},
    )

    assert model_loader.load.call_count == 1
    assert slide_process_calls
    assert imwrite_calls == [tmp_path / "artifact_output" / "mask_qc" / "slide_mask.png"]
    assert geojson_calls == [
        (
            str(tmp_path / "artifact_output" / "mask_qc" / "slide_mask.png"),
            str(geojson_path),
            config.mpp_model / 0.5,
        )
    ]
    assert (tmp_path / "artifact_output" / "maps_qc" / "slide_map_QC.png").is_file()
    assert (tmp_path / "artifact_output" / "overlays_qc" / "slide_overlay_QC.jpg").is_file()


def test_run_tissue_detection_raises_when_stitching_returns_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_config(tmp_path)
    config = ArtifactDetectionConfig(**{**config.__dict__, "model_patch_size": 2})
    model_loader = Mock()
    model_loader.load.return_value = types.SimpleNamespace(
        preprocessing_fn=lambda array: array,
        tissue_model=types.SimpleNamespace(
            predict=lambda tensor: types.SimpleNamespace(
                squeeze=lambda: types.SimpleNamespace(
                    cpu=lambda: types.SimpleNamespace(
                        numpy=lambda: np.zeros((2, 2, 2), dtype=np.float32)
                    )
                )
            )
        ),
    )
    processor = ArtifactProcessor(config, model_loader=model_loader)

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    class FakeTensor:
        def __init__(self, array: npt.NDArray[np.float32]) -> None:
            self.array = array

        def to(self, device: str) -> FakeTensor:
            del device
            return self

        def unsqueeze(self, axis: int) -> FakeTensor:
            del axis
            return self

    fake_slide = types.SimpleNamespace(
        level_dimensions=[(4, 4)],
        properties={"openslide.mpp-x": "2.0"},
        get_thumbnail=lambda size: Image.new("RGB", size, color=(10, 20, 30)),
    )
    monkeypatch.setattr(
        processor_module,
        "load_openslide_module",
        lambda: types.SimpleNamespace(OpenSlide=lambda path: fake_slide),
    )
    monkeypatch.setattr(
        "helpers.wsi.tis_detect_helper_fx.get_preprocessing",
        lambda image, fn: np.zeros((2, 2, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "helpers.wsi.tis_detect_helper_fx.make_class_map",
        lambda mask, colors: np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "helpers.artifact.processor._combine_vertical_tiles", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(
        "sys.modules",
        {
            **__import__("sys").modules,
            "cv2": types.SimpleNamespace(
                IMWRITE_JPEG_QUALITY=95,
                imencode=lambda ext, image, params: (True, image),
                imdecode=lambda image, flag: image,
                addWeighted=lambda a, b, c, d, e: np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            "torch": types.SimpleNamespace(
                inference_mode=lambda: FakeInferenceMode(),
                from_numpy=lambda array: FakeTensor(array),
            ),
        },
    )

    with pytest.raises(RuntimeError, match="No tissue detection output"):
        processor._run_tissue_detection(tmp_path / "slide.svs", tmp_path / "artifact_output")
