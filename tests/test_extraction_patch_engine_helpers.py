# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from helpers.extraction import patch_engine
from helpers.extraction.data_handlers import BaseHandler
from helpers.extraction.profiling import create_phase_stats

PATCH_SIDE = 4
MIN_NONZERO_PIXELS = 4
OVERLAP_RATIO = 0.25
COORDINATE_COLUMNS = 2


def test_check_tissue_percentage_robust_handles_empty_patch() -> None:
    assert patch_engine.check_tissue_percentage_robust(None, 0.1) is False
    assert patch_engine.check_tissue_percentage_robust(np.array([]), 0.1) is False


def test_build_tissue_binary_mask_returns_nonempty_mask_for_colored_patch() -> None:
    patch = np.zeros((PATCH_SIDE, PATCH_SIDE, 3), dtype=np.uint8)
    patch[:, :, 0] = 255

    tissue_mask = patch_engine._build_tissue_binary_mask(patch)

    assert tissue_mask.shape == (PATCH_SIDE, PATCH_SIDE)
    assert tissue_mask.dtype == np.uint8


def test_polygons_to_mask_fills_intersection_inside_patch() -> None:
    mask = patch_engine.polygons_to_mask(
        mask_shape=(PATCH_SIDE, PATCH_SIDE),
        polygons_level0=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
        scale_factor=1.0,
        patch_coords=(0, 0),
    )

    assert mask.shape == (PATCH_SIDE, PATCH_SIDE)
    assert np.count_nonzero(mask) >= MIN_NONZERO_PIXELS


def test_build_scaled_polygon_index_skips_invalid_entries() -> None:
    polygons, tree = patch_engine.build_scaled_polygon_index(
        [
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            [(1, 1)],
            [(0, 0), (4, 4), (4, 0), (0, 4)],
        ],
        scale_factor=2.0,
    )

    assert len(polygons) >= 1
    assert tree is not None


def test_build_artifact_geometry_index_builds_only_known_non_empty_classes() -> None:
    index = patch_engine.build_artifact_geometry_index(
        {
            "Fold": [[(0, 0), (10, 0), (10, 10), (0, 10)]],
            "Unknown": [[(0, 0), (1, 0), (1, 1)]],
            "PenMarking": [[(1, 1)]],
        },
        scale_factor=1.0,
    )

    assert set(index) == {"cov_fold"}
    assert index["cov_fold"].bounds == (0.0, 0.0, 10.0, 10.0)


def test_build_artifact_geometry_index_is_picklable_for_worker_handoff() -> None:
    index = patch_engine.build_artifact_geometry_index(
        {"Fold": [[(0, 0), (10, 0), (10, 10), (0, 10)]]},
        scale_factor=1.0,
    )

    pickle.dumps(index)


def test_prepare_artifact_geometry_index_adds_prepared_geometries() -> None:
    raw_index = patch_engine.build_artifact_geometry_index(
        {"Fold": [[(0, 0), (10, 0), (10, 10), (0, 10)]]},
        scale_factor=1.0,
    )

    prepared_index = patch_engine.prepare_artifact_geometry_index(raw_index)

    assert set(prepared_index) == {"cov_fold"}
    artifact_geometry, prepared_geometry = prepared_index["cov_fold"]
    assert artifact_geometry.bounds == (0.0, 0.0, 10.0, 10.0)
    assert prepared_geometry.intersects(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))


def test_compute_artifact_coverages_from_index_returns_overlap_fraction() -> None:
    raw_index = patch_engine.build_artifact_geometry_index(
        {"Fold": [[(0, 0), (5, 0), (5, 5), (0, 5)]]}, scale_factor=1.0
    )
    index = patch_engine.prepare_artifact_geometry_index(raw_index)

    coverages = patch_engine.compute_artifact_coverages_from_index(
        index,
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        100.0,
    )

    assert coverages["cov_fold"] == OVERLAP_RATIO
    assert coverages["cov_penmarking"] == 0.0


def test_polygons_to_mask_with_index_returns_empty_mask_without_tree() -> None:
    mask = patch_engine.polygons_to_mask_with_index((PATCH_SIDE, PATCH_SIDE), ([], None), (0, 0))

    assert np.count_nonzero(mask) == 0


def test_polygons_to_mask_with_index_fills_matching_polygon() -> None:
    polygon_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (3, 0), (3, 3), (0, 3)]],
        scale_factor=1.0,
    )

    mask = patch_engine.polygons_to_mask_with_index((PATCH_SIDE, PATCH_SIDE), polygon_index, (0, 0))

    assert np.count_nonzero(mask) >= MIN_NONZERO_PIXELS


def test_build_precomputed_region_masks_creates_scan_region_masks() -> None:
    preparation = patch_engine._ExtractionPreparation(
        slide_basename="slide.svs",
        annotations_cancer_level0=[[(0, 0), (4, 0), (4, 4), (0, 4)]],
        annotations_not_cancer_level0=[[(4, 0), (8, 0), (8, 4), (4, 4)]],
        artifact_polygons_by_class_level0={"Fold": [[(0, 0), (4, 0), (4, 4), (0, 4)]]},
    )
    region = patch_engine._ScaledAnnotationRegion(
        scale_factor=1.0,
        target_width=8,
        target_height=4,
        x_start=0,
        y_start=0,
        x_end=8,
        y_end=4,
    )

    label_masks, artifact_masks = patch_engine._build_precomputed_region_masks(
        preparation,
        region,
        use_artifact_filter=True,
    )

    assert label_masks is not None
    assert artifact_masks is not None
    assert label_masks["cancer"].shape == (4, 8)
    assert label_masks["not_cancer"].shape == (4, 8)
    assert np.count_nonzero(label_masks["cancer"][:, :4]) > 0
    assert np.count_nonzero(label_masks["not_cancer"][:, 4:]) > 0
    assert np.count_nonzero(artifact_masks["cov_fold"]) > 0


def test_clip_geometry_to_patch_coords_handles_multipolygon_result() -> None:
    geometry = Polygon([(0, 0), (6, 0), (6, 2), (4, 2), (4, 4), (6, 4), (6, 6), (0, 6)])

    clipped = patch_engine.clip_geometry_to_patch_coords(
        geometry,
        patch_x=1,
        patch_y=1,
        mask_width=4,
        mask_height=4,
    )

    assert len(clipped) >= 1
    assert all(coords.dtype == np.int32 for coords in clipped)
    assert all(coords.shape[1] == COORDINATE_COLUMNS for coords in clipped)


def test_run_extraction_returns_zero_when_handler_finds_no_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        level_downsamples = [1.0]
        level_dimensions = [(100, 100)]

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeHandler(BaseHandler):
        def load_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            del slide, kwargs
            return {"cancer_polygons": [], "not_cancer_polygons": []}

        def _load_raw_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            raise NotImplementedError

    slide = FakeSlide()
    monkeypatch.setattr(
        patch_engine,
        "load_openslide_module",
        lambda: type("OS", (), {"OpenSlide": lambda self, path: slide})(),
    )

    result = patch_engine.run_extraction(
        FakeHandler(),
        "/tmp/slide.svs",
        target_level=0,
        window_size=patch_engine.WINDOW_SIZE,
        stride=patch_engine.WINDOW_SIZE,
        num_workers=1,
        tissue_percentage_req=0.1,
        match_percentage_req=0.1,
        path_cancer_folder="/tmp/cancer",
        path_not_cancer_folder="/tmp/not_cancer",
        path_cancer_mask_folder="/tmp/cancer_mask",
        path_not_cancer_mask_folder="/tmp/not_cancer_mask",
        patient="p1",
    )

    assert result == (0, 0, [])
    assert slide.closed is True


def test_process_window_with_slide_skips_tissue_and_records_profile_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (PATCH_SIDE, PATCH_SIDE), color=(10, 20, 30))

    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: False)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)
    not_cancer_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (4, 0), (4, 4), (0, 4)]],
        scale_factor=1.0,
    )
    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": PATCH_SIDE,
        "use_artifact_filter": False,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.1,
        "cancer_polygon_index": ([], None),
        "not_cancer_polygon_index": not_cancer_index,
        "path_cancer_folder": "/tmp/CANCER",
        "path_cancer_mask_folder": "/tmp/CANCER_MASK",
        "path_not_cancer_folder": "/tmp/NOT_CANCER",
        "path_not_cancer_mask_folder": "/tmp/NOT_CANCER_MASK",
        "patient": "p1",
        "slide_id": "slide",
    }

    result = cast(
        tuple[str, object | None, dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 1, 2),
    )

    assert result[0] == "SKIPPED_TISSUE"
    assert result[1] is None
    assert "tissue_check" in result[2]
    assert result[2]["read_region"].calls == 1


def test_process_window_with_slide_uses_precomputed_label_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (PATCH_SIDE, PATCH_SIDE), color=(10, 20, 30))

    label_region_masks = {
        "cancer": np.ones((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        "not_cancer": np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
    }
    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": PATCH_SIDE,
        "use_artifact_filter": False,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.1,
        "region_x_start": 0,
        "region_y_start": 0,
        "label_region_masks": label_region_masks,
        "cancer_polygon_index": ([], None),
        "not_cancer_polygon_index": ([], None),
        "path_cancer_folder": "/tmp/CANCER",
        "path_cancer_mask_folder": "/tmp/CANCER_MASK",
        "path_not_cancer_folder": "/tmp/NOT_CANCER",
        "path_not_cancer_mask_folder": "/tmp/NOT_CANCER_MASK",
        "patient": "patient-1",
        "slide_id": "slide-a",
        "filename_prefix": patch_engine.build_patch_filename_prefix(
            patient_id="patient-1",
            slide_id="slide-a",
        ),
    }

    monkeypatch.setattr(
        patch_engine,
        "polygons_to_mask_with_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("per-window polygon rasterization should be bypassed")
        ),
    )
    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: True)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)

    result = cast(
        tuple[str, dict[str, Any], dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SAVED_CANCER"
    assert result[1]["label"] == 1
    assert np.count_nonzero(result[1]["_mask_array"]) == PATCH_SIDE * PATCH_SIDE


def test_process_window_with_slide_skips_tissue_before_reading_slide_when_tissue_mask_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def read_region(
            self, _coords: tuple[int, int], _level: int, _size: tuple[int, int]
        ) -> Image.Image:
            raise AssertionError("read_region should not be called when tissue mask rejects window")

    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": PATCH_SIDE,
        "use_artifact_filter": False,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.1,
        "region_x_start": 0,
        "region_y_start": 0,
        "label_region_masks": {
            "cancer": np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
            "not_cancer": np.ones((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        },
        "tissue_region_mask": np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        "cancer_polygon_index": ([], None),
        "not_cancer_polygon_index": ([], None),
    }
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)

    result = cast(
        tuple[str, object | None, dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SKIPPED_TISSUE"
    assert result[1] is None
    assert result[2]["read_region"].calls == 0


def test_process_window_with_slide_builds_not_cancer_patch_with_artifact_coverages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (4, 4), color=(10, 20, 30))

    cancer_index: tuple[list[Polygon], None] = ([], None)
    not_cancer_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (4, 0), (4, 4), (0, 4)]],
        scale_factor=1.0,
    )
    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": 4,
        "use_artifact_filter": True,
        "artifact_geometry_index": {"dummy": object()},
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.01,
        "cancer_polygon_index": cancer_index,
        "not_cancer_polygon_index": not_cancer_index,
        "path_cancer_folder": str(tmp_path / "CANCER"),
        "path_cancer_mask_folder": str(tmp_path / "CANCER_MASK"),
        "path_not_cancer_folder": str(tmp_path / "NOT_CANCER"),
        "path_not_cancer_mask_folder": str(tmp_path / "NOT_CANCER_MASK"),
        "patient": "patient-2",
        "slide_id": "slide-b",
        "filename_prefix": patch_engine.build_patch_filename_prefix(
            patient_id="patient-2",
            slide_id="slide-b",
        ),
    }
    for folder in ["CANCER", "CANCER_MASK", "NOT_CANCER", "NOT_CANCER_MASK"]:
        (tmp_path / folder).mkdir()

    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: True)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)
    monkeypatch.setattr(
        patch_engine,
        "compute_artifact_coverages_from_index",
        lambda artifact_geometry_index, patch_polygon, patch_area: {
            "cov_fold": 0.25,
            "cov_penmarking": 0.0,
            "cov_oof": 0.0,
            "cov_darkspot_foreign": 0.0,
            "cov_edge_airbubble": 0.0,
        },
    )

    result = cast(
        tuple[str, dict[str, Any], dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SAVED_NOT_CANCER"
    assert result[1]["label"] == 0
    assert result[1]["slide_id"] == "slide-b"
    assert result[1]["cov_fold"] == OVERLAP_RATIO
    assert result[1]["_image_array"].shape == (PATCH_SIDE, PATCH_SIDE, 3)
    assert result[1]["_mask_array"].shape == (PATCH_SIDE, PATCH_SIDE)


def test_process_window_with_slide_uses_precomputed_artifact_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (PATCH_SIDE, PATCH_SIDE), color=(10, 20, 30))

    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": PATCH_SIDE,
        "use_artifact_filter": True,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.01,
        "region_x_start": 0,
        "region_y_start": 0,
        "label_region_masks": {
            "cancer": np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
            "not_cancer": np.ones((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        },
        "artifact_region_masks": {
            "cov_fold": np.ones((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        },
        "cancer_polygon_index": ([], None),
        "not_cancer_polygon_index": ([], None),
        "artifact_geometry_index": {"dummy": object()},
        "path_cancer_folder": str(tmp_path / "CANCER"),
        "path_cancer_mask_folder": str(tmp_path / "CANCER_MASK"),
        "path_not_cancer_folder": str(tmp_path / "NOT_CANCER"),
        "path_not_cancer_mask_folder": str(tmp_path / "NOT_CANCER_MASK"),
        "patient": "patient-2",
        "slide_id": "slide-b",
        "filename_prefix": patch_engine.build_patch_filename_prefix(
            patient_id="patient-2",
            slide_id="slide-b",
        ),
    }
    for folder in ["CANCER", "CANCER_MASK", "NOT_CANCER", "NOT_CANCER_MASK"]:
        (tmp_path / folder).mkdir()

    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: True)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)
    monkeypatch.setattr(
        patch_engine,
        "compute_artifact_coverages_from_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("geometry-based artifact coverage should be bypassed")
        ),
    )

    result = cast(
        tuple[str, dict[str, Any], dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SAVED_NOT_CANCER"
    assert result[1]["cov_fold"] == 1.0


def test_process_window_with_slide_builds_cancer_patch_with_nonzero_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (4, 4), color=(10, 20, 30))

    cancer_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (4, 0), (4, 4), (0, 4)]],
        scale_factor=1.0,
    )
    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": 4,
        "use_artifact_filter": False,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 0.01,
        "cancer_polygon_index": cancer_index,
        "not_cancer_polygon_index": ([], None),
        "patient": "patient-1",
        "slide_id": "slide-a",
        "filename_prefix": patch_engine.build_patch_filename_prefix(
            patient_id="patient-1",
            slide_id="slide-a",
        ),
    }

    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: True)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)

    result = cast(
        tuple[str, dict[str, Any], dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SAVED_CANCER"
    assert result[1]["label"] == 1
    assert result[1]["_mask_array"].shape == (4, 4)
    assert np.count_nonzero(result[1]["_mask_array"]) > 0


def test_build_patch_record_accepts_read_only_image_array() -> None:
    patch_np = np.asarray(Image.new("RGB", (4, 4), color=(10, 20, 30)))
    assert patch_np.flags.writeable is False

    record = patch_engine.build_patch_record(
        filename="patch.png",
        label="NOT_CANCER",
        patient_id="p1",
        slide_id="s1",
        artifact_coverages=patch_engine.get_zero_artifact_coverages(),
        patch_np=patch_np,
        final_mask=np.zeros((4, 4), dtype=np.uint8),
    )

    image_array = cast(np.ndarray[Any, Any], record["_image_array"])

    assert image_array.shape == (4, 4, 3)
    assert image_array.flags.writeable is False


def test_process_window_with_slide_skips_overlap_before_reading_slide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def read_region(
            self, _coords: tuple[int, int], _level: int, _size: tuple[int, int]
        ) -> Image.Image:
            raise AssertionError("read_region should not be called for overlap-skipped windows")

    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": 4,
        "use_artifact_filter": False,
        "target_level": 0,
        "tissue_percentage_req": 0.1,
        "match_percentage_req": 1.0,
        "cancer_polygon_index": ([], None),
        "not_cancer_polygon_index": ([], None),
    }
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)

    result = cast(
        tuple[str, object | None, dict[str, Any]],
        patch_engine._process_window_with_slide(FakeSlide(), 0, 0),
    )

    assert result[0] == "SKIPPED_OVERLAP"
    assert result[1] is None
    assert result[2]["read_region"].calls == 0


def test_process_window_batch_returns_profiled_error_when_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        patch_engine,
        "_WORKER_CONTEXT",
        {"path_Image": "/tmp/broken.svs", "profile_output_path": "/tmp/profile.json"},
    )
    monkeypatch.setattr(patch_engine, "_WORKER_SLIDE", None)

    result = cast(
        list[tuple[str, str, dict[str, Any]]], patch_engine.process_window_batch([(0, 0)])
    )

    assert result[0][0] == "ERROR"
    assert "Worker slide handle was not initialized" in result[0][1]
    assert "read_region" in result[0][2]


def test_initialize_worker_opens_slide_once_and_reuses_it_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls: list[str] = []
    closed: list[str] = []

    class FakeSlide:
        def close(self) -> None:
            closed.append("closed")

    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> FakeSlide:
            open_calls.append(path)
            return FakeSlide()

    processed: list[tuple[int, int]] = []

    def fake_process_window_with_slide(
        slide: object, x: int, y: int
    ) -> tuple[str, object, tuple[int, int]]:
        processed.append((x, y))
        return ("OK", slide, (x, y))

    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr("helpers.extraction.patch_engine.atexit.register", lambda callback: None)
    monkeypatch.setattr(patch_engine, "_process_window_with_slide", fake_process_window_with_slide)

    patch_engine._initialize_worker({"path_Image": "/tmp/slide.svs"})
    first = cast(
        list[tuple[str, object, tuple[int, int]]], patch_engine.process_window_batch([(0, 0)])
    )
    second = cast(
        list[tuple[str, object, tuple[int, int]]],
        patch_engine.process_window_batch([(1, 1), (2, 2)]),
    )

    assert open_calls == ["/tmp/slide.svs"]
    assert processed == [(0, 0), (1, 1), (2, 2)]
    assert first[0][0] == "OK"
    assert second[1][2] == (2, 2)

    patch_engine.close_worker_resources()

    assert closed == ["closed"]


def test_initialize_worker_raises_when_slide_open_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> object:
            del path
            raise RuntimeError("boom")

    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr("helpers.extraction.patch_engine.atexit.register", lambda callback: None)

    with pytest.raises(RuntimeError, match="boom"):
        patch_engine._initialize_worker({"path_Image": "/tmp/broken.svs"})


def test_close_worker_resources_is_idempotent() -> None:
    class FakeSlide:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    slide = FakeSlide()
    cast(Any, patch_engine)._WORKER_SLIDE = slide

    patch_engine.close_worker_resources()
    patch_engine.close_worker_resources()

    assert slide.close_calls == 1
    assert patch_engine._WORKER_SLIDE is None


def test_run_extraction_parses_artifact_geojson_and_writes_profile_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSlide:
        level_downsamples = [1.0]
        level_dimensions = [(8, 8)]

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeHandler(BaseHandler):
        def load_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            del slide, kwargs
            return {
                "cancer_polygons": [[(0, 0), (4, 0), (4, 4), (0, 4)]],
                "not_cancer_polygons": [],
            }

        def _load_raw_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            raise NotImplementedError

    slide = FakeSlide()
    geojson_path = tmp_path / "artifacts.geojson"
    geojson_path.write_text(
        """
        {
          "features": [
            {
              "properties": {"classification": {"name": "Fold"}},
              "geometry": {"type": "Polygon", "coordinates": [[[0,0],[2,0],[2,2],[0,2]]]}
            },
            {
              "properties": {"classification": "PenMarking"},
              "geometry": {"type": "MultiPolygon", "coordinates": [[[[3,3],[4,3],[4,4],[3,4]]]]}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    observed_artifacts: dict[str, list[list[tuple[int, int]]]] = {}
    written_profile: dict[str, Any] = {}

    monkeypatch.setattr(
        patch_engine,
        "load_openslide_module",
        lambda: type("OS", (), {"OpenSlide": lambda self, path: slide})(),
    )
    monkeypatch.setattr(patch_engine, "MAX_PRECOMPUTED_MASK_BYTES", 0)
    monkeypatch.setattr(patch_engine, "MAX_PRECOMPUTED_TISSUE_RGB_BYTES", 0)
    monkeypatch.setattr(patch_engine, "HALF_WINDOW", 1)
    monkeypatch.setattr(
        patch_engine,
        "build_artifact_geometry_index",
        lambda artifact_polygons_by_class_level0, scale_factor: (
            observed_artifacts.update(artifact_polygons_by_class_level0) or {}
        ),
    )
    monkeypatch.setattr(
        patch_engine,
        "iter_window_results_serial",
        lambda **kwargs: iter(
            [
                (
                    "SAVED_CANCER",
                    {
                        "filename": "x.png",
                        "label": 1,
                        "patient_id": "p1",
                        "slide_id": "slide",
                        **patch_engine.get_zero_artifact_coverages(),
                    },
                    create_phase_stats(patch_engine.WINDOW_PROFILE_PHASES),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        patch_engine,
        "write_profile_summary",
        lambda path, summary: written_profile.update({"path": path, "summary": summary}),
    )
    monkeypatch.setattr(
        patch_engine,
        "build_profile_summary",
        lambda **kwargs: {
            "slide_name": kwargs["slide_name"],
            "candidate_windows": kwargs["candidate_windows"],
        },
    )

    result = patch_engine.run_extraction(
        FakeHandler(),
        "/tmp/slide.svs",
        target_level=0,
        window_size=4,
        stride=4,
        num_workers=1,
        tissue_percentage_req=0.1,
        match_percentage_req=0.1,
        path_cancer_folder="/tmp/cancer",
        path_not_cancer_folder="/tmp/not_cancer",
        path_cancer_mask_folder="/tmp/cancer_mask",
        path_not_cancer_mask_folder="/tmp/not_cancer_mask",
        patient="p1",
        use_artifact_filter=True,
        path_artifacts_geojson=str(geojson_path),
        profile_output_path=str(tmp_path / "profile.json"),
    )

    assert result[0:2] == (1, 0)
    assert len(result[2]) == 1
    assert len(observed_artifacts["Fold"]) == 1
    assert len(observed_artifacts["PenMarking"]) == 1
    assert written_profile["summary"]["slide_name"] == "slide.svs"
    assert slide.closed is True


def test_build_worker_setup_prefers_precomputed_masks_for_small_regions(tmp_path: Path) -> None:
    class FakeSlide:
        def read_region(
            self, coords: tuple[int, int], level: int, size: tuple[int, int]
        ) -> Image.Image:
            del coords, level, size
            return Image.new("RGB", (4, 4), color=(10, 20, 30))

    request = patch_engine._WorkerSetupRequest(
        path_Image="/tmp/slide.svs",
        kwargs={
            "target_level": 0,
            "window_size": 4,
            "tissue_percentage_req": 0.1,
            "match_percentage_req": 0.1,
            "patient": "p1",
            "num_workers": 1,
            "use_artifact_filter": True,
        },
        slide=FakeSlide(),
        preparation=patch_engine._ExtractionPreparation(
            slide_basename="slide.svs",
            annotations_cancer_level0=[[(0, 0), (4, 0), (4, 4), (0, 4)]],
            annotations_not_cancer_level0=[],
            artifact_polygons_by_class_level0={"Fold": [[(0, 0), (4, 0), (4, 4), (0, 4)]]},
        ),
        region=patch_engine._ScaledAnnotationRegion(
            scale_factor=1.0,
            target_width=4,
            target_height=4,
            x_start=0,
            y_start=0,
            x_end=4,
            y_end=4,
        ),
        filtered_coords=[(0, 0)],
        slide_phase_seconds={},
        profile_output_path=tmp_path / "profile.json",
    )

    worker_setup = patch_engine._build_worker_setup(request)

    assert worker_setup.worker_state["label_region_masks"] is not None
    assert worker_setup.worker_state["artifact_region_masks"] is not None
    assert worker_setup.worker_state["tissue_region_mask"] is not None
    assert worker_setup.worker_state["cancer_polygon_index"] == ([], None)
    assert worker_setup.worker_state["artifact_geometry_index"] == {}


def test_run_extraction_raises_when_workers_report_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSlide:
        level_downsamples = [1.0]
        level_dimensions = [(8, 8)]

        def close(self) -> None:
            return None

    class FakeHandler(BaseHandler):
        def load_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            del slide, kwargs
            return {
                "cancer_polygons": [[(0, 0), (4, 0), (4, 4), (0, 4)]],
                "not_cancer_polygons": [],
            }

        def _load_raw_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
            raise NotImplementedError

    monkeypatch.setattr(
        patch_engine,
        "load_openslide_module",
        lambda: type("OS", (), {"OpenSlide": lambda self, path: FakeSlide()})(),
    )
    monkeypatch.setattr(patch_engine, "MAX_PRECOMPUTED_TISSUE_RGB_BYTES", 0)
    monkeypatch.setattr(patch_engine, "HALF_WINDOW", 1)
    monkeypatch.setattr(
        patch_engine,
        "iter_window_results_serial",
        lambda **kwargs: iter([("ERROR", "traceback text")]),
    )

    with pytest.raises(Exception, match=r"worker process\(es\) failed"):
        patch_engine.run_extraction(
            FakeHandler(),
            "/tmp/slide.svs",
            target_level=0,
            window_size=4,
            stride=4,
            num_workers=1,
            tissue_percentage_req=0.1,
            match_percentage_req=0.1,
            path_cancer_folder="/tmp/cancer",
            path_not_cancer_folder="/tmp/not_cancer",
            path_cancer_mask_folder="/tmp/cancer_mask",
            path_not_cancer_mask_folder="/tmp/not_cancer_mask",
            patient="p1",
        )
