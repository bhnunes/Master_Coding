# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from helpers.extraction import patch_engine
from helpers.extraction.data_handlers import BaseHandler
from helpers.extraction.profiling import create_phase_stats


def test_check_tissue_percentage_robust_handles_empty_patch() -> None:
    assert patch_engine.check_tissue_percentage_robust(None, 0.1) is False
    assert patch_engine.check_tissue_percentage_robust(np.array([]), 0.1) is False


def test_polygons_to_mask_fills_intersection_inside_patch() -> None:
    mask = patch_engine.polygons_to_mask(
        mask_shape=(4, 4),
        polygons_level0=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
        scale_factor=1.0,
        patch_coords=(0, 0),
    )

    assert mask.shape == (4, 4)
    assert np.count_nonzero(mask) >= 4


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


def test_compute_artifact_coverages_from_index_returns_overlap_fraction() -> None:
    index = patch_engine.build_artifact_geometry_index(
        {"Fold": [[(0, 0), (5, 0), (5, 5), (0, 5)]]},
        scale_factor=1.0,
    )

    coverages = patch_engine.compute_artifact_coverages_from_index(
        index,
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        100.0,
    )

    assert coverages["cov_fold"] == 0.25
    assert coverages["cov_penmarking"] == 0.0


def test_polygons_to_mask_with_index_returns_empty_mask_without_tree() -> None:
    mask = patch_engine.polygons_to_mask_with_index((4, 4), ([], None), (0, 0))

    assert np.count_nonzero(mask) == 0


def test_polygons_to_mask_with_index_fills_matching_polygon() -> None:
    polygon_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (3, 0), (3, 3), (0, 3)]],
        scale_factor=1.0,
    )

    mask = patch_engine.polygons_to_mask_with_index((4, 4), polygon_index, (0, 0))

    assert np.count_nonzero(mask) >= 4


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
    assert all(coords.shape[1] == 2 for coords in clipped)


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
            return Image.new("RGB", (4, 4), color=(10, 20, 30))

    monkeypatch.setattr(patch_engine, "check_tissue_percentage_robust", lambda patch, req: False)
    monkeypatch.setattr(patch_engine, "PATCH_AREA", 16)
    not_cancer_index = patch_engine.build_scaled_polygon_index(
        [[(0, 0), (4, 0), (4, 4), (0, 4)]],
        scale_factor=1.0,
    )
    patch_engine._WORKER_CONTEXT = {
        "profile_output_path": "/tmp/profile.json",
        "window_size": 4,
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

    result = patch_engine._process_window_with_slide(FakeSlide(), 1, 2)

    assert result[0] == "SKIPPED_TISSUE"
    assert result[1] is None
    assert "tissue_check" in result[2]
    assert result[2]["read_region"].calls == 1


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

    result = patch_engine._process_window_with_slide(FakeSlide(), 0, 0)

    assert result[0] == "SAVED_NOT_CANCER"
    assert result[1]["label"] == 0
    assert result[1]["slide_id"] == "slide-b"
    assert result[1]["cov_fold"] == 0.25
    assert result[1]["_image_array"].shape == (4, 4, 3)
    assert result[1]["_mask_array"].shape == (4, 4)


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

    result = patch_engine._process_window_with_slide(FakeSlide(), 0, 0)

    assert result[0] == "SKIPPED_OVERLAP"
    assert result[1] is None
    assert result[2]["read_region"].calls == 0


def test_process_window_batch_returns_profiled_error_when_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> object:
            del path
            raise RuntimeError("boom")

    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr(
        patch_engine,
        "_WORKER_CONTEXT",
        {"path_Image": "/tmp/broken.svs", "profile_output_path": "/tmp/profile.json"},
    )

    result = patch_engine.process_window_batch([(0, 0)])

    assert result[0][0] == "ERROR"
    assert "RuntimeError: boom" in result[0][1]
    assert "read_region" in result[0][2]


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
        "iter_window_results",
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
    monkeypatch.setattr(patch_engine, "HALF_WINDOW", 1)
    monkeypatch.setattr(
        patch_engine,
        "iter_window_results",
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
