from __future__ import annotations

from shapely.geometry import Polygon

from helpers.patch_engine import compute_artifact_coverages_for_patch, get_zero_artifact_coverages


def test_compute_artifact_coverages_for_patch_returns_per_class_overlap_ratios() -> None:
    patch_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    coverages = compute_artifact_coverages_for_patch(
        artifact_polygons_by_class_level0={
            "Fold": [[(0, 0), (5, 0), (5, 5), (0, 5)]],
            "PenMarking": [[(0, 0), (10, 0), (10, 2), (0, 2)]],
        },
        patch_polygon=patch_polygon,
        scale_factor=1.0,
        patch_area=100.0,
    )

    assert coverages["cov_fold"] == 0.25
    assert coverages["cov_penmarking"] == 0.2
    assert coverages["cov_oof"] == 0.0
    assert coverages["cov_darkspot_foreign"] == 0.0
    assert coverages["cov_edge_airbubble"] == 0.0


def test_get_zero_artifact_coverages_returns_all_expected_columns() -> None:
    coverages = get_zero_artifact_coverages()

    assert coverages == {
        "cov_fold": 0.0,
        "cov_penmarking": 0.0,
        "cov_oof": 0.0,
        "cov_darkspot_foreign": 0.0,
        "cov_edge_airbubble": 0.0,
    }
