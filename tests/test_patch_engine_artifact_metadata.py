from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from shapely.geometry import Polygon

from helpers.extraction.patch_engine import (
    compute_artifact_coverages_for_patch,
    get_zero_artifact_coverages,
)

FOLD_OVERLAP = 0.25
PENMARKING_OVERLAP = 0.2


def test_compute_artifact_coverages_for_patch_returns_per_class_overlap_ratios() -> None:
    patch_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    compute_coverages = cast(Callable[..., dict[str, float]], compute_artifact_coverages_for_patch)

    coverages = compute_coverages(
        artifact_polygons_by_class_level0={
            "Fold": [[(0, 0), (5, 0), (5, 5), (0, 5)]],
            "PenMarking": [[(0, 0), (10, 0), (10, 2), (0, 2)]],
        },
        patch_polygon=patch_polygon,
        scale_factor=1.0,
        patch_area=100.0,
    )

    assert coverages["cov_fold"] == FOLD_OVERLAP
    assert coverages["cov_penmarking"] == PENMARKING_OVERLAP
    assert coverages["cov_oof"] == 0.0
    assert coverages["cov_darkspot_foreign"] == 0.0
    assert coverages["cov_edge_airbubble"] == 0.0


def test_get_zero_artifact_coverages_returns_all_expected_columns() -> None:
    get_zero_coverages = cast(Callable[[], dict[str, Any]], get_zero_artifact_coverages)

    coverages = get_zero_coverages()

    assert coverages == {
        "cov_fold": 0.0,
        "cov_penmarking": 0.0,
        "cov_oof": 0.0,
        "cov_darkspot_foreign": 0.0,
        "cov_edge_airbubble": 0.0,
    }
