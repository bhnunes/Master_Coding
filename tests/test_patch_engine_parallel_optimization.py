# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from shapely.geometry import Polygon

from helpers.extraction import patch_engine


def test_chunk_coordinates_groups_work_into_stable_batches() -> None:
    chunk_coordinates = cast(
        Callable[..., Iterator[list[tuple[int, int]]]],
        patch_engine.chunk_coordinates,
    )
    batches = list(chunk_coordinates([(1, 2), (3, 4), (5, 6)], batch_size=2))

    assert batches == [[(1, 2), (3, 4)], [(5, 6)]]


def test_process_window_batch_reuses_single_slide_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_paths: list[str] = []
    closed_slides: list[str] = []

    class FakeSlide:
        def __init__(self, path: str) -> None:
            self.path = path

        def close(self) -> None:
            closed_slides.append(self.path)

    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> FakeSlide:
            opened_paths.append(path)
            return FakeSlide(path)

    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr("helpers.extraction.patch_engine.atexit.register", lambda callback: None)
    monkeypatch.setattr(
        patch_engine,
        "_process_window_with_slide",
        lambda slide, x, y: (f"SAVED_{x}_{y}", {"slide": slide.path, "coords": (x, y)}),
    )

    process_window_batch = cast(
        Callable[[list[tuple[int, int]]], list[tuple[str, dict[str, Any]]]],
        patch_engine.process_window_batch,
    )

    patch_engine._initialize_worker({"path_Image": "/tmp/sample.svs"})
    try:
        results = process_window_batch([(10, 20), (30, 40)])
    finally:
        patch_engine.close_worker_resources()

    assert opened_paths == ["/tmp/sample.svs"]
    assert closed_slides == ["/tmp/sample.svs"]
    assert results == [
        ("SAVED_10_20", {"slide": "/tmp/sample.svs", "coords": (10, 20)}),
        ("SAVED_30_40", {"slide": "/tmp/sample.svs", "coords": (30, 40)}),
    ]


def test_iter_window_results_flattens_batch_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_process_window_batch(batch: list[tuple[int, int]]) -> list[tuple[str, dict[str, Any]]]:
        return [(f"SAVED_{x}_{y}", {"coords": (x, y)}) for x, y in batch]

    class FakeSlide:
        def close(self) -> None:
            return None

    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> FakeSlide:
            assert path == "/tmp/sample.svs"
            return FakeSlide()

    class FakePool:
        def __init__(self, *, processes: int, initializer: Any, initargs: tuple[Any, ...]) -> None:
            assert processes == 2
            initializer(*initargs)

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def imap_unordered(
            self, func: Any, batches: Iterator[list[tuple[int, int]]], chunksize: int
        ) -> Iterator[list[tuple[str, dict[str, Any]]]]:
            assert func is fake_process_window_batch
            assert chunksize == 1
            for batch in batches:
                yield func(batch)

    monkeypatch.setattr(patch_engine, "Pool", FakePool)
    monkeypatch.setattr(patch_engine, "process_window_batch", fake_process_window_batch)
    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr("helpers.extraction.patch_engine.atexit.register", lambda callback: None)

    worker_state: dict[str, Any] = {"path_Image": "/tmp/sample.svs"}
    iter_window_results = cast(
        Callable[..., Iterator[tuple[str, dict[str, Any]]]], patch_engine.iter_window_results
    )
    try:
        results = list(
            iter_window_results(
                filtered_coords=[(1, 1), (2, 2), (3, 3)],
                num_workers=2,
                worker_state=worker_state,
                batch_size=2,
            )
        )
    finally:
        patch_engine.close_worker_resources()

    assert results == [
        ("SAVED_1_1", {"coords": (1, 1)}),
        ("SAVED_2_2", {"coords": (2, 2)}),
        ("SAVED_3_3", {"coords": (3, 3)}),
    ]


def test_initialize_worker_prepares_artifact_geometry_index_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSlide:
        def close(self) -> None:
            return None

    class FakeOpenSlideModule:
        def OpenSlide(self, path: str) -> FakeSlide:
            assert path == "/tmp/sample.svs"
            return FakeSlide()

    monkeypatch.setattr(patch_engine, "load_openslide_module", lambda: FakeOpenSlideModule())
    monkeypatch.setattr("helpers.extraction.patch_engine.atexit.register", lambda callback: None)

    worker_state: dict[str, Any] = {
        "path_Image": "/tmp/sample.svs",
        "artifact_geometry_index": patch_engine.build_artifact_geometry_index(
            {"Fold": [[(0, 0), (4, 0), (4, 4), (0, 4)]]},
            scale_factor=1.0,
        ),
    }

    try:
        patch_engine._initialize_worker(worker_state)
        artifact_geometry, prepared_geometry = patch_engine._WORKER_CONTEXT[
            "artifact_geometry_index"
        ]["cov_fold"]
    finally:
        patch_engine.close_worker_resources()

    assert artifact_geometry.bounds == (0.0, 0.0, 4.0, 4.0)
    assert prepared_geometry.intersects(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
