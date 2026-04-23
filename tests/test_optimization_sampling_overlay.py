from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from helpers import optimization_sampling as optimization_sampling_pkg
from helpers.optimization_sampling.overlay import (
    _close_worker_hdf5_handles,
    _overlay_task_order_key,
    _parse_hdf5_ref,
    _process_overlay_task,
    _progress_disabled,
    _progress_file,
    _to_grayscale_mask,
    generate_overlay_images,
    overlay_mask_edges,
)
from helpers.optimization_sampling.sampling import OverlayTask

overlay_module = optimization_sampling_pkg.overlay
NUM_PROCESSES = 3
EXPECTED_SHARD_HANDLE_COUNT = 2


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


def test_overlay_mask_edges_reads_hdf5_image_and_mask_sources(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    output_path = tmp_path / "output" / "overlay.png"
    with h5py.File(source_path, "w") as handle:
        image = np.zeros((1, 6, 6, 3), dtype=np.uint8)
        image[0, 2:4, 2:4, 0] = 255
        mask = np.zeros((1, 6, 6), dtype=np.uint8)
        mask[0, 2:4, 2:4] = 1
        handle.create_dataset("images", data=image)
        handle.create_dataset("masks", data=mask)

    result = overlay_mask_edges(
        f"{source_path}::images[0]",
        f"{source_path}::masks[0]",
        output_path,
        alpha=0.5,
    )

    _close_worker_hdf5_handles()

    assert result is True
    assert output_path.exists()
    overlay = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert int(overlay[:, :, 2].sum()) > 0


def test_overlay_mask_edges_reads_rows_from_multiple_hdf5_shards(tmp_path: Path) -> None:
    shard_a = tmp_path / "HDF5_SHARDS" / "patient_1.h5"
    shard_b = tmp_path / "HDF5_SHARDS" / "patient_2.h5"
    shard_a.parent.mkdir()
    output_a = tmp_path / "output" / "overlay_a.png"
    output_b = tmp_path / "output" / "overlay_b.png"

    for shard_path, bright_channel in ((shard_a, 0), (shard_b, 1)):
        with h5py.File(shard_path, "w") as handle:
            image = np.zeros((1, 6, 6, 3), dtype=np.uint8)
            image[0, 2:4, 2:4, bright_channel] = 255
            mask = np.zeros((1, 6, 6), dtype=np.uint8)
            mask[0, 2:4, 2:4] = 1
            handle.create_dataset("images", data=image)
            handle.create_dataset("masks", data=mask)

    assert overlay_mask_edges(
        f"{shard_a}::images[0]",
        f"{shard_a}::masks[0]",
        output_a,
        alpha=0.5,
    )
    assert overlay_mask_edges(
        f"{shard_b}::images[0]",
        f"{shard_b}::masks[0]",
        output_b,
        alpha=0.5,
    )

    assert output_a.exists()
    assert output_b.exists()
    assert len(overlay_module._WORKER_HDF5_HANDLES) == EXPECTED_SHARD_HANDLE_COUNT

    _close_worker_hdf5_handles()


def test_overlay_task_order_key_groups_rows_by_shard_then_row(tmp_path: Path) -> None:
    shard_a = tmp_path / "HDF5_SHARDS" / "patient_1.h5"
    shard_b = tmp_path / "HDF5_SHARDS" / "patient_2.h5"
    task_b = OverlayTask(
        image_path=f"{shard_b}::images[0]",
        mask_path=f"{shard_b}::masks[0]",
        output_path=tmp_path / "b.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )
    task_a_late = OverlayTask(
        image_path=f"{shard_a}::images[5]",
        mask_path=f"{shard_a}::masks[5]",
        output_path=tmp_path / "a_late.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )
    task_a_early = OverlayTask(
        image_path=f"{shard_a}::images[1]",
        mask_path=f"{shard_a}::masks[1]",
        output_path=tmp_path / "a_early.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )

    ordered = sorted([task_b, task_a_late, task_a_early], key=_overlay_task_order_key)

    assert ordered == [task_a_early, task_a_late, task_b]


def test_generate_overlay_images_returns_empty_list_for_no_tasks() -> None:
    assert generate_overlay_images([], num_processes=2) == []


def test_progress_helpers_use_real_terminal_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStream:
        def isatty(self) -> bool:
            return True

    fake_stream = FakeStream()
    monkeypatch.setattr("helpers.optimization_sampling.overlay.sys.__stderr__", fake_stream)

    assert _progress_file() is fake_stream
    assert _progress_disabled() is False


def test_generate_overlay_images_uses_live_progress_friendly_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = OverlayTask(
        image_path=tmp_path / "image.png",
        mask_path=tmp_path / "mask.png",
        output_path=tmp_path / "out.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )
    seen: dict[str, object] = {}

    class FakePool:
        def __init__(self, processes: int) -> None:
            seen["processes"] = processes

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def imap_unordered(
            self,
            func: Callable[[OverlayTask], bool],
            tasks: Iterable[OverlayTask],
            *,
            chunksize: int,
        ) -> list[bool]:
            seen["func"] = func
            seen["tasks"] = list(tasks)
            seen["chunksize"] = chunksize
            return [True]

    def fake_tqdm(iterable: object, **kwargs: object) -> object:
        seen["tqdm_kwargs"] = kwargs
        return iterable

    monkeypatch.setattr("helpers.optimization_sampling.overlay.Pool", FakePool)
    monkeypatch.setattr("helpers.optimization_sampling.overlay.tqdm", fake_tqdm)
    monkeypatch.setattr("helpers.optimization_sampling.overlay._progress_file", lambda: "stream")
    monkeypatch.setattr("helpers.optimization_sampling.overlay._progress_disabled", lambda: False)

    result = generate_overlay_images([task], num_processes=NUM_PROCESSES)

    assert result == [True]
    assert seen["processes"] == NUM_PROCESSES
    assert seen["chunksize"] == 1
    assert seen["tasks"] == [task]
    assert seen["tqdm_kwargs"] == {
        "total": 1,
        "desc": "Generating Samples",
        "mininterval": 0.5,
        "dynamic_ncols": True,
        "file": "stream",
        "disable": False,
    }


def test_overlay_task_order_key_prefers_hdf5_rows_then_non_hdf5_paths(tmp_path: Path) -> None:
    hdf5_task_late = OverlayTask(
        image_path=f"{tmp_path / 'source.h5'}::images[8]",
        mask_path=f"{tmp_path / 'source.h5'}::masks[8]",
        output_path=tmp_path / "late.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )
    hdf5_task_early = OverlayTask(
        image_path=f"{tmp_path / 'source.h5'}::images[3]",
        mask_path=f"{tmp_path / 'source.h5'}::masks[3]",
        output_path=tmp_path / "early.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )
    png_task = OverlayTask(
        image_path=tmp_path / "image.png",
        mask_path=tmp_path / "mask.png",
        output_path=tmp_path / "png.png",
        color=(1, 2, 3),
        thickness=1,
        alpha=1.0,
    )

    ordered = sorted([hdf5_task_late, png_task, hdf5_task_early], key=_overlay_task_order_key)

    assert ordered == [hdf5_task_early, hdf5_task_late, png_task]


def test_parse_hdf5_ref_extracts_path_and_index(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"

    assert _parse_hdf5_ref(f"{source}::images[12]", "images") == (source, 12)
    assert _parse_hdf5_ref(source / "image.png", "images") is None


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
