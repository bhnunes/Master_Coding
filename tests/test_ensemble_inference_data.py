from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.ensemble_inference import data as inference_data
from helpers.extraction.manifest_paths import to_manifest_path_ref

PATCH_SIDE = 4
RGB_CHANNELS = 3
EXPECTED_RECORD_COUNT = 3
EXPECTED_SHARD_COUNT = 2
FIRST_IMAGE_PIXEL = 7
SECOND_IMAGE_PIXEL = 8
THIRD_IMAGE_PIXEL = 9
FULL_MASK_SUM = 16
TEST_BATCH_SIZE = 2
PREFETCH_FACTOR = 4


def _write_stage2_shard(
    shard_path: Path,
    *,
    patient_id: int,
    pixel_values: list[int],
    mask_values: list[int],
    labels: list[int],
    filenames: list[str],
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.full((PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS), pixel_value, dtype=np.uint8)
                    for pixel_value in pixel_values
                ]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [
                    np.full((PATCH_SIDE, PATCH_SIDE), mask_value, dtype=np.uint8)
                    for mask_value in mask_values
                ]
            ),
        )
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.uint8))
        handle.create_dataset(
            "patient_ids",
            data=np.asarray([patient_id] * len(labels), dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.asarray([filename.encode("utf-8") for filename in filenames]),
        )


def _write_master_manifest(master_manifest_path: Path, shard_paths: list[Path]) -> None:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                slide_id TEXT,
                source_signature TEXT,
                UNIQUE (source_hdf5_path, source_row_index)
            );
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            """
        )
        for shard_path in shard_paths:
            patient_id = int(shard_path.stem.replace("patient_", ""))
            shard_rows = {
                1: [(1, 8, "p1_1.png"), (0, 7, "p1_0.png")],
                2: [(0, 9, "p2_0.png")],
            }[patient_id]
            for row_index, row in enumerate(shard_rows):
                label, _pixel_value, filename = row
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                        source_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        to_manifest_path_ref(shard_path, manifest_path=master_manifest_path),
                        row_index,
                        filename,
                        patient_id,
                        label,
                        f"slide_{patient_id}",
                        f"sig_{patient_id}",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO patch_stage_state (
                        patch_id, split, normalization_method, normalization_artifact_id,
                        is_stage4_accepted, is_stage7_selected, last_updated_stage_name
                    ) VALUES (?, 'TEST', 'NOT_NORMALIZED', NULL, 1, NULL, 'STAGE4')
                    """,
                    (cursor.lastrowid,),
                )
        connection.commit()


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(mask),
        }


@pytest.fixture
def inference_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[Path]]:
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _IdentityTransform(),
    )
    shard_paths = [tmp_path / "PATCHES" / f"patient_{patient_id}.h5" for patient_id in (1, 2)]
    _write_stage2_shard(
        shard_paths[0],
        patient_id=1,
        pixel_values=[7, 8],
        mask_values=[0, 1],
        labels=[1, 1],
        filenames=["p1_0.png", "p1_1.png"],
    )
    _write_stage2_shard(
        shard_paths[1],
        patient_id=2,
        pixel_values=[9],
        mask_values=[3],
        labels=[1],
        filenames=["p2_0.png"],
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_master_manifest(master_manifest_path, shard_paths)
    return master_manifest_path, shard_paths


def test_setup_test_data_returns_layout_when_staging_disabled(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture

    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    assert layout.master_manifest_path == master_manifest_path
    assert layout.local_cache_dir is None
    assert len(layout.records) == EXPECTED_RECORD_COUNT


def test_setup_test_data_prepares_local_cache_when_staging_enabled(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture

    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=True,
    )

    assert layout.local_cache_dir == master_manifest_path.parent / "local" / "patient_shards"
    assert layout.local_cache_dir.exists()


def test_setup_test_data_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing"):
        inference_data.setup_test_data(
            tmp_path / "missing.sqlite",
            tmp_path / "local",
            stage_input_locally=False,
        )


def test_collect_test_dataset_provenance_reports_manifest_metadata(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture
    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    provenance = inference_data.collect_test_dataset_provenance(layout)

    assert provenance["master_manifest_path"] == str(master_manifest_path)
    assert provenance["row_count"] == EXPECTED_RECORD_COUNT
    assert provenance["shard_count"] == EXPECTED_SHARD_COUNT
    assert provenance["attrs"] == {
        "master_manifest_sha256": provenance["master_manifest_sha256"],
        "stage4_split_bundle_id": None,
        "runtime_normalization_method": "none",
        "normalization_method": "none",
        "normalization_artifact_id": None,
    }


def test_test_dataset_reads_canonical_rows_in_manifest_order(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture
    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = inference_data.TestDataset(layout)
    first_image, first_mask, first_patient_id, first_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str], dataset[0]
    )
    second_image, second_mask, second_patient_id, second_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str], dataset[1]
    )
    third_image, third_mask, third_patient_id, third_filename = cast(
        tuple[torch.Tensor, torch.Tensor, str, str], dataset[2]
    )

    assert int(first_image[0, 0, 0]) == FIRST_IMAGE_PIXEL
    assert int(second_image[0, 0, 0]) == SECOND_IMAGE_PIXEL
    assert int(third_image[0, 0, 0]) == THIRD_IMAGE_PIXEL
    assert int(first_mask.sum()) == 0
    assert int(second_mask.sum()) == FULL_MASK_SUM
    assert int(third_mask.sum()) == FULL_MASK_SUM
    assert first_patient_id == "1"
    assert second_patient_id == "1"
    assert third_patient_id == "2"
    assert first_filename == "p1_0.png"
    assert second_filename == "p1_1.png"
    assert third_filename == "p2_0.png"


def test_test_dataset_returns_none_triplet_when_transform_fails(
    inference_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTransform:
        def __call__(
            self,
            *,
            image: npt.NDArray[np.generic],
            mask: npt.NDArray[np.generic],
        ) -> dict[str, torch.Tensor]:
            del image, mask
            raise RuntimeError("boom")

    master_manifest_path, _shard_paths = inference_fixture
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: FailingTransform(),
    )
    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = inference_data.TestDataset(layout)

    assert dataset[0] == (None, None, None, None)


def test_test_dataset_state_reset_clears_open_handles(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture
    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = inference_data.TestDataset(layout)
    _ = dataset[0]

    state = dataset.__getstate__()
    assert state["h5_file"] is None
    assert state["_atexit_registered"] is False

    dataset.__setstate__(state)

    assert dataset.h5_file is None
    assert dataset.images_dset is None
    assert dataset._opened_pid is None


def test_collate_test_batch_filters_invalid_items() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.uint8)
    mask = torch.zeros((4, 4), dtype=torch.uint8)

    batch = inference_data.collate_test_batch(
        [(image, mask, "p1", "f1.png"), (None, None, None, None), None]
    )

    assert batch is not None
    images, masks, patient_ids, filenames = batch
    assert tuple(images.shape) == (1, 3, 4, 4)
    assert tuple(masks.shape) == (1, 4, 4)
    assert patient_ids == ["p1"]
    assert filenames == ["f1.png"]


def test_collate_test_batch_returns_none_when_all_items_invalid() -> None:
    assert inference_data.collate_test_batch([(None, None, None, None), None]) is None


def test_create_test_dataloader_sets_worker_dependent_flags(
    inference_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = inference_fixture
    layout = inference_data.setup_test_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    loader = inference_data.create_test_dataloader(
        layout,
        batch_size=TEST_BATCH_SIZE,
        workers=1,
    )

    assert loader.batch_size == TEST_BATCH_SIZE
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == PREFETCH_FACTOR
    assert loader.collate_fn is inference_data.collate_test_batch
    assert loader.worker_init_fn is inference_data.worker_init_fn  # type: ignore[attr-defined]
