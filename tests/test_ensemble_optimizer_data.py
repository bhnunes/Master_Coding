from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.ensemble_optimizer import data as optimizer_data

PATCH_SIDE = 4
RGB_CHANNELS = 3
EXPECTED_ROW_COUNT = 4
EXPECTED_SHARD_COUNT = 2
FILTERED_PATIENT_SAMPLE_COUNT = 2
FIRST_IMAGE_PIXEL = 10
SECOND_IMAGE_PIXEL = 20
THIRD_IMAGE_PIXEL = 30
VALIDATION_BATCH_SIZE = 2


def _write_stage2_shard(
    shard_path: Path,
    *,
    patient_id: int,
    pixel_values: list[int],
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
                [np.full((PATCH_SIDE, PATCH_SIDE), label, dtype=np.uint8) for label in labels]
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
            filenames = [f"p{patient_id}_{row_index}.png" for row_index in range(2)]
            labels = [0, 1]
            for row_index, (filename, label) in enumerate(zip(filenames, labels, strict=True)):
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                        source_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(shard_path),
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
                    ) VALUES (?, 'VALIDATION', 'NOT_NORMALIZED', NULL, 1, NULL, 'STAGE4')
                    """,
                    (cursor.lastrowid,),
                )
        connection.commit()


class _ChannelFirstTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(np.moveaxis(mask, -1, 0)),
        }


class _ChannelLastTransform:
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
def optimizer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[Path]]:
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _ChannelFirstTransform(),
    )
    shard_paths = [tmp_path / "PATCHES" / f"patient_{patient_id}.h5" for patient_id in (1, 2)]
    _write_stage2_shard(
        shard_paths[0],
        patient_id=1,
        pixel_values=[10, 20],
        labels=[0, 1],
        filenames=["p1_0.png", "p1_1.png"],
    )
    _write_stage2_shard(
        shard_paths[1],
        patient_id=2,
        pixel_values=[30, 40],
        labels=[1, 0],
        filenames=["p2_0.png", "p2_1.png"],
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_master_manifest(master_manifest_path, shard_paths)
    return master_manifest_path, shard_paths


def test_setup_validation_data_returns_layout_without_local_staging(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture

    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    assert layout.master_manifest_path == master_manifest_path
    assert layout.local_cache_dir is None
    assert len(layout.records) == EXPECTED_ROW_COUNT


def test_setup_validation_data_prepares_local_cache_when_enabled(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture

    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=True,
    )

    assert layout.local_cache_dir == master_manifest_path.parent / "local" / "patient_shards"
    assert layout.local_cache_dir.exists()


def test_collect_validation_provenance_reports_manifest_metadata(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    provenance = optimizer_data.collect_validation_provenance(layout)

    assert provenance["master_manifest_path"] == str(master_manifest_path)
    assert provenance["row_count"] == EXPECTED_ROW_COUNT
    assert provenance["shard_count"] == EXPECTED_SHARD_COUNT


def test_validation_dataset_builds_two_channel_mask(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout)
    image, mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[1])

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (2, 4, 4)
    assert patient_id == "1"
    assert torch.equal(mask.sum(dim=0), torch.ones((4, 4), dtype=mask.dtype))


def test_validation_dataset_can_filter_to_allowed_patients(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout, allowed_patients={"2"})

    assert len(dataset) == FILTERED_PATIENT_SAMPLE_COUNT
    _image, _mask, patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[0])
    assert patient_id == "2"


def test_validation_dataset_reads_canonical_rows_in_manifest_order(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout)
    first_image, _first_mask, first_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[0]
    )
    second_image, _second_mask, second_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[1]
    )
    third_image, _third_mask, third_patient_id = cast(
        tuple[torch.Tensor, torch.Tensor, str], dataset[2]
    )

    assert int(first_image[0, 0, 0]) == FIRST_IMAGE_PIXEL
    assert int(second_image[0, 0, 0]) == SECOND_IMAGE_PIXEL
    assert int(third_image[0, 0, 0]) == THIRD_IMAGE_PIXEL
    assert first_patient_id == "1"
    assert second_patient_id == "1"
    assert third_patient_id == "2"


def test_validation_dataset_permute_branch_handles_hwc_mask_output(
    optimizer_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _ChannelLastTransform(),
    )
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout)
    _image, mask, _patient_id = cast(tuple[torch.Tensor, torch.Tensor, str], dataset[0])

    assert tuple(mask.shape) == (2, 4, 4)


def test_validation_dataset_returns_none_triplet_when_transform_fails(
    optimizer_fixture: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTransform:
        def __call__(
            self, *, image: npt.NDArray[np.generic], mask: npt.NDArray[np.generic]
        ) -> dict[str, torch.Tensor]:
            del image, mask
            raise RuntimeError("boom")

    master_manifest_path, _shard_paths = optimizer_fixture
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: FailingTransform(),
    )
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout)

    assert dataset[0] == (None, None, None)


def test_validation_dataset_state_reset_clears_open_handles(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataset = optimizer_data.ValidationDataset(layout)
    _ = dataset[0]

    state = dataset.__getstate__()
    assert state["h5_file"] is None
    assert state["_atexit_registered"] is False

    dataset.__setstate__(state)

    assert dataset.h5_file is None
    assert dataset.masks_dset is None
    assert dataset._opened_pid is None


def test_collate_validation_batch_filters_invalid_items() -> None:
    image = torch.ones((3, 4, 4), dtype=torch.float32)
    mask = torch.zeros((2, 4, 4), dtype=torch.float32)

    batch = optimizer_data.collate_validation_batch([(image, mask, "p1"), (None, None, None), None])

    assert batch is not None
    images, masks, patient_ids = batch
    assert tuple(images.shape) == (1, 3, 4, 4)
    assert tuple(masks.shape) == (1, 2, 4, 4)
    assert patient_ids == ["p1"]


def test_create_validation_dataloader_uses_expected_collate_and_worker_init(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    dataloader = optimizer_data.create_validation_dataloader(
        layout,
        batch_size=VALIDATION_BATCH_SIZE,
        workers=0,
    )

    assert dataloader.collate_fn is optimizer_data.collate_validation_batch
    assert dataloader.worker_init_fn is optimizer_data.worker_init_fn  # type: ignore[attr-defined]


def test_summarize_validation_data_preserves_patient_order_and_positive_labels(
    optimizer_fixture: tuple[Path, list[Path]],
) -> None:
    master_manifest_path, _shard_paths = optimizer_fixture
    layout = optimizer_data.setup_validation_data(
        master_manifest_path,
        master_manifest_path.parent / "local",
        stage_input_locally=False,
    )

    ordered_patients, positive_patients = optimizer_data.summarize_validation_data(layout)

    assert ordered_patients == ["1", "2"]
    assert positive_patients == {"1", "2"}
