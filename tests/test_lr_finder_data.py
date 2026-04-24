from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch.utils.data import Dataset

from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref
from helpers.lr_finder import data as lr_data
from helpers.lr_finder.config import BCEDiceSearchSpace, LRFinderConfig, ModelPlan

PATCH_SIDE = 4
RGB_CHANNELS = 3
LR_FINDER_BATCH_SIZE = 2
LR_FINDER_LHS_SAMPLES = 2
LR_FINDER_END_LR = 0.1
LR_FINDER_NUM_ITER = 5
WEIGHT_DECAY = 1e-4
START_LR = 1e-8
VALIDATION_ROW_COUNT = 2
FIRST_TRAIN_PIXEL = 11
SECOND_TRAIN_PIXEL = 44
NORMALIZED_FIRST_PIXEL = 16


def _build_config(tmp_path: Path, *, smart_sampling: bool = False) -> LRFinderConfig:
    return LRFinderConfig(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "local",
        stage_input_locally=False,
        overwrite_output=True,
        smart_sampling=smart_sampling,
        execution_mode="PAPER",
        amp_precision="fp32",
        seed=24,
        batch_size=LR_FINDER_BATCH_SIZE,
        workers=0,
        use_subset=False,
        subset_ratio=1.0,
        num_lhs_samples=LR_FINDER_LHS_SAMPLES,
        end_lr=LR_FINDER_END_LR,
        num_iter=LR_FINDER_NUM_ITER,
        num_repeats=1,
        optimizer_weight_decay=WEIGHT_DECAY,
        optimizer_start_lr=START_LR,
        pdf_name="report.pdf",
        hf_token=None,
        search_space=BCEDiceSearchSpace(),
        model_plans=[ModelPlan(architecture="FPN", encoder="resnet34")],
    )


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
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL,
                source_slide_path TEXT NOT NULL,
                annotation_path TEXT,
                artifacts_geojson_path TEXT,
                stage2_case_record_id INTEGER NOT NULL,
                stage2_processing_signature TEXT,
                stage2_status TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
        state_path = master_manifest_path.parent / "normalization_stats.json"
        state_path.write_text('{"method": "NOT_NORMALIZED"}', encoding="utf-8")
        connection.execute(
            """
            INSERT INTO normalization_artifacts (
                run_id, method, state_path, state_sha256, template_path, template_sha256, fit_scope
            ) VALUES (1, 'NOT_NORMALIZED', ?, 'unused', NULL, NULL, 'TRAIN')
            """,
            (to_manifest_path_ref(state_path, manifest_path=master_manifest_path),),
        )
        selected_rows = {(str(shard_paths[0]), 0), (str(shard_paths[1]), 1)}
        split_by_patient = {1: "TRAIN", 2: "TRAIN", 3: "VALIDATION"}
        for shard_path in shard_paths:
            patient_id = int(shard_path.stem.replace("patient_", ""))
            filenames = [f"p{patient_id}_{row_index}.png" for row_index in range(2)]
            labels = [0, 1]
            for row_index, (filename, label) in enumerate(zip(filenames, labels, strict=True)):
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path,
                        source_row_index,
                        filename,
                        patient_id,
                        label,
                        slide_id,
                        source_signature,
                        source_image_path,
                        source_mask_path,
                        source_slide_path,
                        stage2_case_record_id,
                        stage2_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        to_manifest_path_ref(shard_path, manifest_path=master_manifest_path),
                        row_index,
                        filename,
                        patient_id,
                        label,
                        f"slide_{patient_id}",
                        f"sig_{patient_id}",
                        build_hdf5_dataset_ref("images", row_index),
                        build_hdf5_dataset_ref("masks", row_index),
                        f"/slides/{patient_id}.svs",
                        patient_id,
                        "COMPLETED",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO patch_stage_state (
                        patch_id,
                        split,
                        normalization_method,
                        normalization_artifact_id,
                        sampling_decision,
                        is_stage4_accepted,
                        is_stage7_selected,
                        last_updated_stage_name
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, 'STAGE6')
                    """,
                    (
                        cursor.lastrowid,
                        split_by_patient[patient_id],
                        "NOT_NORMALIZED",
                        None,
                        (
                            "sampled_kept"
                            if (str(shard_path), row_index) in selected_rows
                            else "rejected_reducible"
                        ),
                        1 if (str(shard_path), row_index) in selected_rows else 0,
                    ),
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
            "mask": torch.from_numpy(np.asarray(mask)),
        }


class _RecordingNormalizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def normalize_image(
        self,
        image: npt.NDArray[np.uint8],
        *,
        cache_key: object | None = None,
    ) -> npt.NDArray[np.uint8]:
        self.calls.append(str(cache_key))
        return np.asarray(image + 5, dtype=np.uint8)


class _TinyDataset(Dataset[tuple[str, str]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[str, str]:
        del index
        return ("image", "mask")


@pytest.fixture
def lr_finder_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LRFinderConfig, list[Path]]:
    monkeypatch.setattr(
        "helpers.training.canonical_dataset.get_transforms",
        lambda mode, img_size: _IdentityTransform(),
    )
    shard_paths = [tmp_path / "PATCHES" / f"patient_{patient_id}.h5" for patient_id in (1, 2, 3)]
    _write_stage2_shard(
        shard_paths[0],
        patient_id=1,
        pixel_values=[11, 22],
        labels=[0, 1],
        filenames=["p1_0.png", "p1_1.png"],
    )
    _write_stage2_shard(
        shard_paths[1],
        patient_id=2,
        pixel_values=[33, 44],
        labels=[0, 1],
        filenames=["p2_0.png", "p2_1.png"],
    )
    _write_stage2_shard(
        shard_paths[2],
        patient_id=3,
        pixel_values=[55, 66],
        labels=[0, 1],
        filenames=["p3_0.png", "p3_1.png"],
    )
    config = _build_config(tmp_path, smart_sampling=True)
    _write_master_manifest(config.master_manifest_path, shard_paths)
    return config, shard_paths


def test_prepare_training_data_uses_sqlite_rows_for_smart_sampling(
    lr_finder_fixture: tuple[LRFinderConfig, list[Path]],
) -> None:
    config, _shard_paths = lr_finder_fixture

    prepared = lr_data.prepare_training_data(config)
    dataset = cast(Any, prepared.dataset)

    assert prepared.source_split_name == "TRAIN_SELECTED"
    assert dataset.get_labels().tolist() == [0, 1]
    assert dataset.get_patient_ids().tolist() == ["1", "2"]
    assert prepared.training_provenance["split"] == "TRAIN"
    assert prepared.training_provenance["smart_sampling"] is True
    assert prepared.validation_provenance["split"] == "VALIDATION"
    assert prepared.validation_provenance["row_count"] == VALIDATION_ROW_COUNT


def test_prepare_training_data_reads_canonical_stage2_rows(
    lr_finder_fixture: tuple[LRFinderConfig, list[Path]],
) -> None:
    config, _shard_paths = lr_finder_fixture

    prepared = lr_data.prepare_training_data(config)
    first_image, first_mask = prepared.dataset[0]
    second_image, second_mask = prepared.dataset[1]

    assert int(first_image[0, 0, 0]) == FIRST_TRAIN_PIXEL
    assert int(second_image[0, 0, 0]) == SECOND_TRAIN_PIXEL
    assert int(first_mask[0, 0]) == 0
    assert int(second_mask[0, 0]) == 1


def test_prepare_training_data_uses_shared_stain_normalizer(
    lr_finder_fixture: tuple[LRFinderConfig, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _shard_paths = lr_finder_fixture
    normalizer = _RecordingNormalizer()
    monkeypatch.setattr(lr_data, "build_split_stain_normalizer", lambda *args, **kwargs: normalizer)

    prepared = lr_data.prepare_training_data(config)
    image, _mask = prepared.dataset[0]

    assert normalizer.calls == ["p1_0.png"]
    assert int(image[0, 0, 0]) == NORMALIZED_FIRST_PIXEL


def test_prepare_training_data_builds_weights_and_subset(
    lr_finder_fixture: tuple[LRFinderConfig, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _shard_paths = lr_finder_fixture
    config = LRFinderConfig(**{**config.__dict__, "use_subset": True, "subset_ratio": 0.5})

    def fake_subset(dataset: Any, ratio: float, split_name: str, seed: int) -> Any:
        del dataset, ratio, split_name, seed
        return type("FakeSubset", (), {"indices": [1]})()

    monkeypatch.setattr(lr_data, "create_stratified_subset_within_patients", fake_subset)

    prepared = lr_data.prepare_training_data(config)
    dataset = cast(Any, prepared.dataset)

    assert dataset.get_labels().tolist() == [1]
    assert prepared.sample_weights.tolist() == [1.0]
    assert prepared.training_provenance["row_count"] == 1


def test_build_train_loader_uses_weighted_sampler() -> None:
    dataset = _TinyDataset()
    sample_weights = torch.tensor([2.0], dtype=torch.float32)

    loader = lr_data.build_train_loader(dataset, sample_weights, batch_size=1, workers=0, seed=24)

    assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert loader.worker_init_fn is lr_data.worker_init_fn  # type: ignore[attr-defined]
