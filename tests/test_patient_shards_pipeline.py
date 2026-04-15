import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from helpers.patient_shards.config import PatientShardsConfig
from helpers.patient_shards.io import (
    SourceSplitWriteContext,
)
from helpers.patient_shards.io import (
    build_source_split_write_context as build_source_split_write_context_from_io,
)
from helpers.patient_shards.pipeline import (
    run_patient_shards_pipeline,
    validate_split_shard_integrity,
)


def _write_stage5_split(
    path: Path,
    *,
    patient_ids: list[int],
    labels: list[int],
    pixel_values: list[int],
) -> None:
    filenames = [
        f"PATIENT_{patient_id}_PATCH_{index + 1:03d}.png"
        for index, patient_id in enumerate(patient_ids)
    ]
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [np.full((4, 4, 3), value, dtype=np.uint8) for value in pixel_values],
                axis=0,
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [np.full((4, 4), label * 255, dtype=np.uint8) for label in labels],
                axis=0,
            ),
        )
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.asarray(patient_ids, dtype=np.int32))
        handle.create_dataset("filenames", data=np.asarray([name.encode() for name in filenames]))
        handle.attrs["source_signature"] = f"{path.stem}-signature"
        handle.attrs["source_hdf5_sha256"] = f"{path.stem}-sha"
        handle.attrs["upstream_source_signature"] = "stage4-signature"


def _write_stage5_manifest(base_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 10,
                "filename": "PATIENT_10_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 0,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_002.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 1,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 0,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_003.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 2,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "VALIDATION",
                "label": 1,
                "patient_id": 30,
                "filename": "PATIENT_30_PATCH_001.png",
                "relative_hdf5_path": "VALIDATION.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "TEST",
                "label": 0,
                "patient_id": 40,
                "filename": "PATIENT_40_PATCH_001.png",
                "relative_hdf5_path": "TEST.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
        ]
    ).to_csv(base_dir / "manifest.csv", index=False)


def test_run_patient_shards_pipeline_writes_one_shard_per_patient(tmp_path: Path) -> None:
    stage5_dir = tmp_path / "stage5"
    stage5_dir.mkdir()
    _write_stage5_manifest(stage5_dir)
    _write_stage5_split(
        stage5_dir / "TRAIN.h5",
        patient_ids=[10, 20, 20],
        labels=[1, 0, 0],
        pixel_values=[10, 20, 30],
    )
    _write_stage5_split(
        stage5_dir / "VALIDATION.h5",
        patient_ids=[30],
        labels=[1],
        pixel_values=[40],
    )
    _write_stage5_split(
        stage5_dir / "TEST.h5",
        patient_ids=[40],
        labels=[0],
        pixel_values=[50],
    )

    summary = run_patient_shards_pipeline(
        PatientShardsConfig(
            stage5_base_dir=stage5_dir,
            output_base_dir=tmp_path / "stage6_5",
            overwrite_output=True,
            hdf5_compression="NONE",
            copy_batch_size=2,
            log_folder=tmp_path / "logs",
            log_file_name="patient_shards.log",
        )
    )

    assert summary.total_shards == 4
    assert summary.split_shard_counts == {"TRAIN": 2, "VALIDATION": 1, "TEST": 1}

    with h5py.File(tmp_path / "stage6_5" / "TRAIN_shards" / "20.h5", "r") as handle:
        assert handle["patient_ids"][:].tolist() == [20, 20]
        assert handle["labels"][:].tolist() == [0, 0]
        assert handle["filenames"][:].tolist() == [
            b"PATIENT_20_PATCH_002.png",
            b"PATIENT_20_PATCH_003.png",
        ]
        assert handle.attrs["patient_id"] == 20
        assert handle.attrs["split_name"] == "TRAIN"
        assert handle.attrs["source_signature"] == "TRAIN-signature"
        assert handle.attrs["upstream_source_signature"] == "stage4-signature"
        assert handle.attrs["source_split_hdf5_path"] == str(stage5_dir / "TRAIN.h5")
        assert handle.attrs["source_split_hdf5_sha256"]
        assert json.loads(handle.attrs["source_split_row_indices_json"]) == [1, 2]
        assert handle.attrs["selection_signature"]

    train_summary = json.loads(
        (tmp_path / "stage6_5" / "TRAIN_shards" / "summary.json").read_text()
    )
    assert train_summary["shards"] == [
        {
            "split": "TRAIN",
            "patient_id": 10,
            "relative_hdf5_path": "TRAIN_shards/10.h5",
            "rows": 1,
            "label_0_count": 0,
            "label_1_count": 1,
        },
        {
            "split": "TRAIN",
            "patient_id": 20,
            "relative_hdf5_path": "TRAIN_shards/20.h5",
            "rows": 2,
            "label_0_count": 2,
            "label_1_count": 0,
        },
    ]

    split_manifest = pq.read_table(tmp_path / "stage6_5" / "TRAIN_shards" / "manifest.parquet")
    assert split_manifest.to_pylist() == [
        {
            "split": "TRAIN",
            "patient_id": 10,
            "relative_hdf5_path": "TRAIN_shards/10.h5",
            "rows": 1,
            "label_0_count": 0,
            "label_1_count": 1,
        },
        {
            "split": "TRAIN",
            "patient_id": 20,
            "relative_hdf5_path": "TRAIN_shards/20.h5",
            "rows": 2,
            "label_0_count": 2,
            "label_1_count": 0,
        },
    ]

    sample_manifest = pq.read_table(
        tmp_path / "stage6_5" / "TRAIN_shards" / "sample_manifest.parquet"
    )
    assert sample_manifest.to_pylist() == [
        {
            "split": "TRAIN",
            "patient_id": 10,
            "relative_hdf5_path": "TRAIN_shards/10.h5",
            "row_in_shard": 0,
            "label": 1,
            "filename": "PATIENT_10_PATCH_001.png",
        },
        {
            "split": "TRAIN",
            "patient_id": 20,
            "relative_hdf5_path": "TRAIN_shards/20.h5",
            "row_in_shard": 0,
            "label": 0,
            "filename": "PATIENT_20_PATCH_002.png",
        },
        {
            "split": "TRAIN",
            "patient_id": 20,
            "relative_hdf5_path": "TRAIN_shards/20.h5",
            "row_in_shard": 1,
            "label": 0,
            "filename": "PATIENT_20_PATCH_003.png",
        },
    ]


def test_run_patient_shards_pipeline_builds_source_context_once_per_non_empty_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage5_dir = tmp_path / "stage5"
    stage5_dir.mkdir()
    _write_stage5_manifest(stage5_dir)
    _write_stage5_split(
        stage5_dir / "TRAIN.h5",
        patient_ids=[10, 20, 20],
        labels=[1, 0, 0],
        pixel_values=[10, 20, 30],
    )
    _write_stage5_split(
        stage5_dir / "VALIDATION.h5",
        patient_ids=[30],
        labels=[1],
        pixel_values=[40],
    )
    _write_stage5_split(
        stage5_dir / "TEST.h5",
        patient_ids=[40],
        labels=[0],
        pixel_values=[50],
    )

    build_context_calls: list[Path] = []

    def tracking_build_source_split_write_context(source_path: Path) -> SourceSplitWriteContext:
        build_context_calls.append(source_path)
        return build_source_split_write_context_from_io(source_path)

    monkeypatch.setattr(
        "helpers.patient_shards.pipeline.build_source_split_write_context",
        tracking_build_source_split_write_context,
    )

    run_patient_shards_pipeline(
        PatientShardsConfig(
            stage5_base_dir=stage5_dir,
            output_base_dir=tmp_path / "stage6_5",
            overwrite_output=True,
            hdf5_compression="NONE",
            copy_batch_size=2,
            log_folder=tmp_path / "logs",
            log_file_name="patient_shards.log",
        )
    )

    assert build_context_calls == [
        stage5_dir / "TRAIN.h5",
        stage5_dir / "VALIDATION.h5",
        stage5_dir / "TEST.h5",
    ]


def test_run_patient_shards_pipeline_rejects_missing_non_empty_stage5_split(tmp_path: Path) -> None:
    stage5_dir = tmp_path / "stage5"
    stage5_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 10,
                "filename": "PATIENT_10_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            }
        ]
    ).to_csv(stage5_dir / "manifest.csv", index=False)

    with pytest.raises(FileNotFoundError, match="TRAIN"):
        run_patient_shards_pipeline(
            PatientShardsConfig(
                stage5_base_dir=stage5_dir,
                output_base_dir=tmp_path / "stage6_5",
                overwrite_output=True,
                hdf5_compression="NONE",
                copy_batch_size=2,
                log_folder=tmp_path / "logs",
                log_file_name="patient_shards.log",
            )
        )


def test_validate_split_shard_integrity_rejects_multi_patient_shard(tmp_path: Path) -> None:
    stage5_split = tmp_path / "TRAIN.h5"
    _write_stage5_split(
        stage5_split,
        patient_ids=[10, 20],
        labels=[1, 0],
        pixel_values=[10, 20],
    )
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()
    with h5py.File(shard_dir / "10.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 20], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_20_PATCH_002.png"]),
        )
        handle.attrs["patient_id"] = 10
        handle.attrs["split_name"] = "TRAIN"
        handle.attrs["source_split_hdf5_path"] = str(stage5_split)
        handle.attrs["source_split_row_indices_json"] = json.dumps([0, 1])

    with pytest.raises(ValueError, match="contains patient ids"):
        validate_split_shard_integrity("TRAIN", stage5_split, shard_dir)


def test_validate_split_shard_integrity_rejects_class_count_mismatch(tmp_path: Path) -> None:
    stage5_split = tmp_path / "TRAIN.h5"
    _write_stage5_split(
        stage5_split,
        patient_ids=[10, 10],
        labels=[1, 0],
        pixel_values=[10, 20],
    )
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()
    with h5py.File(shard_dir / "10.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 10], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_10_PATCH_002.png"]),
        )
        handle.attrs["patient_id"] = 10
        handle.attrs["split_name"] = "TRAIN"
        handle.attrs["source_split_hdf5_path"] = str(stage5_split)
        handle.attrs["source_split_row_indices_json"] = json.dumps([0, 1])

    with pytest.raises(ValueError, match="label/filename alignment"):
        validate_split_shard_integrity("TRAIN", stage5_split, shard_dir)
