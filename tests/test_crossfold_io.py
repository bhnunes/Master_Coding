from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from helpers.crossfold.io import verify_split_hdf5_integrity, write_split_hdf5


def test_write_split_hdf5_writes_split_contract_from_source_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.array(
                [np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4, 3), 20, dtype=np.uint8)]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.array([np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]),
        )
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 20], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_20_PATCH_001.png"]),
        )

    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 1,
            }
        ]
    )

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["labels"][:].tolist() == [1]
        assert handle["patient_ids"][:].tolist() == [20]
        assert handle["filenames"][:].tolist() == [b"PATIENT_20_PATCH_001.png"]

    verify_split_hdf5_integrity(output_path, split_df)


def test_write_split_hdf5_applies_normalizer_when_provided(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([20], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_20_PATCH_001.png"]))

    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 0,
            }
        ]
    )

    class FakeNormalizer:
        def transform(self, image_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
            return (image_rgb + 10).astype(np.uint8)

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=FakeNormalizer(),
        normalization_method="MACENKO",
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert int(handle["images"][0, 0, 0, 0]) == 10


def test_write_split_hdf5_carries_stage4_cleaning_lineage_attrs(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([20], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_20_PATCH_001.png"]))
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/accepted_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "abc123"

    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 0,
            }
        ]
    )

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["stage4_cleaning_manifest_path"] == "/tmp/accepted_manifest.csv"
        assert handle.attrs["stage4_cleaning_manifest_sha256"] == "abc123"


def test_write_split_hdf5_respects_none_compression_and_cached_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 20], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_20_PATCH_001.png"]),
        )

    split_df = pd.DataFrame(
        [
            {
                "label": 0,
                "patient_id": 10,
                "filename": "PATIENT_10_PATCH_001.png",
                "source_row_index": 0,
            },
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 1,
            },
        ]
    )

    def fail_collect(_: Path) -> dict[str, Any]:
        raise AssertionError(
            "collect_hdf5_provenance should not be called when provenance is cached"
        )

    monkeypatch.setattr("helpers.crossfold.io.collect_hdf5_provenance", fail_collect)

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        source_hdf5_provenance={"path": str(source_path), "sha256": "cached-hash", "attrs": {}},
        hdf5_compression="NONE",
        copy_batch_size=2,
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["images"].compression is None
        assert handle.attrs["source_hdf5_sha256"] == "cached-hash"


def test_verify_split_hdf5_integrity_rejects_metadata_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([20], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_20_PATCH_001.png"]))

    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 0,
            }
        ]
    )

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        overwrite=True,
    )

    with h5py.File(output_path, "a") as handle:
        handle["patient_ids"][0] = 99

    with pytest.raises(ValueError, match="row metadata does not match"):
        verify_split_hdf5_integrity(output_path, split_df)


def test_write_split_hdf5_emits_progress_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 20], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_20_PATCH_001.png"]),
        )

    split_df = pd.DataFrame(
        [
            {
                "label": 0,
                "patient_id": 10,
                "filename": "PATIENT_10_PATCH_001.png",
                "source_row_index": 0,
            },
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 1,
            },
        ]
    )

    caplog.set_level("INFO")
    write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        hdf5_compression="NONE",
        copy_batch_size=1,
        overwrite=True,
    )

    assert any("Stage 5 split write" in message for message in caplog.messages)
    assert any("Stage 5 split verify" in message for message in caplog.messages)


def test_write_split_hdf5_reads_rows_from_multiple_stage2_sources(tmp_path: Path) -> None:
    source_a = tmp_path / "PATCHES" / "HDF5_SHARDS" / "a.h5"
    source_b = tmp_path / "PATCHES" / "HDF5_SHARDS" / "b.h5"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_a, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"A.png"]))
    with h5py.File(source_b, "w") as handle:
        handle.create_dataset("images", data=np.ones((1, 4, 4, 3), dtype=np.uint8) * 7)
        handle.create_dataset("masks", data=np.ones((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([20], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"B.png"]))

    split_df = pd.DataFrame(
        [
            {
                "label": 0,
                "patient_id": 10,
                "filename": "A.png",
                "source_hdf5_path": str(source_a),
                "source_row_index": 0,
            },
            {
                "label": 1,
                "patient_id": 20,
                "filename": "B.png",
                "source_hdf5_path": str(source_b),
                "source_row_index": 0,
            },
        ]
    )

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=tmp_path / "master_manifest.sqlite",
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        source_hdf5_provenance={
            "path": str(tmp_path / "master_manifest.sqlite"),
            "sha256": "sqlite-sha",
            "attrs": {"stage4_cleaning_manifest_path": str(tmp_path / "master_manifest.sqlite")},
        },
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["labels"][:].tolist() == [0, 1]
        assert int(handle["images"][1, 0, 0, 0]) == 7
        assert handle.attrs["source_hdf5_sha256"] == "sqlite-sha"
