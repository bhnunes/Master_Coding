import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers.packaging.writer import (
    copy_source_hdf5_dataset,
    filter_source_hdf5_by_manifest,
    merge_source_hdf5_shards,
)


def test_copy_source_hdf5_dataset_preserves_contract_and_records_upstream_signature(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "stage2_source.h5"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_7_PATCH_001.png"]))
        handle.create_dataset("source_image_paths", data=np.array([b"/src/p7.png"]))
        handle.create_dataset("source_mask_paths", data=np.array([b"/src/p7_mask.png"]))
        handle.attrs["source_signature"] = "stage2-signature"

    copy_source_hdf5_dataset(source_path, output_path, overwrite=True)

    with h5py.File(output_path, "r") as handle:
        assert handle["labels"][:].tolist() == [1]
        assert handle["patient_ids"][:].tolist() == [7]
        assert handle["filenames"][:].tolist() == [b"PATIENT_7_PATCH_001.png"]
        assert handle.attrs["upstream_source_signature"] == "stage2-signature"


def test_copy_source_hdf5_dataset_allows_missing_optional_source_ref_datasets(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "stage2_source.h5"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_7_PATCH_001.png"]))

    copy_source_hdf5_dataset(source_path, output_path, overwrite=True)

    with h5py.File(output_path, "r") as handle:
        assert "source_image_paths" not in handle
        assert "source_mask_paths" not in handle


def test_copy_source_hdf5_dataset_rejects_missing_core_datasets(tmp_path: Path) -> None:
    source_path = tmp_path / "stage2_source.h5"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))

    with pytest.raises(ValueError, match="missing required datasets"):
        copy_source_hdf5_dataset(source_path, output_path, overwrite=True)


def test_merge_source_hdf5_shards_writes_one_canonical_dataset(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()
    for index, patient_id in enumerate((7, 8), start=1):
        with h5py.File(shard_dir / f"slide_{index}.h5", "w") as handle:
            handle.create_dataset("images", data=np.full((1, 4, 4, 3), index, dtype=np.uint8))
            handle.create_dataset("masks", data=np.full((1, 4, 4), index % 2, dtype=np.uint8))
            handle.create_dataset("labels", data=np.array([index % 2], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id], dtype=np.int32))
            handle.create_dataset(
                "filenames",
                data=np.array([f"PATIENT_{patient_id}_PATCH_001.png".encode()]),
            )
            handle.create_dataset(
                "source_image_paths",
                data=np.array([f"{shard_dir / f'slide_{index}.h5'}::images[0]".encode()]),
            )
            handle.create_dataset(
                "source_mask_paths",
                data=np.array([f"{shard_dir / f'slide_{index}.h5'}::masks[0]".encode()]),
            )
            handle.create_dataset(
                "slide_ids",
                data=np.array([f"slide_{index}".encode()]),
            )

    output_path = tmp_path / "SOURCE_DATASET.h5"
    merge_source_hdf5_shards(
        shard_dir,
        output_path,
        overwrite=True,
        compression=None,
        copy_batch_size=256,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [7, 8]
        assert handle["filenames"][:].tolist() == [
            b"PATIENT_7_PATCH_001.png",
            b"PATIENT_8_PATCH_001.png",
        ]


def test_merge_source_hdf5_shards_rejects_empty_directory(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()

    with pytest.raises(ValueError, match="No HDF5 shards found"):
        merge_source_hdf5_shards(
            shard_dir,
            tmp_path / "SOURCE_DATASET.h5",
            overwrite=True,
            compression=None,
            copy_batch_size=256,
        )


def test_merge_source_hdf5_shards_rejects_manifest_signature_mismatch(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()
    shard_path = shard_dir / "slide_1.h5"
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_7_PATCH_001.png"]))
        handle.attrs["source_signature"] = "actual-sig"

    (shard_dir / "manifest.json").write_text(
        '{"version":1,"shards":[{"relative_path":"slide_1.h5","row_count":1,'
        '"source_signature":"stale-sig"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest signature mismatch"):
        merge_source_hdf5_shards(
            shard_dir,
            tmp_path / "SOURCE_DATASET.h5",
            overwrite=True,
            compression=None,
            copy_batch_size=256,
        )


def test_merge_source_hdf5_shards_logs_progress(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()
    for index, patient_id in enumerate((7, 8), start=1):
        with h5py.File(shard_dir / f"slide_{index}.h5", "w") as handle:
            handle.create_dataset("images", data=np.full((1, 4, 4, 3), index, dtype=np.uint8))
            handle.create_dataset("masks", data=np.full((1, 4, 4), index % 2, dtype=np.uint8))
            handle.create_dataset("labels", data=np.array([index % 2], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id], dtype=np.int32))
            handle.create_dataset(
                "filenames",
                data=np.array([f"PATIENT_{patient_id}_PATCH_001.png".encode()]),
            )

    output_path = tmp_path / "SOURCE_DATASET.h5"
    caplog.set_level(logging.INFO)

    merge_source_hdf5_shards(
        shard_dir,
        output_path,
        overwrite=True,
        compression=None,
        copy_batch_size=256,
    )

    assert "Merging 2 HDF5 shard(s)" in caplog.text
    assert "Read Stage 2 shards: 2/2 (100.0%)" in caplog.text
    assert "Wrote merged rows: 2/2 (100.0%)" in caplog.text


def test_filter_source_hdf5_by_manifest_writes_only_accepted_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_11_PATCH_001.png", b"PATIENT_22_PATCH_001.png"]),
        )
        handle.create_dataset(
            "source_image_paths",
            data=np.array(
                [f"{source_path}::images[0]".encode(), f"{source_path}::images[1]".encode()]
            ),
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array(
                [f"{source_path}::masks[0]".encode(), f"{source_path}::masks[1]".encode()]
            ),
        )
        handle.create_dataset("slide_ids", data=np.array([b"slide_a", b"slide_b"]))
        handle.attrs["source_signature"] = "stage2-signature"

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        f"PATIENT_22_PATCH_001.png,accepted,0.1,22,slide_b,{source_path},1\n",
        encoding="utf-8",
    )

    filter_source_hdf5_by_manifest(
        source_path,
        manifest_path,
        output_path,
        overwrite=True,
        compression=None,
        copy_batch_size=256,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [22]
        assert handle["filenames"][:].tolist() == [b"PATIENT_22_PATCH_001.png"]
        assert handle["slide_ids"][:].tolist() == [b"slide_b"]
        assert handle.attrs["upstream_source_signature"] == "stage2-signature"
        assert handle.attrs["stage4_cleaning_manifest_path"] == str(manifest_path)


def test_filter_source_hdf5_by_manifest_logs_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_11_PATCH_001.png", b"PATIENT_22_PATCH_001.png"]),
        )

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        f"PATIENT_11_PATCH_001.png,accepted,0.1,11,slide_a,{source_path},0\n"
        f"PATIENT_22_PATCH_001.png,accepted,0.1,22,slide_b,{source_path},1\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.INFO)

    filter_source_hdf5_by_manifest(
        source_path,
        manifest_path,
        output_path,
        overwrite=True,
        compression=None,
        copy_batch_size=256,
    )

    assert "Filtering accepted rows: 2/2 (100.0%)" in caplog.text


def test_filter_source_hdf5_by_manifest_preserves_manifest_row_order(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    output_path = tmp_path / "packaged" / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((3, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22, 33], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array(
                [
                    b"PATIENT_11_PATCH_001.png",
                    b"PATIENT_22_PATCH_001.png",
                    b"PATIENT_33_PATCH_001.png",
                ]
            ),
        )
        handle.create_dataset("slide_ids", data=np.array([b"slide_a", b"slide_b", b"slide_c"]))
        handle.attrs["source_signature"] = "stage2-signature"

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        f"PATIENT_33_PATCH_001.png,accepted,0.1,33,slide_c,{source_path},2\n"
        f"PATIENT_11_PATCH_001.png,accepted,0.1,11,slide_a,{source_path},0\n",
        encoding="utf-8",
    )

    filter_source_hdf5_by_manifest(
        source_path,
        manifest_path,
        output_path,
        overwrite=True,
        compression=None,
        copy_batch_size=256,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [33, 11]
        assert handle["filenames"][:].tolist() == [
            b"PATIENT_33_PATCH_001.png",
            b"PATIENT_11_PATCH_001.png",
        ]
        assert handle["source_row_indices"][:].tolist() == [2, 0]


def test_filter_source_hdf5_by_manifest_rejects_row_from_other_source(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_11_PATCH_001.png"]))

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        "PATIENT_11_PATCH_001.png,accepted,0.1,11,slide_a,/other/source.h5,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match source_hdf5_path"):
        filter_source_hdf5_by_manifest(
            source_path,
            manifest_path,
            tmp_path / "packaged" / "SOURCE_DATASET.h5",
            overwrite=True,
            compression=None,
            copy_batch_size=256,
        )
