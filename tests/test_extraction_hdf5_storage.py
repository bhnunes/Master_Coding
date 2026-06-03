from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers.extraction.hdf5_storage import write_slide_patch_dataset_hdf5

IMAGE_PIXEL_VALUE = 77


def test_write_slide_patch_dataset_hdf5_writes_canonical_stage2_shard(tmp_path: Path) -> None:
    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_a.h5"
    write_slide_patch_dataset_hdf5(
        output_path=output_path,
        records=[
            {
                "filename": "cancer.png",
                "label": 1,
                "patient_id": "1001",
                "slide_id": "slide_a",
                "_image_array": np.full((4, 4, 3), 120, dtype=np.uint8),
                "_mask_array": np.ones((4, 4), dtype=np.uint8),
            },
            {
                "filename": "not_cancer.png",
                "label": 0,
                "patient_id": "1001",
                "slide_id": "slide_a",
                "_image_array": np.full((4, 4, 3), 10, dtype=np.uint8),
                "_mask_array": np.zeros((4, 4), dtype=np.uint8),
            },
        ],
    )

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {
            "images",
            "masks",
            "labels",
            "patient_ids",
            "filenames",
            "slide_ids",
            "source_image_paths",
            "source_mask_paths",
        }
        assert handle["labels"][:].tolist() == [1, 0]
        assert handle["patient_ids"][:].tolist() == [1001, 1001]
        assert handle["slide_ids"][:].tolist() == [b"slide_a", b"slide_a"]
        assert handle["filenames"][:].tolist() == [b"cancer.png", b"not_cancer.png"]
        assert handle["source_image_paths"][:].tolist() == [
            f"{output_path}::images[0]".encode(),
            f"{output_path}::images[1]".encode(),
        ]
        assert handle["source_mask_paths"][:].tolist() == [
            f"{output_path}::masks[0]".encode(),
            f"{output_path}::masks[1]".encode(),
        ]
        assert set(np.unique(handle["masks"][0]).tolist()) <= {0, 1}
        assert handle.attrs["source_signature"]


def test_write_slide_patch_dataset_hdf5_removes_stale_shard_when_no_records(tmp_path: Path) -> None:
    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_a.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"stale")

    result = write_slide_patch_dataset_hdf5(
        output_path=output_path,
        records=[],
    )

    assert result is None
    assert not output_path.exists()


def test_write_slide_patch_dataset_hdf5_supports_in_memory_patch_records(tmp_path: Path) -> None:
    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_b.h5"

    write_slide_patch_dataset_hdf5(
        output_path=output_path,
        records=[
            {
                "filename": "cancer.png",
                "label": 1,
                "patient_id": "2002",
                "slide_id": "slide_b",
                "_image_array": np.full((4, 4, 3), 77, dtype=np.uint8),
                "_mask_array": np.ones((4, 4), dtype=np.uint8),
            }
        ],
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [2002]
        assert int(handle["images"][0, 0, 0, 0]) == IMAGE_PIXEL_VALUE
        assert set(np.unique(handle["masks"][0]).tolist()) == {1}


def test_write_slide_patch_dataset_hdf5_signature_is_stable_across_batch_sizes(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_batched.h5"
    records = [
        {
            "filename": f"patch_{index}.png",
            "label": index % 2,
            "patient_id": "2002",
            "slide_id": "slide_batched",
            "_image_array": np.full((4, 4, 3), index, dtype=np.uint8),
            "_mask_array": np.full((4, 4), index % 2, dtype=np.uint8),
        }
        for index in range(5)
    ]

    write_slide_patch_dataset_hdf5(output_path=output_path, records=records, batch_size=1)
    with h5py.File(output_path, "r") as handle:
        single_row_signature = str(handle.attrs["source_signature"])

    write_slide_patch_dataset_hdf5(output_path=output_path, records=records, batch_size=3)
    with h5py.File(output_path, "r") as handle:
        batched_signature = str(handle.attrs["source_signature"])
        assert handle["labels"][:].tolist() == [0, 1, 0, 1, 0]

    assert batched_signature == single_row_signature


def test_write_slide_patch_dataset_hdf5_rejects_records_without_in_memory_arrays(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="in-memory '_image_array' and '_mask_array'"):
        write_slide_patch_dataset_hdf5(
            output_path=tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_c.h5",
            records=[
                {
                    "filename": "missing.png",
                    "label": 1,
                    "patient_id": "3003",
                    "slide_id": "slide_c",
                }
            ],
        )
