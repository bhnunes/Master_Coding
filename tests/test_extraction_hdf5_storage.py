from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from helpers.extraction.hdf5_storage import write_slide_patch_dataset_hdf5


def _write_rgb_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def _write_mask_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(path)


def test_write_slide_patch_dataset_hdf5_writes_canonical_stage2_shard(tmp_path: Path) -> None:
    cancer_dir = tmp_path / "PATCHES" / "CANCER"
    cancer_mask_dir = tmp_path / "PATCHES" / "CANCER_MASK"
    not_cancer_dir = tmp_path / "PATCHES" / "NOT_CANCER"
    not_cancer_mask_dir = tmp_path / "PATCHES" / "NOT_CANCER_MASK"
    _write_rgb_png(cancer_dir / "cancer.png", 120)
    _write_mask_png(cancer_mask_dir / "cancer.png", 255)
    _write_rgb_png(not_cancer_dir / "not_cancer.png", 10)
    _write_mask_png(not_cancer_mask_dir / "not_cancer.png", 0)

    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_a.h5"
    write_slide_patch_dataset_hdf5(
        output_path=output_path,
        records=[
            {
                "filename": "cancer.png",
                "label": 1,
                "patient_id": "1001",
                "slide_id": "slide_a",
            },
            {
                "filename": "not_cancer.png",
                "label": 0,
                "patient_id": "1001",
                "slide_id": "slide_a",
            },
        ],
        cancer_folder=cancer_dir,
        not_cancer_folder=not_cancer_dir,
        cancer_mask_folder=cancer_mask_dir,
        not_cancer_mask_folder=not_cancer_mask_dir,
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
        cancer_folder=tmp_path / "PATCHES" / "CANCER",
        not_cancer_folder=tmp_path / "PATCHES" / "NOT_CANCER",
        cancer_mask_folder=tmp_path / "PATCHES" / "CANCER_MASK",
        not_cancer_mask_folder=tmp_path / "PATCHES" / "NOT_CANCER_MASK",
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
        cancer_folder=tmp_path / "PATCHES" / "CANCER",
        not_cancer_folder=tmp_path / "PATCHES" / "NOT_CANCER",
        cancer_mask_folder=tmp_path / "PATCHES" / "CANCER_MASK",
        not_cancer_mask_folder=tmp_path / "PATCHES" / "NOT_CANCER_MASK",
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [2002]
        assert int(handle["images"][0, 0, 0, 0]) == 77
        assert set(np.unique(handle["masks"][0]).tolist()) == {1}
