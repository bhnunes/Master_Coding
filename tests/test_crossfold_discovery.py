from pathlib import Path

import h5py
import numpy as np

from helpers.crossfold.discovery import load_patch_dataset


def test_load_patch_dataset_reads_hdf5_source_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
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
            data=np.array([b"/src/p11.png", b"/src/p22.png"]),
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array([b"/src/p11_mask.png", b"/src/p22_mask.png"]),
        )

    dataset = load_patch_dataset(source_path)

    assert dataset[["patient_id", "label", "filename", "source_row_index"]].to_dict("records") == [
        {
            "patient_id": 11,
            "label": 1,
            "filename": "PATIENT_11_PATCH_001.png",
            "source_row_index": 0,
        },
        {
            "patient_id": 22,
            "label": 0,
            "filename": "PATIENT_22_PATCH_001.png",
            "source_row_index": 1,
        },
    ]


def test_load_patch_dataset_requires_source_paths_in_hdf5_dataset(tmp_path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_11_PATCH_001.png"]))

    try:
        load_patch_dataset(source_path)
    except ValueError as error:
        assert "source_image_paths" in str(error)
    else:
        raise AssertionError("Expected HDF5 dataset without source paths to be rejected.")
