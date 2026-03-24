from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from helpers.sanity.disk_checks import check_manifest_disk_parity


def test_check_manifest_disk_parity_fails_when_hdf5_row_metadata_differs(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_002.png"]))

    manifest_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
            }
        ]
    )

    result = check_manifest_disk_parity(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "filename mismatch" in result.details.lower()


def test_check_manifest_disk_parity_uses_hdf5_manifest_contract(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))

    manifest_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
            }
        ]
    )

    result = check_manifest_disk_parity(manifest_df, tmp_path, "TRAIN")

    assert result.status == "PASS"
    assert "hdf5" in result.details.lower()
