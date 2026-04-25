from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from helpers.sanity.disk_checks import (
    IndexedInspection,
    check_decode_and_shapes,
    check_manifest_disk_parity,
    check_mask_pixel_values,
    collect_hdf5_row_inspections,
)


def _manifest_row(
    source_path: Path,
    *,
    row_index: int = 0,
    label: int = 1,
    patient_id: int = 1,
    filename: str = "PATIENT_1_PATCH_001.png",
) -> dict[str, object]:
    return {
        "label": label,
        "patient_id": patient_id,
        "filename": filename,
        "source_hdf5_path": str(source_path),
        "source_row_index": row_index,
    }


def test_check_manifest_disk_parity_fails_when_hdf5_row_metadata_differs(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_002.png"]))

    manifest_df = pd.DataFrame([_manifest_row(split_path)])

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

    manifest_df = pd.DataFrame([_manifest_row(split_path)])

    result = check_manifest_disk_parity(manifest_df, tmp_path, "TRAIN")

    assert result.status == "PASS"
    assert "hdf5" in result.details.lower()


def test_collect_hdf5_row_inspections_handles_out_of_order_rows(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    images = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    masks = np.zeros((3, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[2, 1, 1] = 1
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=np.array([0, 1, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 1, 2], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"a.png", b"b.png", b"c.png"]),
        )

    manifest_df = pd.DataFrame(
        [
            _manifest_row(split_path, row_index=2, label=1, patient_id=2, filename="c.png"),
            _manifest_row(split_path, row_index=0, label=0, patient_id=1, filename="a.png"),
        ]
    )

    inspections = collect_hdf5_row_inspections(manifest_df, tmp_path, "TRAIN")

    assert inspections[0].inspection["mask_has_positive_pixels"] is True
    assert inspections[1].inspection["mask_has_positive_pixels"] is True
    assert inspections[0].inspection["mask_unique_values"] == (0, 1)
    assert inspections[1].inspection["image_shape"] == (4, 4)


def test_disk_checks_accept_precomputed_row_inspections(tmp_path: Path) -> None:
    manifest_df = pd.DataFrame(
        [
            _manifest_row(tmp_path / "missing.h5")
        ]
    )
    inspections = {
        0: IndexedInspection(
            manifest_index=0,
            filename="PATIENT_1_PATCH_001.png",
            inspection={
                "image_shape": (4, 4),
                "mask_shape": (4, 4),
                "mask_unique_values": (0, 1),
                "mask_has_positive_pixels": True,
            },
        )
    }

    decode_result = check_decode_and_shapes(
        manifest_df,
        tmp_path,
        "TRAIN",
        sample_n=1,
        row_inspections=inspections,
    )
    mask_result = check_mask_pixel_values(
        manifest_df,
        tmp_path,
        "TRAIN",
        sample_n=1,
        row_inspections=inspections,
    )

    assert decode_result.status == "PASS"
    assert mask_result.status == "PASS"
