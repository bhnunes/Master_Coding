from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from helpers.sanity.disk_checks import IndexedInspection
from helpers.sanity.semantic_checks import check_mask_label_semantics, check_split_class_presence


def _manifest_row(
    source_path: Path,
    *,
    label: int,
    filename: str,
) -> dict[str, object]:
    return {
        "label": label,
        "filename": filename,
        "source_hdf5_path": str(source_path),
        "source_row_index": 0,
    }


def test_check_mask_label_semantics_reads_hdf5_masks(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        mask[0, 1:3, 1:3] = 1
        handle.create_dataset("masks", data=mask)
        handle.create_dataset("labels", data=np.array([0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))

    manifest_df = pd.DataFrame(
        [_manifest_row(split_path, label=0, filename="PATIENT_1_PATCH_001.png")]
    )

    result = check_mask_label_semantics(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "not_cancer" in result.details.lower()


def test_check_mask_label_semantics_fails_for_empty_cancer_hdf5_mask(tmp_path: Path) -> None:
    split_path = tmp_path / "TRAIN.h5"
    with h5py.File(split_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_2_PATCH_001.png"]))

    manifest_df = pd.DataFrame(
        [_manifest_row(split_path, label=1, filename="PATIENT_2_PATCH_001.png")]
    )

    result = check_mask_label_semantics(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "cancer" in result.details.lower()


def test_check_mask_label_semantics_accepts_precomputed_row_inspections(tmp_path: Path) -> None:
    manifest_df = pd.DataFrame(
        [_manifest_row(tmp_path / "missing.h5", label=0, filename="PATIENT_1_PATCH_001.png")]
    )

    result = check_mask_label_semantics(
        manifest_df,
        tmp_path,
        "TRAIN",
        row_inspections={
            0: IndexedInspection(
                manifest_index=0,
                filename="PATIENT_1_PATCH_001.png",
                inspection={
                    "image_shape": (4, 4),
                    "mask_shape": (4, 4),
                    "mask_unique_values": (0,),
                    "mask_has_positive_pixels": False,
                },
            )
        },
    )

    assert result.status == "PASS"


def test_check_split_class_presence_fails_for_one_class_split() -> None:
    manifest_df = pd.DataFrame(
        [
            {"label": 0, "filename": "negative_1.png"},
            {"label": 0, "filename": "negative_2.png"},
        ]
    )

    result = check_split_class_presence(manifest_df)

    assert result.status == "FAIL"
    assert "Only one class" in result.details


def test_check_split_class_presence_passes_when_both_classes_exist() -> None:
    manifest_df = pd.DataFrame(
        [
            {"label": 0, "filename": "negative_1.png"},
            {"label": 1, "filename": "positive_1.png"},
        ]
    )

    result = check_split_class_presence(manifest_df)

    assert result.status == "PASS"
