from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from helpers.sanity.manifest_checks import (
    check_manifest_schema,
    check_split_constraints_from_run_config,
    check_split_stats_against_manifest,
    check_stage4_cleaning_lineage,
)


def test_check_split_stats_against_manifest_fails_when_counts_drift() -> None:
    manifest_df = pd.DataFrame(
        [
            {"run_id": "run-1", "split": "TRAIN", "label": 1, "patient_id": 1},
            {"run_id": "run-1", "split": "TRAIN", "label": 0, "patient_id": 2},
        ]
    )
    split_stats_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "n_patients": 2,
                "n_pos_patients": 1,
                "n_neg_patients": 1,
                "n_images": 3,
                "patches_per_patient_mean": 1.0,
                "patches_per_patient_std": 0.0,
                "patches_per_patient_min": 1,
                "patches_per_patient_q1": 1.0,
                "patches_per_patient_median": 1.0,
                "patches_per_patient_q3": 1.0,
                "patches_per_patient_max": 1,
                "n_images_neg": 1,
                "n_images_pos": 1,
            }
        ]
    )

    result = check_split_stats_against_manifest(manifest_df, split_stats_df)

    assert result.status == "FAIL"
    assert "n_images" in result.details


def test_check_manifest_schema_accepts_hdf5_native_manifest_columns() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
                "source_hdf5_path": "/tmp/TRAIN.h5",
                "source_row_index": 0,
                "source_image_path": "/tmp/TRAIN.h5::images[0]",
                "source_mask_path": "/tmp/TRAIN.h5::masks[0]",
            }
        ]
    )

    result = check_manifest_schema(manifest_df)

    assert result.status == "PASS"


def test_check_split_constraints_from_run_config_accepts_new_stage5_constraint_keys() -> None:
    manifest_df = pd.DataFrame(
        [
            {"split": "TRAIN", "patient_id": 1},
            {"split": "VALIDATION", "patient_id": 2},
            {"split": "VALIDATION", "patient_id": 3},
            {"split": "TEST", "patient_id": 4},
            {"split": "TEST", "patient_id": 5},
        ]
    )

    result = check_split_constraints_from_run_config(
        manifest_df,
        {
            "constraints": {
                "test_patient_count": 2,
                "validation_patient_count": 2,
            }
        },
    )

    assert result.status == "PASS"


def test_check_split_constraints_from_run_config_fails_for_new_stage5_constraint_violation() -> (
    None
):
    manifest_df = pd.DataFrame(
        [
            {"split": "TRAIN", "patient_id": 1},
            {"split": "VALIDATION", "patient_id": 2},
            {"split": "TEST", "patient_id": 3},
        ]
    )

    result = check_split_constraints_from_run_config(
        manifest_df,
        {
            "constraints": {
                "test_patient_count": 2,
                "validation_patient_count": 2,
            }
        },
    )

    assert result.status == "FAIL"
    assert "VALIDATION patients 1 < 2" in result.details
    assert "TEST patients 1 < 2" in result.details


def test_check_stage4_cleaning_lineage_fails_on_split_attr_mismatch(tmp_path: Path) -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
                "source_hdf5_path": str(tmp_path / "TRAIN.h5"),
                "source_row_index": 0,
                "source_image_path": f"{tmp_path / 'TRAIN.h5'}::images[0]",
                "source_mask_path": f"{tmp_path / 'TRAIN.h5'}::masks[0]",
            }
        ]
    )
    with h5py.File(tmp_path / "TRAIN.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/other_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "other-sha"

    result = check_stage4_cleaning_lineage(
        manifest_df,
        {
            "source_hdf5_provenance": {
                "attrs": {
                    "stage4_cleaning_manifest_path": "/tmp/accepted_manifest.csv",
                    "stage4_cleaning_manifest_sha256": "abc123",
                }
            }
        },
        tmp_path,
    )

    assert result.status == "FAIL"


def test_check_stage4_cleaning_lineage_passes_when_split_attrs_match(tmp_path: Path) -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
                "source_hdf5_path": str(tmp_path / "TRAIN.h5"),
                "source_row_index": 0,
                "source_image_path": f"{tmp_path / 'TRAIN.h5'}::images[0]",
                "source_mask_path": f"{tmp_path / 'TRAIN.h5'}::masks[0]",
            }
        ]
    )
    with h5py.File(tmp_path / "TRAIN.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/accepted_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "abc123"

    result = check_stage4_cleaning_lineage(
        manifest_df,
        {
            "source_hdf5_provenance": {
                "attrs": {
                    "stage4_cleaning_manifest_path": "/tmp/accepted_manifest.csv",
                    "stage4_cleaning_manifest_sha256": "abc123",
                }
            }
        },
        tmp_path,
    )

    assert result.status == "PASS"
