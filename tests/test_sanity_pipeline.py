import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from helpers.extraction.manifest_paths import build_hdf5_dataset_ref
from helpers.sanity.config import SanityConfig
from helpers.sanity.pipeline import run_sanity_pipeline


def test_run_sanity_pipeline_rejects_duplicate_filenames(tmp_path: Path) -> None:
    output_dir = tmp_path
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "TEST",
                "label": 1,
                "patient_id": 2,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TEST.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
        ]
    )
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "split_stats.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")
    for split_name in ("TRAIN", "TEST"):
        with h5py.File(output_dir / f"{split_name}.h5", "w") as handle:
            handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
            handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
            handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
            handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))

    report = run_sanity_pipeline(
        SanityConfig(
            base_dir=output_dir,
            sample_pairs=10,
            full_mask_scan=False,
            full_shape_scan=False,
            checksum_mode="OFF",
            enforce_filename_uniqueness=True,
            enforce_regex_patient_id_match=True,
            enforce_split_stats_parity=False,
            fail_on_empty_cancer_mask=True,
            fail_on_positive_not_cancer_mask=True,
        )
    )

    assert report.verdict == "SPLITS REJECTED"
    assert report.dataset_checks["Filename Uniqueness"].status == "FAIL"


def test_run_sanity_pipeline_checks_logical_source_refs(tmp_path: Path) -> None:
    output_dir = tmp_path
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
                "source_image_path": build_hdf5_dataset_ref("images", 0),
                "source_mask_path": build_hdf5_dataset_ref("masks", 1),
            }
        ]
    )
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "n_patients": 1,
                "n_pos_patients": 1,
                "n_neg_patients": 0,
                "n_images": 1,
                "patches_per_patient_mean": 1.0,
                "patches_per_patient_std": 0.0,
                "patches_per_patient_min": 1,
                "patches_per_patient_q1": 1.0,
                "patches_per_patient_median": 1.0,
                "patches_per_patient_q3": 1.0,
                "patches_per_patient_max": 1,
                "n_images_neg": 0,
                "n_images_pos": 1,
            }
        ]
    ).to_csv(output_dir / "split_stats.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")
    with h5py.File(output_dir / "TRAIN.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.ones((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))

    report = run_sanity_pipeline(
        SanityConfig(
            base_dir=output_dir,
            sample_pairs=10,
            full_mask_scan=False,
            full_shape_scan=False,
            checksum_mode="OFF",
            enforce_filename_uniqueness=True,
            enforce_regex_patient_id_match=True,
            enforce_split_stats_parity=True,
            fail_on_empty_cancer_mask=True,
            fail_on_positive_not_cancer_mask=True,
        )
    )

    assert report.dataset_checks["Source Reference Contract"].status == "FAIL"


def test_run_sanity_pipeline_fails_on_stage4_cleaning_lineage_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            }
        ]
    )
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "split_stats.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_hdf5_provenance": {
                    "attrs": {
                        "stage4_cleaning_manifest_path": "/tmp/accepted_manifest.csv",
                        "stage4_cleaning_manifest_sha256": "abc123",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(output_dir / "TRAIN.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/other_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "other-sha"

    report = run_sanity_pipeline(
        SanityConfig(
            base_dir=output_dir,
            sample_pairs=10,
            full_mask_scan=False,
            full_shape_scan=False,
            checksum_mode="OFF",
            enforce_filename_uniqueness=True,
            enforce_regex_patient_id_match=True,
            enforce_split_stats_parity=False,
            fail_on_empty_cancer_mask=True,
            fail_on_positive_not_cancer_mask=True,
        )
    )

    assert report.dataset_checks["Stage4 Cleaning Lineage"].status == "FAIL"


def test_run_sanity_pipeline_rejects_non_singleton_stage5_layout(tmp_path: Path) -> None:
    output_dir = tmp_path
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_hdf5_path": "TRAIN_shards/1.h5",
                "hdf5_row_index": 0,
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            }
        ]
    )
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "n_patients": 1,
                "n_pos_patients": 1,
                "n_neg_patients": 0,
                "n_images": 1,
                "patches_per_patient_mean": 1.0,
                "patches_per_patient_std": 0.0,
                "patches_per_patient_min": 1,
                "patches_per_patient_q1": 1.0,
                "patches_per_patient_median": 1.0,
                "patches_per_patient_q3": 1.0,
                "patches_per_patient_max": 1,
                "n_images_neg": 0,
                "n_images_pos": 1,
            }
        ]
    ).to_csv(output_dir / "split_stats.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")
    with h5py.File(output_dir / "TRAIN.h5", "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.ones((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_1_PATCH_001.png"]))

    report = run_sanity_pipeline(
        SanityConfig(
            base_dir=output_dir,
            sample_pairs=10,
            full_mask_scan=False,
            full_shape_scan=False,
            checksum_mode="OFF",
            enforce_filename_uniqueness=True,
            enforce_regex_patient_id_match=True,
            enforce_split_stats_parity=True,
            fail_on_empty_cancer_mask=True,
            fail_on_positive_not_cancer_mask=True,
        )
    )

    assert report.verdict == "SPLITS REJECTED"
    assert report.dataset_checks["Stage 5 Singleton Layout"].status == "FAIL"
