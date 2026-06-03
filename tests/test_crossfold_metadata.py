import json
from pathlib import Path

import h5py
import pandas as pd

from helpers.crossfold.provenance import (
    ManifestWriteConfig,
    build_hdf5_manifest_from_split_dfs,
    write_manifest_and_log_stats,
)

OBJECTIVE_SCORE = 0.125


def _split_data() -> dict[str, object]:
    train_df = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": "/src/p1.png",
                "mask_path": "/src/p1_mask.png",
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
            },
            {
                "patient_id": 2,
                "image_path": "/src/p2.png",
                "mask_path": "/src/p2_mask.png",
                "label": 0,
                "filename": "PATIENT_2_PATCH_001.png",
            },
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                "patient_id": 3,
                "image_path": "/src/p3.png",
                "mask_path": "/src/p3_mask.png",
                "label": 1,
                "filename": "PATIENT_3_PATCH_001.png",
            },
            {
                "patient_id": 4,
                "image_path": "/src/p4.png",
                "mask_path": "/src/p4_mask.png",
                "label": 0,
                "filename": "PATIENT_4_PATCH_001.png",
            },
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "patient_id": 5,
                "image_path": "/src/p5.png",
                "mask_path": "/src/p5_mask.png",
                "label": 1,
                "filename": "PATIENT_5_PATCH_001.png",
            },
            {
                "patient_id": 6,
                "image_path": "/src/p6.png",
                "mask_path": "/src/p6_mask.png",
                "label": 0,
                "filename": "PATIENT_6_PATCH_001.png",
            },
        ]
    )
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "constraints": {
            "random_state": 42,
            "global_cancer_ratio": 0.5,
            "optuna_trials": 25,
        },
        "train_patients": [1, 2],
        "val_patients": [3, 4],
        "test_patients": [5, 6],
        "split_seed": 42,
        "split_attempt": 1,
        "objective_score": 0.125,
    }


def test_run_config_records_new_stage5_split_metadata(tmp_path: Path) -> None:
    split_data = _split_data()
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=split_data,
    )
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=[[[[0, 0, 0]]]], dtype="uint8")
        handle.create_dataset("masks", data=[[[0]]], dtype="uint8")
        handle.create_dataset("labels", data=[0], dtype="uint8")
        handle.create_dataset("patient_ids", data=[1], dtype="int32")
        handle.create_dataset("filenames", data=[b"PATIENT_1_PATCH_001.png"])

    verification = {
        "TRAIN": {
            "expected_cancer_samples": 1.0,
            "expected_non_cancer_samples": 1.0,
            "chi2_stat": 0.0,
        },
        "VALIDATION": {
            "expected_cancer_samples": 1.0,
            "expected_non_cancer_samples": 1.0,
            "chi2_stat": 0.0,
        },
        "TEST": {
            "expected_cancer_samples": 1.0,
            "expected_non_cancer_samples": 1.0,
            "chi2_stat": 0.0,
        },
    }
    extra = {
        "split_selection": {
            "method": "optuna_greedy_sample_ratio_stratified",
            "tie_break_priority": ["TEST", "VALIDATION", "TRAIN"],
            "global_cancer_ratio": 0.5,
            "optuna_trials": 25,
            "loss_metric": "sum_absolute_split_ratio_delta",
            "final_loss": 0.125,
        },
        "verification": verification,
    }

    write_manifest_and_log_stats(
        ManifestWriteConfig(
            output_dir=tmp_path,
            run_id="run-1",
            normalization_method="NOT_NORMALIZED",
            is_normalized=False,
            source_hdf5_path=source_path,
            split_data=split_data,
            manifest_df=manifest_df,
            calc_checksums=False,
            extra=extra,
        )
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert "optimize_training_set" not in payload
    assert payload["objective_score"] == OBJECTIVE_SCORE
    assert "objective_score_split" not in payload
    assert payload["extra"]["split_selection"] == extra["split_selection"]
    assert payload["extra"]["verification"] == verification
