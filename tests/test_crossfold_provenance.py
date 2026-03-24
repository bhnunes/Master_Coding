import json
from pathlib import Path

import pandas as pd

from helpers.crossfold.provenance import (
    build_hdf5_manifest_from_split_dfs,
    build_split_stats_dataframe,
    write_manifest_and_log_stats,
)


def _split_data() -> dict[str, object]:
    train_df = pd.DataFrame(
        [
            {
                "patient_id": 2,
                "image_path": "/src/p2.png",
                "mask_path": "/src/p2_mask.png",
                "label": 0,
                "filename": "PATIENT_2_PATCH_001.png",
            },
            {
                "patient_id": 1,
                "image_path": "/src/p1.png",
                "mask_path": "/src/p1_mask.png",
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
            },
        ]
    )
    return {
        "train_df": train_df,
        "val_df": pd.DataFrame(columns=train_df.columns),
        "test_df": pd.DataFrame(columns=train_df.columns),
        "constraints": {"random_state": 42},
        "train_patients": [1, 2],
        "val_patients": [],
        "test_patients": [],
        "split_seed": 99,
        "split_attempt": 1,
        "objective_score": None,
        "objective_score_split": "TRAIN",
    }


def test_build_hdf5_manifest_from_split_dfs_tracks_relative_hdf5_rows(tmp_path: Path) -> None:
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=_split_data(),
    )

    assert manifest_df[["split", "relative_hdf5_path", "hdf5_row_index"]].to_dict("records") == [
        {
            "split": "TRAIN",
            "relative_hdf5_path": "TRAIN.h5",
            "hdf5_row_index": 0,
        },
        {
            "split": "TRAIN",
            "relative_hdf5_path": "TRAIN.h5",
            "hdf5_row_index": 1,
        },
    ]


def test_build_split_stats_dataframe_summarizes_each_non_empty_split() -> None:
    stats_df = build_split_stats_dataframe(_split_data(), run_id="run-1")

    assert stats_df.to_dict("records") == [
        {
            "run_id": "run-1",
            "split": "TRAIN",
            "n_patients": 2,
            "n_pos_patients": 1,
            "n_neg_patients": 1,
            "n_images": 2,
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


def test_write_manifest_and_log_stats_writes_expected_artifacts(tmp_path: Path) -> None:
    split_data = _split_data()
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=split_data,
    )

    write_manifest_and_log_stats(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        source_hdf5_path=tmp_path / "SOURCE_DATASET.h5",
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=False,
        extra={"note": "ok"},
    )

    assert (tmp_path / "manifest.csv").is_file()
    assert (tmp_path / "split_stats.csv").is_file()
    assert (tmp_path / "run_config.json").is_file()


def test_run_config_json_uses_safe_json_encoding(tmp_path: Path) -> None:
    split_data = _split_data()
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=split_data,
    )

    write_manifest_and_log_stats(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        source_hdf5_path=tmp_path / "SOURCE_DATASET.h5",
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=False,
        extra={"value": 1},
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert payload["extra"]["value"] == 1


def test_write_manifest_and_log_stats_rejects_checksum_mode_for_hdf5_manifests(
    tmp_path: Path,
) -> None:
    split_data = _split_data()
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=split_data,
    )

    try:
        write_manifest_and_log_stats(
            output_dir=tmp_path,
            run_id="run-1",
            normalization_method="NOT_NORMALIZED",
            is_normalized=False,
            source_hdf5_path=tmp_path / "SOURCE_DATASET.h5",
            split_data=split_data,
            manifest_df=manifest_df,
            calc_checksums=True,
        )
    except ValueError as error:
        assert "HDF5-native" in str(error)
    else:
        raise AssertionError("Expected HDF5 checksum request to be rejected.")
