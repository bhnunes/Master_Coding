import json
from pathlib import Path

import h5py
import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

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
    }


def test_build_hdf5_manifest_from_split_dfs_tracks_canonical_source_rows(tmp_path: Path) -> None:
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        split_data=_split_data(),
    )

    assert manifest_df[["split", "source_row_index", "source_image_path"]].to_dict("records") == [
        {
            "split": "TRAIN",
            "source_row_index": 0,
            "source_image_path": "/src/p2.png",
        },
        {
            "split": "TRAIN",
            "source_row_index": 1,
            "source_image_path": "/src/p1.png",
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

    with h5py.File(tmp_path / "SOURCE_DATASET.h5", "w") as handle:
        handle.create_dataset("images", data=[[[[0, 0, 0]]]], dtype="uint8")
        handle.create_dataset("masks", data=[[[0]]], dtype="uint8")
        handle.create_dataset("labels", data=[0], dtype="uint8")
        handle.create_dataset("patient_ids", data=[1], dtype="int32")
        handle.create_dataset("filenames", data=[b"PATIENT_1_PATCH_001.png"])

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

    with h5py.File(tmp_path / "SOURCE_DATASET.h5", "w") as handle:
        handle.create_dataset("images", data=[[[[0, 0, 0]]]], dtype="uint8")
        handle.create_dataset("masks", data=[[[0]]], dtype="uint8")
        handle.create_dataset("labels", data=[0], dtype="uint8")
        handle.create_dataset("patient_ids", data=[1], dtype="int32")
        handle.create_dataset("filenames", data=[b"PATIENT_1_PATCH_001.png"])

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


def test_write_manifest_and_log_stats_records_stage4_cleaning_lineage(tmp_path: Path) -> None:
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
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/accepted_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "abc123"

    write_manifest_and_log_stats(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        source_hdf5_path=source_path,
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=False,
        extra={"value": 1},
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert (
        payload["source_hdf5_provenance"]["attrs"]["stage4_cleaning_manifest_path"]
        == "/tmp/accepted_manifest.csv"
    )
    assert payload["source_hdf5_provenance"]["attrs"]["stage4_cleaning_manifest_sha256"] == "abc123"


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


def test_write_manifest_and_log_stats_reuses_cached_source_provenance(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
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

    def fail_collect(_: Path) -> dict[str, object]:
        raise AssertionError("collect_hdf5_provenance should not run when provenance is cached")

    monkeypatch.setattr("helpers.crossfold.provenance.collect_hdf5_provenance", fail_collect)

    cached_provenance = {"path": str(source_path), "sha256": "cached-hash", "attrs": {}}
    write_manifest_and_log_stats(
        output_dir=tmp_path,
        run_id="run-1",
        normalization_method="NOT_NORMALIZED",
        is_normalized=False,
        source_hdf5_path=source_path,
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=False,
        source_hdf5_provenance=cached_provenance,
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert payload["source_hdf5_provenance"] == cached_provenance
