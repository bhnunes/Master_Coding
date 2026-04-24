from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref
from helpers.extraction.master_manifest import MasterManifest
from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.training.master_manifest_queries import load_training_records

PATCH_SIDE = 4
RGB_CHANNELS = 3
TOTAL_INPUT_SAMPLES = 6
SELECTED_SAMPLE_COUNT = 4
REJECTED_SAMPLE_COUNT = 2
PATIENT_COUNT = 2


def _write_stage2_patient_shards_and_master_manifest(tmp_path: Path) -> Path:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    patient_1_path = tmp_path / "PATCHES" / "1.h5"
    patient_2_path = tmp_path / "PATCHES" / "2.h5"
    patient_1_path.parent.mkdir(parents=True, exist_ok=True)

    _write_patient_shard(
        patient_1_path,
        patient_id=1,
        labels=[0, 1, 0],
        filenames=["PATIENT_1_a.png", "PATIENT_1_b.png", "PATIENT_1_c.png"],
        mask_positive_rows={1},
    )
    _write_patient_shard(
        patient_2_path,
        patient_id=2,
        labels=[0, 1, 0],
        filenames=["PATIENT_2_a.png", "PATIENT_2_b.png", "PATIENT_2_c.png"],
        mask_positive_rows={1},
    )

    manifest = MasterManifest(master_manifest_path)
    manifest.initialize()
    with sqlite3.connect(master_manifest_path) as connection:
        for patient_id, shard_path, filenames in (
            (1, patient_1_path, ["PATIENT_1_a.png", "PATIENT_1_b.png", "PATIENT_1_c.png"]),
            (2, patient_2_path, ["PATIENT_2_a.png", "PATIENT_2_b.png", "PATIENT_2_c.png"]),
        ):
            for row_index, filename in enumerate(filenames):
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path,
                        source_row_index,
                        filename,
                        patient_id,
                        label,
                        slide_id,
                        source_signature,
                        source_image_path,
                        source_mask_path,
                        source_slide_path,
                        stage2_case_record_id,
                        stage2_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        to_manifest_path_ref(shard_path, manifest_path=master_manifest_path),
                        row_index,
                        filename,
                        patient_id,
                        1 if row_index == 1 else 0,
                        f"slide_{patient_id}",
                        f"sig-{patient_id}",
                        build_hdf5_dataset_ref("images", row_index),
                        build_hdf5_dataset_ref("masks", row_index),
                        f"/slides/{patient_id}.svs",
                        patient_id,
                        "COMPLETED",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO patch_stage_state (
                        patch_id,
                        split,
                        is_stage4_accepted,
                        last_updated_stage_name
                    ) VALUES (?, 'TRAIN', 1, 'STAGE4')
                    """,
                    ((cursor.lastrowid or 0),),
                )
        connection.commit()
    return master_manifest_path


def _write_patient_shard(
    shard_path: Path,
    *,
    patient_id: int,
    labels: list[int],
    filenames: list[str],
    mask_positive_rows: set[int],
) -> None:
    images = np.arange(
        len(labels) * PATCH_SIDE * PATCH_SIDE * RGB_CHANNELS, dtype=np.uint8
    ).reshape(len(labels), PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS)
    masks = np.zeros((len(labels), PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
    for row_index in mask_positive_rows:
        masks[row_index] = 1
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.uint8))
        handle.create_dataset(
            "patient_ids",
            data=np.asarray([patient_id] * len(labels), dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.asarray([filename.encode("utf-8") for filename in filenames]),
        )
        handle.attrs["source_signature"] = f"sig-{patient_id}"


def _build_config(tmp_path: Path, **overrides: object) -> SmartSamplerConfig:
    values: dict[str, Any] = {
        "master_manifest_path": tmp_path / "master_manifest.sqlite",
        "output_dir": tmp_path / "out",
        "output_filename": "TRAIN_FILTERED_shards",
        "local_work_dir": None,
        "stage_input_locally": False,
        "stage_outputs_locally": False,
        "clean_local_work_dir": True,
        "write_sidecars": True,
        "overwrite_output": True,
        "model_name": "owkin/phikon-v2",
        "batch_size": 8,
        "device": "cpu",
        "n_start": 8,
        "n_max": 8,
        "growth_factor": 2.0,
        "stability_threshold": 0.85,
        "stability_repeats": 2,
        "max_steps": 2,
        "intersection_ratio_threshold": 0.2,
        "k_min": 20,
        "k_max": 80,
        "adaptive_keep_enabled": True,
        "keep_min": 2,
        "keep_step": 1,
        "keep_improvement_threshold": 0.02,
        "keep_patience": 2,
        "m_max": 1,
        "seed": 42,
        "num_workers": 0,
        "protect_positive_labels": True,
        "protect_mask_positive": True,
        "positive_mask_fraction_threshold": 0.0,
    }
    values.update(overrides)
    return SmartSamplerConfig(**values)


class _DummyEmbeddingExtractor:
    def __init__(self, config: SmartSamplerConfig) -> None:
        self.config = config

    def get_embeddings(
        self, _h5_path: str, indices: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        values = indices.astype(np.float32).reshape(-1, 1)
        return np.concatenate([values, values + 0.5], axis=1)


def test_run_smart_sampling_pipeline_updates_sqlite_and_writes_sidecars(tmp_path: Path) -> None:
    master_manifest_path = _write_stage2_patient_shards_and_master_manifest(tmp_path)

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, master_manifest_path=master_manifest_path),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.output_dir == tmp_path / "out"
    assert outputs.selection_csv_path == tmp_path / "out" / "train_filtered_selection.csv"
    assert outputs.stats_csv_path == tmp_path / "out" / "patient_filter_stats.csv"
    assert outputs.run_config_path == tmp_path / "out" / "filter_run_config.json"
    assert outputs.summary_json_path == tmp_path / "out" / "filter_summary.json"
    assert outputs.total_input_samples == TOTAL_INPUT_SAMPLES
    assert outputs.selected_sample_count == SELECTED_SAMPLE_COUNT
    assert outputs.rejected_sample_count == REJECTED_SAMPLE_COUNT
    assert outputs.kept_fraction == pytest.approx(SELECTED_SAMPLE_COUNT / TOTAL_INPUT_SAMPLES)
    assert outputs.patient_count == PATIENT_COUNT
    assert outputs.patients_reduced_count == PATIENT_COUNT
    assert not (tmp_path / "out" / "TRAIN_FILTERED_shards").exists()

    selection_manifest = pd.read_csv(outputs.selection_csv_path)
    assert set(selection_manifest["selection_bucket"].unique()) == {
        "protected_kept",
        "legacy_sampled",
    }
    assert {"source_hdf5_path", "source_row_index", "filename", "label"}.issubset(
        selection_manifest.columns
    )

    summary = json.loads(outputs.summary_json_path.read_text(encoding="utf-8"))
    assert summary["master_manifest_path"] == str(master_manifest_path)
    assert summary["protected_kept_samples"] == REJECTED_SAMPLE_COUNT
    assert summary["sampled_reducible_samples"] == REJECTED_SAMPLE_COUNT
    assert summary["selected_positive_label_count"] == REJECTED_SAMPLE_COUNT
    assert summary["selected_negative_label_count"] == REJECTED_SAMPLE_COUNT

    with sqlite3.connect(master_manifest_path) as connection:
        stage_rows = connection.execute(
            "SELECT sampling_decision, is_stage7_selected, last_updated_stage_name "
            "FROM patch_stage_state ORDER BY patch_id ASC"
        ).fetchall()
        run_rows = connection.execute(
            "SELECT stage_name, config_path, input_summary_json_path FROM runs ORDER BY run_id ASC"
        ).fetchall()

    assert (
        sum(1 for decision, *_ in stage_rows if decision == "protected_kept")
        == REJECTED_SAMPLE_COUNT
    )
    assert (
        sum(1 for decision, *_ in stage_rows if decision == "sampled_kept") == REJECTED_SAMPLE_COUNT
    )
    assert (
        sum(1 for decision, *_ in stage_rows if decision == "rejected_reducible")
        == REJECTED_SAMPLE_COUNT
    )
    assert sum(1 for _, selected, _ in stage_rows if selected == 1) == SELECTED_SAMPLE_COUNT
    assert {stage_name for _, _, stage_name in stage_rows} == {"STAGE6"}
    assert run_rows == [
        (
            "STAGE6",
            to_manifest_path_ref(
                tmp_path / "out" / "filter_run_config.json",
                manifest_path=master_manifest_path,
            ),
            to_manifest_path_ref(
                tmp_path / "out" / "filter_summary.json",
                manifest_path=master_manifest_path,
            ),
        )
    ]


def test_run_smart_sampling_pipeline_can_stage_inputs_locally_and_publish_sidecars(
    tmp_path: Path,
) -> None:
    master_manifest_path = _write_stage2_patient_shards_and_master_manifest(tmp_path)
    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"

    outputs = run_smart_sampling_pipeline(
        _build_config(
            tmp_path,
            master_manifest_path=master_manifest_path,
            output_dir=remote_output_dir,
            local_work_dir=local_work_dir,
            stage_input_locally=True,
            stage_outputs_locally=True,
        ),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.output_dir == remote_output_dir
    assert outputs.selection_csv_path == remote_output_dir / "train_filtered_selection.csv"
    assert outputs.stats_csv_path == remote_output_dir / "patient_filter_stats.csv"
    assert outputs.run_config_path == remote_output_dir / "filter_run_config.json"
    assert outputs.summary_json_path == remote_output_dir / "filter_summary.json"
    assert outputs.selection_csv_path.exists()
    assert outputs.stats_csv_path.exists()
    assert outputs.run_config_path.exists()
    assert outputs.summary_json_path.exists()
    assert not local_work_dir.exists()


def test_run_smart_sampling_pipeline_matches_runtime_stage7_selected_query(tmp_path: Path) -> None:
    master_manifest_path = _write_stage2_patient_shards_and_master_manifest(tmp_path)

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, master_manifest_path=master_manifest_path),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    selected_records = load_training_records(master_manifest_path, smart_sampling=True)
    selected_identities = {
        (str(record.source_hdf5_path), record.source_row_index) for record in selected_records
    }
    selection_manifest = pd.read_csv(outputs.selection_csv_path)
    manifest_selected_identities = {
        (str(row.source_hdf5_path), int(row.source_row_index))
        for row in selection_manifest.itertuples(index=False)
        if str(row.selection_bucket) != "rejected_reducible"
    }

    assert len(selected_records) == outputs.selected_sample_count
    assert selected_identities == manifest_selected_identities
    assert all(record.is_stage7_selected for record in selected_records)
    assert {record.sampling_decision for record in selected_records} == {
        "protected_kept",
        "sampled_kept",
    }


def test_run_smart_sampling_pipeline_can_use_gist_selector(tmp_path: Path) -> None:
    master_manifest_path = _write_stage2_patient_shards_and_master_manifest(tmp_path)

    outputs = run_smart_sampling_pipeline(
        _build_config(
            tmp_path,
            master_manifest_path=master_manifest_path,
            use_gist=True,
        ),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.selection_csv_path is not None
    selection_manifest = pd.read_csv(outputs.selection_csv_path)
    sampled_rows = selection_manifest[selection_manifest["selection_bucket"] == "gist_sampled"]
    assert set(sampled_rows["selection_method"].unique()).issubset(
        {"gist_facility_location", "gist_keep_all"}
    )
