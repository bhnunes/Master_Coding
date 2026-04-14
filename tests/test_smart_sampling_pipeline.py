from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.smart_sampling.selection import select_patient_samples as run_select_patient_samples
from helpers.smart_sampling.writer import write_filtered_patient_shard
from helpers.training import data as training_data


def _write_training_shards(shard_dir: Path) -> None:
    images = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    masks = np.zeros((6, 4, 4), dtype=np.uint8)
    masks[1] = 1
    masks[4] = 1
    labels = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
    patient_ids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    filenames = np.array(
        [
            b"PATIENT_1_a.png",
            b"PATIENT_1_b.png",
            b"PATIENT_1_c.png",
            b"PATIENT_2_a.png",
            b"PATIENT_2_b.png",
            b"PATIENT_2_c.png",
        ],
        dtype="S32",
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, object]] = []
    for patient_id in (1, 2):
        patient_mask = patient_ids == patient_id
        shard_path = shard_dir / f"{patient_id}.h5"
        patient_images = images[patient_mask]
        patient_masks = masks[patient_mask]
        patient_labels = labels[patient_mask]
        patient_patient_ids = patient_ids[patient_mask]
        patient_filenames = filenames[patient_mask]
        with h5py.File(shard_path, "w") as handle:
            handle.create_dataset("images", data=patient_images)
            handle.create_dataset("masks", data=patient_masks)
            handle.create_dataset("labels", data=patient_labels)
            handle.create_dataset("patient_ids", data=patient_patient_ids)
            handle.create_dataset("filenames", data=patient_filenames)
            handle.attrs["patient_id"] = patient_id
            handle.attrs["split_name"] = "TRAIN"
            handle.attrs["source_split_hdf5_path"] = "/tmp/stage5/TRAIN.h5"
            handle.attrs["source_split_row_indices_json"] = json.dumps(
                np.where(patient_mask)[0].astype(int).tolist()
            )
        shard_records.append(
            {
                "split": "TRAIN",
                "patient_id": patient_id,
                "relative_hdf5_path": f"TRAIN_shards/{patient_id}.h5",
                "rows": int(patient_mask.sum()),
                "label_0_count": int((patient_labels == 0).sum()),
                "label_1_count": int((patient_labels == 1).sum()),
            }
        )
    pq.write_table(pa.Table.from_pylist(shard_records), shard_dir / "manifest.parquet")


def _build_config(tmp_path: Path, **overrides: object) -> SmartSamplerConfig:
    values: dict[str, Any] = {
        "source_h5_path": tmp_path / "TRAIN.h5",
        "source_shard_dir": tmp_path / "TRAIN_shards",
        "source_manifest_path": tmp_path / "TRAIN_shards" / "manifest.parquet",
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


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(mask),
        }


def test_run_smart_sampling_pipeline_produces_training_compatible_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, source_h5_path=None, source_shard_dir=source_shard_dir),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.filtered_shard_dir == tmp_path / "out" / "TRAIN_FILTERED_shards"
    assert outputs.selection_csv_path is not None
    assert outputs.stats_csv_path is not None
    assert outputs.run_config_path is not None
    assert outputs.summary_json_path == tmp_path / "out" / "filter_summary.json"
    assert outputs.total_input_samples == 6
    assert outputs.selected_sample_count == 4
    assert outputs.rejected_sample_count == 2
    assert outputs.kept_fraction == pytest.approx(4 / 6)
    assert outputs.patient_count == 2
    assert outputs.patients_reduced_count == 2

    manifest = pq.read_table(outputs.filtered_shard_dir / "manifest.parquet").to_pylist()
    assert manifest == [
        {
            "split": "TRAIN_FILTERED",
            "patient_id": 1,
            "relative_hdf5_path": "TRAIN_FILTERED_shards/1.h5",
            "rows": 2,
            "label_0_count": 1,
            "label_1_count": 1,
        },
        {
            "split": "TRAIN_FILTERED",
            "patient_id": 2,
            "relative_hdf5_path": "TRAIN_FILTERED_shards/2.h5",
            "rows": 2,
            "label_0_count": 1,
            "label_1_count": 1,
        },
    ]
    with h5py.File(outputs.filtered_shard_dir / "1.h5", "r") as handle:
        assert set(handle.keys()) == {"filenames", "images", "labels", "masks", "patient_ids"}
        assert handle["patient_ids"][:].tolist() == [1, 1]
        assert bool(handle.attrs["stage7_label_aware"])
        assert handle.attrs["stage7_holdout_mode"] == "within_patient_patch_holdout"
    with h5py.File(outputs.filtered_shard_dir / "2.h5", "r") as handle:
        assert handle["patient_ids"][:].tolist() == [2, 2]

    assert outputs.summary_json_path is not None
    assert outputs.summary_json_path is not None
    assert outputs.summary_json_path is not None
    summary = json.loads(outputs.summary_json_path.read_text(encoding="utf-8"))
    assert summary["total_input_samples"] == 6
    assert summary["kept_samples"] == 4
    assert summary["rejected_samples"] == 2
    assert summary["patients_reduced_count"] == 2
    assert summary["protected_kept_samples"] == 2
    assert summary["protected_positive_label_kept_samples"] == 2
    assert summary["protected_mask_positive_kept_samples"] == 2
    assert summary["sampled_reducible_samples"] == 2
    assert summary["rejected_reducible_samples"] == 2
    assert summary["label_aware_stage7"] is True
    assert summary["holdout_evaluation_mode"] == "within_patient_patch_holdout"
    assert summary["total_positive_label_count"] == 2
    assert summary["selected_positive_label_count"] == 2
    assert summary["selected_negative_label_count"] == 2
    assert summary["model_name"] == "owkin/phikon-v2"


def test_run_smart_sampling_pipeline_stages_outputs_locally_and_publishes_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"
    config = _build_config(
        tmp_path,
        source_h5_path=None,
        source_shard_dir=source_shard_dir,
        output_dir=remote_output_dir,
        local_work_dir=local_work_dir,
        stage_input_locally=True,
        stage_outputs_locally=True,
    )

    outputs = run_smart_sampling_pipeline(config, extractor_factory=_DummyEmbeddingExtractor)

    assert outputs.filtered_shard_dir == remote_output_dir / "TRAIN_FILTERED_shards"
    assert outputs.selection_csv_path == remote_output_dir / "train_filtered_selection.csv"
    assert outputs.stats_csv_path == remote_output_dir / "patient_filter_stats.csv"
    assert outputs.run_config_path == remote_output_dir / "filter_run_config.json"
    assert outputs.summary_json_path == remote_output_dir / "filter_summary.json"
    assert not local_work_dir.exists()
    assert outputs.filtered_shard_dir.exists()
    assert (outputs.filtered_shard_dir / "1.h5").exists()
    assert (outputs.filtered_shard_dir / "2.h5").exists()
    assert (outputs.filtered_shard_dir / "manifest.parquet").exists()
    assert outputs.summary_json_path.exists()


def test_run_smart_sampling_pipeline_keeps_local_work_dir_on_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"
    config = _build_config(
        tmp_path,
        source_h5_path=None,
        source_shard_dir=source_shard_dir,
        output_dir=remote_output_dir,
        local_work_dir=local_work_dir,
        stage_input_locally=True,
        stage_outputs_locally=True,
    )

    def fail_publish(*args: object, **kwargs: object) -> object:
        raise OSError("drive unavailable")

    monkeypatch.setattr("helpers.smart_sampling.pipeline.publish_outputs", fail_publish)

    with pytest.raises(OSError, match="drive unavailable"):
        run_smart_sampling_pipeline(config, extractor_factory=_DummyEmbeddingExtractor)

    assert (remote_output_dir / "TRAIN_FILTERED_shards" / "1.h5").exists()
    assert (remote_output_dir / "TRAIN_FILTERED_shards" / "2.h5").exists()
    assert (remote_output_dir / "TRAIN_FILTERED_shards" / "manifest.parquet").exists()
    assert not (local_work_dir / "input" / "1.h5").exists()
    assert not (local_work_dir / "input" / "2.h5").exists()
    assert not (local_work_dir / "output" / "TRAIN_FILTERED_shards" / "1.h5").exists()
    assert not (local_work_dir / "output" / "TRAIN_FILTERED_shards" / "2.h5").exists()
    assert (local_work_dir / "output" / "filter_summary.json").exists()


def test_run_smart_sampling_pipeline_keeps_current_patient_workspace_when_shard_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"
    config = _build_config(
        tmp_path,
        source_h5_path=None,
        source_shard_dir=source_shard_dir,
        output_dir=remote_output_dir,
        local_work_dir=local_work_dir,
        stage_input_locally=True,
        stage_outputs_locally=True,
    )

    def fail_patient_publish(*args: object, **kwargs: object) -> object:
        raise OSError("patient publish failed")

    monkeypatch.setattr(
        "helpers.smart_sampling.pipeline.publish_patient_output", fail_patient_publish
    )

    with pytest.raises(OSError, match="patient publish failed"):
        run_smart_sampling_pipeline(config, extractor_factory=_DummyEmbeddingExtractor)

    assert (local_work_dir / "input" / "1.h5").exists()
    assert (local_work_dir / "output" / "TRAIN_FILTERED_shards" / "1.h5").exists()
    assert not (remote_output_dir / "TRAIN_FILTERED_shards" / "1.h5").exists()


def test_run_smart_sampling_pipeline_skips_patients_with_existing_filtered_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    remote_output_dir.mkdir()
    config = _build_config(
        tmp_path,
        source_h5_path=None,
        source_shard_dir=source_shard_dir,
        output_dir=remote_output_dir,
        overwrite_output=False,
    )

    preexisting_output_path = remote_output_dir / "TRAIN_FILTERED_shards" / "1.h5"
    write_filtered_patient_shard(
        config,
        patient_id=1,
        source_shard_path=source_shard_dir / "1.h5",
        source_relative_path="TRAIN_shards/1.h5",
        output_path=preexisting_output_path,
        output_relative_path="TRAIN_FILTERED_shards/1.h5",
        selected_row_indices=[0, 1],
        signature_source_dir=source_shard_dir,
        stage7_summary_attrs={
            "protected_kept_samples": 1,
            "protected_positive_label_kept_samples": 1,
            "protected_mask_positive_kept_samples": 1,
            "sampled_reducible_samples": 1,
            "rejected_reducible_samples": 1,
            "total_positive_label_count": 1,
            "total_negative_label_count": 2,
            "selected_positive_label_count": 1,
            "selected_negative_label_count": 1,
            "patients_reduced_count": 1,
        },
    )

    seen_patient_ids: list[int] = []

    def record_selection(
        h5_path: str,
        patient_id: int,
        patient_indices: np.ndarray[tuple[int], np.dtype[np.int64]],
        extractor: Any,
        selection_config: SmartSamplerConfig,
    ) -> object:
        seen_patient_ids.append(patient_id)
        return run_select_patient_samples(
            h5_path,
            patient_id,
            patient_indices,
            extractor,
            selection_config,
        )

    monkeypatch.setattr("helpers.smart_sampling.pipeline.select_patient_samples", record_selection)

    outputs = run_smart_sampling_pipeline(config, extractor_factory=_DummyEmbeddingExtractor)

    assert seen_patient_ids == [2]
    assert outputs.selected_sample_count == 4
    assert outputs.rejected_sample_count == 2
    assert outputs.patients_reduced_count == 2
    manifest = pq.read_table(outputs.filtered_shard_dir / "manifest.parquet").to_pylist()
    assert manifest == [
        {
            "split": "TRAIN_FILTERED",
            "patient_id": 1,
            "relative_hdf5_path": "TRAIN_FILTERED_shards/1.h5",
            "rows": 2,
            "label_0_count": 1,
            "label_1_count": 1,
        },
        {
            "split": "TRAIN_FILTERED",
            "patient_id": 2,
            "relative_hdf5_path": "TRAIN_FILTERED_shards/2.h5",
            "rows": 2,
            "label_0_count": 1,
            "label_1_count": 1,
        },
    ]
    assert outputs.summary_json_path is not None
    summary = json.loads(outputs.summary_json_path.read_text(encoding="utf-8"))
    assert summary["protected_kept_samples"] == 2
    assert summary["sampled_reducible_samples"] == 2
    assert summary["selected_positive_label_count"] == 2
    assert summary["selected_negative_label_count"] == 2


def test_run_smart_sampling_pipeline_reports_noop_summary_when_every_patch_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(
            tmp_path,
            source_h5_path=None,
            source_shard_dir=source_shard_dir,
            adaptive_keep_enabled=False,
            keep_min=6,
            keep_improvement_threshold=0.5,
            keep_patience=1,
            m_max=6,
            protect_positive_labels=False,
            protect_mask_positive=False,
        ),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.total_input_samples == 6
    assert outputs.selected_sample_count == 6
    assert outputs.rejected_sample_count == 0
    assert outputs.kept_fraction == pytest.approx(1.0)
    assert outputs.patients_reduced_count == 0


def test_run_smart_sampling_pipeline_can_use_gist_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(
            tmp_path, source_h5_path=None, source_shard_dir=source_shard_dir, use_gist=True
        ),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.selected_sample_count == 4
    assert outputs.selection_csv_path is not None
    selection_manifest = pd.read_csv(outputs.selection_csv_path)
    sampled_rows = selection_manifest[selection_manifest["selection_bucket"] == "gist_sampled"]
    assert set(sampled_rows["selection_method"].unique()).issubset(
        {"gist_facility_location", "gist_keep_all"}
    )


def test_run_smart_sampling_pipeline_records_protected_and_sampled_selection_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_shard_dir = tmp_path / "TRAIN_shards"
    _write_training_shards(source_shard_dir)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, source_h5_path=None, source_shard_dir=source_shard_dir),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.selection_csv_path is not None
    assert outputs.stats_csv_path is not None
    selection_manifest = pd.read_csv(outputs.selection_csv_path)
    stats = pd.read_csv(outputs.stats_csv_path)

    assert set(selection_manifest["selection_bucket"].unique()) == {
        "protected_kept",
        "legacy_sampled",
    }
    assert set(
        selection_manifest.loc[
            selection_manifest["selection_bucket"] == "protected_kept", "selection_method"
        ].unique()
    ) == {"protected_retention"}
    assert stats["protected_count"].tolist() == [1, 1]
    assert stats["sampled_reducible_count"].tolist() == [1, 1]
    assert stats["rejected_reducible_count"].tolist() == [1, 1]
    assert "plateau_threshold" in selection_manifest.columns
    assert "plateau_stop_reason" in selection_manifest.columns
    assert "heldout_count" in stats.columns
    assert "adaptive_m_target" in stats.columns
    assert "label_aware_stage7" in selection_manifest.columns
    assert "protected_positive_label_count" in stats.columns
    assert "selected_positive_label_count" in stats.columns
