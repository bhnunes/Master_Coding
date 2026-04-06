from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.training import data as training_data
from helpers.training.data import HybridProstateDataset


def _write_training_hdf5(path: Path) -> None:
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

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("patient_ids", data=patient_ids)
        handle.create_dataset("filenames", data=filenames)


def _build_config(tmp_path: Path, **overrides: object) -> SmartSamplerConfig:
    values: dict[str, Any] = {
        "source_h5_path": tmp_path / "TRAIN.h5",
        "output_dir": tmp_path / "out",
        "output_filename": "TRAIN_FILTERED.h5",
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
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, source_h5_path=source_path),
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.filtered_h5_path == tmp_path / "out" / "TRAIN_FILTERED.h5"
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

    with h5py.File(outputs.filtered_h5_path, "r") as handle:
        assert set(handle.keys()) == {"filenames", "images", "labels", "masks", "patient_ids"}
        assert handle["patient_ids"][:].tolist() == [1, 1, 2, 2]

    assert outputs.summary_json_path is not None
    summary = json.loads(outputs.summary_json_path.read_text(encoding="utf-8"))
    assert summary["total_input_samples"] == 6
    assert summary["kept_samples"] == 4
    assert summary["rejected_samples"] == 2
    assert summary["patients_reduced_count"] == 2
    assert summary["protected_kept_samples"] == 2
    assert summary["sampled_reducible_samples"] == 2
    assert summary["rejected_reducible_samples"] == 2
    assert summary["model_name"] == "owkin/phikon-v2"

    dataset = HybridProstateDataset(str(outputs.filtered_h5_path), mode="train")
    assert len(dataset) == 4


def test_run_smart_sampling_pipeline_stages_outputs_locally_and_publishes_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"
    config = _build_config(
        tmp_path,
        source_h5_path=source_path,
        output_dir=remote_output_dir,
        local_work_dir=local_work_dir,
        stage_input_locally=True,
        stage_outputs_locally=True,
    )

    outputs = run_smart_sampling_pipeline(config, extractor_factory=_DummyEmbeddingExtractor)

    assert outputs.filtered_h5_path == remote_output_dir / "TRAIN_FILTERED.h5"
    assert outputs.selection_csv_path == remote_output_dir / "train_filtered_selection.csv"
    assert outputs.stats_csv_path == remote_output_dir / "patient_filter_stats.csv"
    assert outputs.run_config_path == remote_output_dir / "filter_run_config.json"
    assert outputs.summary_json_path == remote_output_dir / "filter_summary.json"
    assert not local_work_dir.exists()
    assert outputs.filtered_h5_path.exists()
    assert outputs.summary_json_path.exists()


def test_run_smart_sampling_pipeline_keeps_local_work_dir_on_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    remote_output_dir = tmp_path / "drive"
    local_work_dir = tmp_path / "content"
    config = _build_config(
        tmp_path,
        source_h5_path=source_path,
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

    assert (local_work_dir / "input" / source_path.name).exists()
    assert (local_work_dir / "output" / "TRAIN_FILTERED.h5").exists()
    assert (local_work_dir / "output" / "filter_summary.json").exists()


def test_run_smart_sampling_pipeline_reports_noop_summary_when_every_patch_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(
            tmp_path,
            source_h5_path=source_path,
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
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, source_h5_path=source_path, use_gist=True),
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
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    outputs = run_smart_sampling_pipeline(
        _build_config(tmp_path, source_h5_path=source_path),
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
