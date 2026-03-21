from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.metadata import SelectedModelMetadata
from helpers.ensemble_optimizer.pipeline import (
    EnsembleOptimizerOutputs,
    _execute_pipeline,
    run_ensemble_optimizer_pipeline,
)


def test_run_ensemble_optimizer_pipeline_writes_run_config(tmp_path: Path) -> None:
    config = EnsembleOptimizerConfig(
        hdf5_drive_dir=tmp_path / "dataset",
        metadata_dir=tmp_path / "metadata",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        pred_cache_dir=tmp_path / "pred_cache",
        stage_input_locally=False,
        overwrite_output=True,
        seed=24,
        batch_size=8,
        workers=1,
        top_models=4,
        sort_metric="best_val_auprc_pixel_score",
        val_holdout_frac=0.2,
        semantic_architectures=("SWIN",),
        spatial_architectures=("FPN",),
        roi_context_scale=4,
        roi_max_median=0.6,
        roi_empty_max=0.5,
        roi_min_pos_recall=0.8,
        spill_penalty_lambda=0.1,
        spatial_patient_policy="positive_only",
        num_trials_semantic=5,
        num_trials_spatial=7,
    )

    def fake_runner(config: EnsembleOptimizerConfig) -> EnsembleOptimizerOutputs:
        recipe_path = config.output_dir / "ENSEMBLE_TWO_STREAM_2026-03-20_10_00_00.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps({"ensemble_strategy": "two_stream_spatial_gating"}), encoding="utf-8"
        )
        return EnsembleOptimizerOutputs(
            recipe_path=recipe_path, run_config_path=config.output_dir / "run_config.json"
        )

    outputs = run_ensemble_optimizer_pipeline(config, pipeline_runner=fake_runner)

    run_config = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert outputs.recipe_path.name.startswith("ENSEMBLE_TWO_STREAM_")
    assert run_config["top_models"] == 4
    assert run_config["semantic_architectures"] == ["SWIN"]


def test_execute_pipeline_selects_models_from_optimization_subset_before_holdout_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = EnsembleOptimizerConfig(
        hdf5_drive_dir=tmp_path / "dataset",
        metadata_dir=tmp_path / "metadata",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        pred_cache_dir=tmp_path / "pred_cache",
        stage_input_locally=False,
        overwrite_output=True,
        seed=24,
        batch_size=8,
        workers=1,
        top_models=2,
        sort_metric="best_val_auprc_pixel_score",
        val_holdout_frac=0.2,
        semantic_architectures=("SWIN",),
        spatial_architectures=("FPN",),
        roi_context_scale=4,
        roi_max_median=0.6,
        roi_empty_max=0.5,
        roi_min_pos_recall=0.8,
        spill_penalty_lambda=0.1,
        spatial_patient_policy="positive_only",
        num_trials_semantic=5,
        num_trials_spatial=7,
    )
    selected_models = [
        SelectedModelMetadata(
            architecture="SWIN",
            encoder="enc-a",
            checkpoint_path="a.ckpt",
            metadata_filename="a_meta.json",
            sort_metric_value=0.2,
            raw_metadata={},
        ),
        SelectedModelMetadata(
            architecture="FPN",
            encoder="enc-b",
            checkpoint_path="b.ckpt",
            metadata_filename="b_meta.json",
            sort_metric_value=0.1,
            raw_metadata={},
        ),
    ]
    observed: dict[str, object] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.setup_validation_hdf5",
        lambda *args, **kwargs: tmp_path / "dataset" / "VALIDATION.h5",
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline._build_validation_split",
        lambda config, validation_h5_path: type(
            "Split",
            (),
            {"optimization_patients": {"p1", "p2"}, "holdout_patients": {"p3"}},
        )(),
    )

    def fake_select_models(
        config: EnsembleOptimizerConfig,
        validation_h5_path: Path,
        optimization_patients: set[str],
        device: Any,
    ) -> tuple[list[SelectedModelMetadata], dict[str, float]]:
        del config, validation_h5_path, device
        observed["optimization_patients"] = optimization_patients
        return selected_models, {"a_meta.json": 0.9, "b_meta.json": 0.8}

    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline._select_models_from_optimization_subset",
        fake_select_models,
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.load_ensemble_models",
        lambda selected_models, device: ([object(), object()], []),
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.create_validation_dataloader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.run_two_stream_optimization",
        lambda config, models, dataloader, device, predefined_split: type(
            "Result",
            (),
            {
                "semantic_indices": [0],
                "spatial_indices": [1],
                "semantic_weights": [1.0],
                "spatial_weights": [1.0],
                "roi_threshold": 0.4,
                "holdout_metrics": {"Macro_AUPRC_in_ROI": 0.7},
            },
        )(),
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.build_recipe_metadata",
        lambda **kwargs: {"ensemble_strategy": "two_stream_spatial_gating"},
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.pipeline.write_recipe_metadata",
        lambda payload, output_dir, timestamp: output_dir / "recipe.json",
    )

    outputs = _execute_pipeline(config)

    run_config = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert observed["optimization_patients"] == {"p1", "p2"}
    assert run_config["optimization_patients"] == ["p1", "p2"]
    assert run_config["holdout_patients"] == ["p3"]
    assert run_config["selected_models"][0]["optimization_subset_score"] == 0.9
