from __future__ import annotations

import json
from pathlib import Path

from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.pipeline import (
    EnsembleOptimizerOutputs,
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
