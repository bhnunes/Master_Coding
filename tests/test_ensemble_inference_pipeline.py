from __future__ import annotations

import json
from pathlib import Path

from helpers.ensemble_inference.config import EnsembleInferenceConfig
from helpers.ensemble_inference.pipeline import (
    EnsembleInferenceOutputs,
    run_ensemble_inference_pipeline,
)


def test_run_ensemble_inference_pipeline_writes_run_config(tmp_path: Path) -> None:
    config = EnsembleInferenceConfig(
        recipe_path=tmp_path / "recipe.json",
        hdf5_drive_dir=tmp_path / "dataset",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=8,
        workers=1,
        seed=24,
        visualization_samples=3,
        export_csv=True,
        export_latex=True,
        export_visualizations=True,
    )

    def fake_runner(config: EnsembleInferenceConfig) -> EnsembleInferenceOutputs:
        output_dir = config.output_dir or config.recipe_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        recipe_copy_path = output_dir / config.recipe_path.name
        recipe_copy_path.write_text(
            json.dumps({"ensemble_strategy": "two_stream_spatial_gating"}), encoding="utf-8"
        )
        return EnsembleInferenceOutputs(
            output_dir=output_dir,
            run_config_path=output_dir / "ensemble_inference_run_config.json",
            recipe_copy_path=recipe_copy_path,
            metrics_json_path=output_dir / "metrics.json",
            confusion_matrix_path=output_dir / "confusion_matrix.png",
            csv_report_path=output_dir / "report.csv",
            latex_report_path=output_dir / "report.tex",
            pdf_report_path=output_dir / "report.pdf",
        )

    outputs = run_ensemble_inference_pipeline(config, pipeline_runner=fake_runner)

    payload = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert payload["batch_size"] == 8
    assert payload["visualization_samples"] == 3
    assert outputs.recipe_copy_path.name == "recipe.json"
