from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from helpers.ensemble_inference.config import EnsembleInferenceConfig
from helpers.ensemble_inference.pipeline import (
    EnsembleInferenceOutputs,
    PipelinePreparation,
    _execute_pipeline,
    run_ensemble_inference_pipeline,
)
from helpers.ensemble_inference.recipe import EnsembleRecipe
from helpers.ensemble_postprocessing import PostprocessingConfig

PIPELINE_BATCH_SIZE = 8
PIPELINE_VISUALIZATION_SAMPLES = 3


@pytest.fixture(autouse=True)
def _stub_cuda_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("helpers.ensemble_inference.pipeline.require_cuda_device", lambda: "cuda")


def _pipeline_config(
    tmp_path: Path,
    *,
    visualization_samples: int,
    export_visualizations: bool,
) -> EnsembleInferenceConfig:
    return EnsembleInferenceConfig(
        recipe_path=tmp_path / "recipe.json",
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=PIPELINE_BATCH_SIZE,
        workers=1,
        seed=24,
        visualization_samples=visualization_samples,
        export_csv=False,
        export_latex=False,
        export_visualizations=export_visualizations,
    )


def _stub_successful_execute_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_copy_path = tmp_path / "reports" / "recipe.json"
    test_layout = type(
        "_Layout",
        (),
        {
            "master_manifest_path": tmp_path / "master_manifest.sqlite",
        },
    )()
    recipe = EnsembleRecipe(
        strategy="two_stream_spatial_gating",
        roi_threshold=0.33,
        decision_threshold=0.57,
        postprocessing_config=PostprocessingConfig(
            min_component_area_px=0,
            min_patient_positive_patches=1,
            min_patient_positive_area_fraction=0.0,
            min_component_area_fraction_patch=0.0,
        ),
        roi_scale=4,
        model_registry=[],
        raw_payload={},
    )

    def fake_prepare_pipeline(
        config: EnsembleInferenceConfig,
        *,
        output_dir: Path,
        normalizer_device: torch.device,
    ) -> PipelinePreparation:
        del config, normalizer_device
        output_dir.mkdir(parents=True, exist_ok=True)
        recipe_copy_path.write_text("{}", encoding="utf-8")
        return PipelinePreparation(
            recipe=recipe,
            recipe_payload={},
            recipe_copy_path=recipe_copy_path,
            test_layout=test_layout,
            test_dataset_provenance={"attrs": {}},
            observed_checkpoint_hashes={},
        )

    def fake_save_confusion_matrix_png(
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        output_path: Path,
    ) -> Path:
        del tp, fp, fn, tn
        output_path.write_text("png", encoding="utf-8")
        return output_path

    def fake_run_reports(**kwargs: object) -> tuple[Path, Path | None, Path | None, Path | None]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        report_path = output_dir / "report.md"
        report_path.write_text("report", encoding="utf-8")
        return report_path, None, None, None

    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.require_cuda_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.resolve_dataloader_stain_normalizer_device",
        lambda device, workers: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline._prepare_pipeline", fake_prepare_pipeline
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.create_test_dataloader", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.load_recipe_models", lambda *a, **k: ([], [])
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.analyze_ensemble_metrics",
        lambda *a, **k: {"confusion_matrix": {"tp": 1, "fp": 0, "fn": 0, "tn": 1}},
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.save_confusion_matrix_png",
        fake_save_confusion_matrix_png,
    )
    monkeypatch.setattr("helpers.ensemble_inference.pipeline._run_reports", fake_run_reports)
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.collect_runtime_environment",
        lambda: {"git_commit": "test"},
    )
    monkeypatch.setattr("helpers.ensemble_inference.pipeline.hash_file_sha256", lambda path: "sha")


def test_run_ensemble_inference_pipeline_writes_run_config(tmp_path: Path) -> None:
    config = EnsembleInferenceConfig(
        recipe_path=tmp_path / "recipe.json",
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=PIPELINE_BATCH_SIZE,
        workers=1,
        seed=24,
        visualization_samples=PIPELINE_VISUALIZATION_SAMPLES,
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
            markdown_report_path=output_dir / "report.md",
            csv_report_path=output_dir / "report.csv",
            latex_report_path=output_dir / "report.tex",
            pdf_report_path=output_dir / "report.pdf",
        )

    outputs = run_ensemble_inference_pipeline(config, pipeline_runner=fake_runner)

    payload = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert payload["batch_size"] == PIPELINE_BATCH_SIZE
    assert payload["visualization_samples"] == PIPELINE_VISUALIZATION_SAMPLES
    assert payload["runtime_vahadane_backend"] == "fixed_source"
    assert "runtime_environment" in payload
    assert "git_commit" in payload["runtime_environment"]
    assert outputs.recipe_copy_path.name == "recipe.json"
    assert outputs.markdown_report_path.name == "report.md"


def test_execute_pipeline_skips_visualization_export_when_feature_flag_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_successful_execute_pipeline(tmp_path, monkeypatch)
    config = _pipeline_config(
        tmp_path,
        visualization_samples=PIPELINE_VISUALIZATION_SAMPLES,
        export_visualizations=False,
    )

    def fail_export_visualizations(*args: object, **kwargs: object) -> list[Path]:
        del args, kwargs
        raise AssertionError("visualization export should have been skipped")

    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.export_visualizations",
        fail_export_visualizations,
    )

    outputs = _execute_pipeline(config)

    payload = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert payload["export_visualizations"] is False
    assert not (outputs.output_dir / "visualizations").exists()
    assert "Visualization image-sample export skipped" in capsys.readouterr().out


def test_execute_pipeline_exports_visualizations_when_feature_flag_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_successful_execute_pipeline(tmp_path, monkeypatch)
    config = _pipeline_config(
        tmp_path,
        visualization_samples=PIPELINE_VISUALIZATION_SAMPLES,
        export_visualizations=True,
    )
    captured: dict[str, int] = {}

    def fake_export_visualizations(*args: object, **kwargs: object) -> list[Path]:
        del args, kwargs
        captured["calls"] = captured.get("calls", 0) + 1
        return []

    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.export_visualizations",
        fake_export_visualizations,
    )

    outputs = _execute_pipeline(config)
    payload = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))

    assert payload["export_visualizations"] is True
    assert payload["visualization_samples"] == PIPELINE_VISUALIZATION_SAMPLES
    assert captured == {"calls": 1}


def test_execute_pipeline_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"model-v2")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "recipe_schema_version": 2,
                "calibration_objective": "balanced_rule6",
                "ensemble_strategy": "two_stream_spatial_gating",
                "roi_config": {"threshold": 0.33, "scale": 4},
                "decision_config": {"threshold": 0.57},
                "postprocessing_config": {
                    "method": "threshold_components_patient_suppression",
                    "min_component_area_px": 16,
                    "min_patient_positive_patches": 3,
                    "min_patient_positive_area_fraction": 0.0,
                    "min_component_area_fraction_patch": 0.0,
                },
                "validation_calibration_summary": {"objective": "balanced_rule6"},
                "model_registry": [
                    {
                        "architecture": "SWIN",
                        "encoder": "enc-a",
                        "checkpoint_path": str(checkpoint_path),
                        "stream_role": "semantic",
                        "weight": 1.0,
                        "checkpoint_sha256": "wrong-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = EnsembleInferenceConfig(
        recipe_path=recipe_path,
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=PIPELINE_BATCH_SIZE,
        workers=1,
        seed=24,
        visualization_samples=0,
        export_csv=False,
        export_latex=False,
        export_visualizations=False,
    )
    assert config.output_dir is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    test_layout = type(
        "_Layout",
        (),
        {
            "master_manifest_path": tmp_path / "master_manifest.sqlite",
        },
    )()
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.setup_test_data",
        lambda *args, **kwargs: test_layout,
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.collect_test_dataset_provenance",
        lambda layout: {
            "attrs": {
                "master_manifest_sha256": "manifest-sha",
                "stage4_split_bundle_id": None,
                "runtime_normalization_method": "none",
                "runtime_vahadane_backend": None,
                "normalization_method": "none",
                "normalization_artifact_id": None,
            }
        },
    )

    with pytest.raises(ValueError, match="Checkpoint provenance mismatch"):
        _execute_pipeline(config)


def test_execute_pipeline_rejects_test_hdf5_lineage_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"model-v1")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "recipe_schema_version": 2,
                "calibration_objective": "balanced_rule6",
                "ensemble_strategy": "two_stream_spatial_gating",
                "roi_config": {"threshold": 0.33, "scale": 4},
                "decision_config": {"threshold": 0.57},
                "postprocessing_config": {
                    "method": "threshold_components_patient_suppression",
                    "min_component_area_px": 16,
                    "min_patient_positive_patches": 3,
                    "min_patient_positive_area_fraction": 0.0,
                    "min_component_area_fraction_patch": 0.0,
                },
                "validation_calibration_summary": {"objective": "balanced_rule6"},
                "model_registry": [
                    {
                        "architecture": "SWIN",
                        "encoder": "enc-a",
                        "checkpoint_path": str(checkpoint_path),
                        "stream_role": "semantic",
                        "weight": 1.0,
                        "checkpoint_sha256": "ok-hash",
                    }
                ],
                "provenance": {
                    "validation": {
                        "sha256": "validation-sha",
                        "attrs": {
                            "master_manifest_sha256": "manifest-sha",
                            "stage4_split_bundle_id": None,
                            "runtime_normalization_method": "none",
                            "runtime_vahadane_backend": None,
                            "normalization_method": "none",
                            "normalization_artifact_id": None,
                        },
                    },
                    "validation_lineage": {
                        "master_manifest_sha256": "manifest-sha",
                        "stage4_split_bundle_id": None,
                        "runtime_normalization_method": "none",
                        "runtime_vahadane_backend": None,
                        "normalization_method": "none",
                        "normalization_artifact_id": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = EnsembleInferenceConfig(
        recipe_path=recipe_path,
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=PIPELINE_BATCH_SIZE,
        workers=1,
        seed=24,
        visualization_samples=0,
        export_csv=False,
        export_latex=False,
        export_visualizations=False,
    )
    assert config.output_dir is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.setup_test_data",
        lambda *args, **kwargs: type(
            "_Layout",
            (),
            {
                "master_manifest_path": tmp_path / "master_manifest.sqlite",
            },
        )(),
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.collect_test_dataset_provenance",
        lambda layout: {
            "attrs": {
                "master_manifest_sha256": "other-manifest-sha",
                "stage4_split_bundle_id": None,
                "runtime_normalization_method": "none",
                "runtime_vahadane_backend": None,
                "normalization_method": "none",
                "normalization_artifact_id": None,
            }
        },
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.hash_file_sha256",
        lambda path: (
            "ok-hash" if Path(path) == checkpoint_path or Path(path) == recipe_path else "h5-sha"
        ),
    )

    with pytest.raises(ValueError, match="Dataset lineage mismatch"):
        _execute_pipeline(config)


def test_execute_pipeline_rejects_missing_validation_lineage_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"model-v1")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "recipe_schema_version": 2,
                "calibration_objective": "balanced_rule6",
                "ensemble_strategy": "two_stream_spatial_gating",
                "roi_config": {"threshold": 0.33, "scale": 4},
                "decision_config": {"threshold": 0.57},
                "postprocessing_config": {
                    "method": "threshold_components_patient_suppression",
                    "min_component_area_px": 16,
                    "min_patient_positive_patches": 3,
                    "min_patient_positive_area_fraction": 0.0,
                    "min_component_area_fraction_patch": 0.0,
                },
                "validation_calibration_summary": {"objective": "balanced_rule6"},
                "model_registry": [
                    {
                        "architecture": "SWIN",
                        "encoder": "enc-a",
                        "checkpoint_path": str(checkpoint_path),
                        "stream_role": "semantic",
                        "weight": 1.0,
                        "checkpoint_sha256": "ok-hash",
                    }
                ],
                "provenance": {
                    "validation": {
                        "sha256": "validation-sha",
                        "attrs": {
                            "master_manifest_sha256": "manifest-sha",
                            "stage4_split_bundle_id": None,
                            "runtime_normalization_method": "none",
                            "runtime_vahadane_backend": None,
                            "normalization_method": "none",
                            "normalization_artifact_id": None,
                        },
                    },
                    "validation_lineage": {
                        "master_manifest_sha256": "manifest-sha",
                        "normalization_method": "none",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = EnsembleInferenceConfig(
        recipe_path=recipe_path,
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        batch_size=8,
        workers=1,
        seed=24,
        visualization_samples=0,
        export_csv=False,
        export_latex=False,
        export_visualizations=False,
    )
    assert config.output_dir is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.setup_test_data",
        lambda *args, **kwargs: type(
            "_Layout",
            (),
            {
                "master_manifest_path": tmp_path / "master_manifest.sqlite",
            },
        )(),
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.collect_test_dataset_provenance",
        lambda layout: {
            "attrs": {
                "master_manifest_sha256": "manifest-sha",
                "stage4_split_bundle_id": None,
                "runtime_normalization_method": "none",
                "runtime_vahadane_backend": None,
                "normalization_method": "none",
                "normalization_artifact_id": None,
            }
        },
    )
    monkeypatch.setattr(
        "helpers.ensemble_inference.pipeline.hash_file_sha256",
        lambda path: (
            "ok-hash" if Path(path) == checkpoint_path or Path(path) == recipe_path else "other"
        ),
    )

    with pytest.raises(ValueError, match="validation_lineage missing 'stage4_split_bundle_id'"):
        _execute_pipeline(config)
