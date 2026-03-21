from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.ensemble_inference.config import EnsembleInferenceConfig
from helpers.ensemble_inference.data import create_test_dataloader, setup_test_hdf5
from helpers.ensemble_inference.inference import analyze_ensemble_metrics, export_visualizations
from helpers.ensemble_inference.models import load_recipe_models
from helpers.ensemble_inference.recipe import load_recipe_payload, parse_ensemble_recipe
from helpers.ensemble_inference.reporting import (
    export_results_to_csv,
    render_scientific_analysis_report,
    save_confusion_matrix_png,
    write_ensemble_report_latex,
)
from helpers.provenance import collect_runtime_environment
from helpers.training.gpu import GPUNormalizer
from helpers.training.runtime import seed_everything
from helpers.training.utils import get_formatted_datetime_string

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class EnsembleInferenceOutputs:
    output_dir: Path
    run_config_path: Path
    recipe_copy_path: Path
    metrics_json_path: Path
    confusion_matrix_path: Path
    csv_report_path: Path | None
    latex_report_path: Path | None
    pdf_report_path: Path | None


def _resolve_output_dir(config: EnsembleInferenceConfig) -> Path:
    return config.output_dir or config.recipe_path.parent


def _prepare_output_dir(config: EnsembleInferenceConfig) -> Path:
    output_dir = _resolve_output_dir(config)
    if config.output_dir is not None and output_dir.exists() and config.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _serialize_config(config: EnsembleInferenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["recipe_path"] = str(config.recipe_path)
    payload["hdf5_drive_dir"] = str(config.hdf5_drive_dir)
    payload["output_dir"] = str(config.output_dir) if config.output_dir is not None else None
    payload["local_data_dir"] = str(config.local_data_dir)
    payload["log_folder"] = str(config.log_folder)
    return payload


def _execute_pipeline(config: EnsembleInferenceConfig) -> EnsembleInferenceOutputs:
    output_dir = _resolve_output_dir(config)
    timestamp = get_formatted_datetime_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    recipe_payload = load_recipe_payload(config.recipe_path)
    recipe = parse_ensemble_recipe(recipe_payload)
    recipe_copy_path = output_dir / config.recipe_path.name
    if config.recipe_path.resolve() != recipe_copy_path.resolve():
        shutil.copy2(config.recipe_path, recipe_copy_path)
    else:
        recipe_copy_path.write_text(json.dumps(recipe_payload, indent=2), encoding="utf-8")

    test_h5_path = setup_test_hdf5(
        config.hdf5_drive_dir,
        config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    test_loader = create_test_dataloader(
        test_h5_path,
        batch_size=config.batch_size,
        workers=config.workers,
    )
    models, constituent_info = load_recipe_models(recipe, device)
    gpu_normalizer = GPUNormalizer(IMAGENET_MEAN, IMAGENET_STD, device)
    metrics = analyze_ensemble_metrics(
        models,
        constituent_info,
        test_loader,
        device=device,
        optimal_threshold=recipe.roi_threshold,
        roi_scale=recipe.roi_scale,
        train_mean=IMAGENET_MEAN,
        train_std=IMAGENET_STD,
        gpu_normalizer=gpu_normalizer,
        seed=config.seed,
    )

    print(render_scientific_analysis_report(metrics))
    metrics_json_path = output_dir / f"ENSEMBLE_METRICS_{timestamp}.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion = metrics.get("confusion_matrix", {})
    confusion_matrix_path = save_confusion_matrix_png(
        int(confusion.get("tp", 0)),
        int(confusion.get("fp", 0)),
        int(confusion.get("fn", 0)),
        int(confusion.get("tn", 0)),
        output_dir / "confusion_matrix.png",
    )

    csv_report_path: Path | None = None
    if config.export_csv:
        csv_report_path = export_results_to_csv(
            ensemble_recipe=recipe_payload,
            metrics_results=metrics,
            output_dir=output_dir,
            timestamp=timestamp,
            recipe_path=config.recipe_path,
            dataset_dir=config.hdf5_drive_dir,
            seed=config.seed,
            batch_size=config.batch_size,
        )

    latex_report_path: Path | None = None
    pdf_report_path: Path | None = None
    if config.export_latex:
        latex_report_path, pdf_report_path = write_ensemble_report_latex(
            ensemble_recipe=recipe_payload,
            ensemble_metrics=metrics,
            train_mean=IMAGENET_MEAN,
            train_std=IMAGENET_STD,
            cm_png_path=confusion_matrix_path,
            output_dir=output_dir,
            timestamp=timestamp,
        )

    if config.export_visualizations and config.visualization_samples > 0:
        export_visualizations(
            models,
            test_loader,
            device=device,
            threshold=recipe.roi_threshold,
            roi_scale=recipe.roi_scale,
            train_mean=IMAGENET_MEAN,
            train_std=IMAGENET_STD,
            constituent_models_info=constituent_info,
            gpu_normalizer=gpu_normalizer,
            output_dir=output_dir / "visualizations",
            num_samples=config.visualization_samples,
        )

    run_config_path = output_dir / "ensemble_inference_run_config.json"
    run_payload = _serialize_config(config)
    run_payload.update(
        {
            "generated_at": timestamp,
            "runtime_environment": collect_runtime_environment(),
            "test_h5_path": str(test_h5_path),
            "output_dir": str(output_dir),
            "recipe_copy_path": str(recipe_copy_path),
            "metrics_json_path": str(metrics_json_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "csv_report_path": str(csv_report_path) if csv_report_path is not None else None,
            "latex_report_path": str(latex_report_path) if latex_report_path is not None else None,
            "pdf_report_path": str(pdf_report_path) if pdf_report_path is not None else None,
        }
    )
    run_config_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    return EnsembleInferenceOutputs(
        output_dir=output_dir,
        run_config_path=run_config_path,
        recipe_copy_path=recipe_copy_path,
        metrics_json_path=metrics_json_path,
        confusion_matrix_path=confusion_matrix_path,
        csv_report_path=csv_report_path,
        latex_report_path=latex_report_path,
        pdf_report_path=pdf_report_path,
    )


def run_ensemble_inference_pipeline(
    config: EnsembleInferenceConfig,
    *,
    pipeline_runner: Callable[
        [EnsembleInferenceConfig], EnsembleInferenceOutputs
    ] = _execute_pipeline,
) -> EnsembleInferenceOutputs:
    output_dir = _prepare_output_dir(config)
    outputs = pipeline_runner(config)
    if outputs.run_config_path.exists():
        return outputs
    payload = _serialize_config(config)
    payload["output_dir"] = str(output_dir)
    payload["runtime_environment"] = collect_runtime_environment()
    outputs.run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return outputs
