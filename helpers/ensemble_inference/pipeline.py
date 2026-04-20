from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.ensemble_inference.config import EnsembleInferenceConfig
from helpers.ensemble_inference.data import (
    collect_test_dataset_provenance,
    create_test_dataloader,
    setup_test_data,
)
from helpers.ensemble_inference.inference import analyze_ensemble_metrics, export_visualizations
from helpers.ensemble_inference.models import load_recipe_models
from helpers.ensemble_inference.recipe import load_recipe_payload, parse_ensemble_recipe
from helpers.ensemble_inference.reporting import (
    export_results_to_csv,
    render_scientific_analysis_report,
    save_confusion_matrix_png,
    write_ensemble_report_latex,
    write_ensemble_report_markdown,
)
from helpers.provenance import (
    collect_runtime_environment,
    hash_file_sha256,
    hash_json_payload,
)
from helpers.training.gpu import GPUNormalizer
from helpers.training.runtime import seed_everything
from helpers.training.utils import get_formatted_datetime_string

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _validate_recipe_dataset_lineage(
    recipe_payload: dict[str, Any],
    test_dataset_provenance: dict[str, Any],
) -> None:
    provenance = recipe_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Recipe provenance mismatch: missing provenance payload.")
    validation_lineage = provenance.get("validation_lineage")
    if not isinstance(validation_lineage, dict):
        raise ValueError("Recipe provenance mismatch: missing validation_lineage payload.")

    observed_attrs = test_dataset_provenance.get("attrs")
    if not isinstance(observed_attrs, dict):
        raise ValueError("Dataset lineage mismatch: test_dataset_provenance is missing attrs.")

    lineage_keys = (
        "master_manifest_sha256",
        "normalization_method",
        "normalization_artifact_id",
    )
    mismatches: list[str] = []
    for key in lineage_keys:
        if key not in validation_lineage:
            raise ValueError(f"Recipe provenance mismatch: validation_lineage missing '{key}'.")
        expected = validation_lineage.get(key)
        observed = observed_attrs.get(key)
        if observed != expected:
            mismatches.append(f"{key}: recipe={expected} test={observed}")
    if mismatches:
        raise ValueError(
            "Dataset lineage mismatch between recipe validation provenance and Stage 11 TEST rows: "
            + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class EnsembleInferenceOutputs:
    output_dir: Path
    run_config_path: Path
    recipe_copy_path: Path
    metrics_json_path: Path
    confusion_matrix_path: Path
    markdown_report_path: Path
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
    payload["master_manifest_path"] = str(config.master_manifest_path)
    payload["output_dir"] = str(config.output_dir) if config.output_dir is not None else None
    payload["local_data_dir"] = str(config.local_data_dir)
    payload["log_folder"] = str(config.log_folder)
    return payload


def _execute_pipeline(config: EnsembleInferenceConfig) -> EnsembleInferenceOutputs:
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_formatted_datetime_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    recipe_payload = load_recipe_payload(config.recipe_path)
    expected_recipe_signature = recipe_payload.get("recipe_signature")
    if isinstance(expected_recipe_signature, str) and expected_recipe_signature.strip():
        observed_recipe_signature = hash_json_payload(
            {key: value for key, value in recipe_payload.items() if key != "recipe_signature"}
        )
        if observed_recipe_signature != expected_recipe_signature:
            raise ValueError("Recipe provenance mismatch: recipe_signature does not match payload.")
    recipe = parse_ensemble_recipe(recipe_payload)
    recipe_copy_path = output_dir / config.recipe_path.name
    if config.recipe_path.resolve() != recipe_copy_path.resolve():
        shutil.copy2(config.recipe_path, recipe_copy_path)
    else:
        recipe_copy_path.write_text(json.dumps(recipe_payload, indent=2), encoding="utf-8")

    test_layout = setup_test_data(
        config.master_manifest_path,
        config.local_data_dir,
        stage_input_locally=config.stage_input_locally,
    )
    observed_checkpoint_hashes: dict[str, str] = {}
    for entry in recipe_payload.get("model_registry", []):
        checkpoint_path = Path(str(entry.get("checkpoint_path", "")).strip())
        expected_checkpoint_hash = entry.get("checkpoint_sha256")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Recipe checkpoint does not exist: {checkpoint_path}")
        observed_hash = hash_file_sha256(checkpoint_path)
        observed_checkpoint_hashes[str(checkpoint_path)] = observed_hash
        if expected_checkpoint_hash is not None and observed_hash != expected_checkpoint_hash:
            raise ValueError(
                f"Checkpoint provenance mismatch for '{checkpoint_path}'. "
                "The on-disk checkpoint content no longer matches the recipe."
            )

    test_dataset_provenance = collect_test_dataset_provenance(test_layout)
    _validate_recipe_dataset_lineage(recipe_payload, test_dataset_provenance)
    test_loader = create_test_dataloader(
        test_layout,
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
        roi_threshold=recipe.roi_threshold,
        decision_threshold=recipe.decision_threshold,
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
    markdown_report_path = write_ensemble_report_markdown(
        ensemble_recipe=recipe_payload,
        ensemble_metrics=metrics,
        train_mean=IMAGENET_MEAN,
        train_std=IMAGENET_STD,
        output_dir=output_dir,
        timestamp=timestamp,
    )

    csv_report_path: Path | None = None
    if config.export_csv:
        csv_report_path = export_results_to_csv(
            ensemble_recipe=recipe_payload,
            metrics_results=metrics,
            output_dir=output_dir,
            timestamp=timestamp,
            recipe_path=config.recipe_path,
            dataset_dir=config.master_manifest_path,
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
            roi_threshold=recipe.roi_threshold,
            decision_threshold=recipe.decision_threshold,
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
            "test_master_manifest_path": str(test_layout.master_manifest_path),
            "test_dataset_provenance": test_dataset_provenance,
            "roi_threshold": recipe.roi_threshold,
            "decision_threshold": recipe.decision_threshold,
            "output_dir": str(output_dir),
            "recipe_copy_path": str(recipe_copy_path),
            "recipe_sha256": hash_file_sha256(recipe_copy_path),
            "observed_checkpoint_hashes": observed_checkpoint_hashes,
            "metrics_json_path": str(metrics_json_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "markdown_report_path": str(markdown_report_path),
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
        markdown_report_path=markdown_report_path,
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
