from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from helpers.lr_finder.config import LRFinderConfig
from helpers.lr_finder.reporting import build_latex_report
from helpers.lr_finder.runner import ScreeningOutputs, run_lr_finder_screening
from helpers.provenance import collect_runtime_environment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LRFinderOutputs:
    tex_path: Path
    pdf_path: Path
    run_config_path: Path
    summary_all_path: Path
    lhs_samples_path: Path
    valid_records: int
    completed_trials: int
    failed_trials: int
    architecture_trial_stats: dict[str, dict[str, int]]


def _prepare_output_dir(config: LRFinderConfig) -> None:
    if config.output_dir.exists() and config.overwrite_output:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _serialize_config(config: LRFinderConfig) -> dict[str, Any]:
    return {
        "master_manifest_path": str(config.master_manifest_path),
        "output_dir": str(config.output_dir),
        "local_data_dir": str(config.local_data_dir),
        "stain_matrix_cache_path": (
            str(config.stain_matrix_cache_path)
            if config.stain_matrix_cache_path is not None
            else None
        ),
        "stage_input_locally": config.stage_input_locally,
        "overwrite_output": config.overwrite_output,
        "smart_sampling": config.smart_sampling,
        "execution_mode": config.execution_mode,
        "amp_precision": config.amp_precision,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "use_subset": config.use_subset,
        "subset_ratio": config.subset_ratio,
        "num_lhs_samples": config.num_lhs_samples,
        "end_lr": config.end_lr,
        "num_iter": config.num_iter,
        "num_repeats": config.num_repeats,
        "runtime_normalization_method": config.runtime_normalization_method,
        "runtime_vahadane_backend": config.runtime_vahadane_backend,
        "optimizer_weight_decay": config.optimizer_weight_decay,
        "optimizer_start_lr": config.optimizer_start_lr,
        "pdf_name": config.pdf_name,
        "search_space": asdict(config.search_space),
        "model_plans": [asdict(plan) for plan in config.model_plans],
    }


def run_lr_finder_pipeline(
    config: LRFinderConfig,
    *,
    screening_runner: Callable[[LRFinderConfig], ScreeningOutputs] = run_lr_finder_screening,
    report_builder: Callable[..., tuple[Path, Path]] = build_latex_report,
) -> LRFinderOutputs:
    logger.info("Preparing LR finder output directory at %s", config.output_dir)
    _prepare_output_dir(config)
    logger.info("Starting LR finder screening")
    screening_outputs = screening_runner(config)
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_mode": config.execution_mode,
        "amp_precision": config.amp_precision,
        "completed_trials": str(screening_outputs.completed_trials),
        "failed_trials": str(screening_outputs.failed_trials),
    }
    logger.info(
        "Building LR finder report with %s valid records (%s failed trials)",
        len(screening_outputs.records),
        screening_outputs.failed_trials,
    )
    tex_path, pdf_path = report_builder(
        screening_outputs.records,
        config.output_dir,
        meta,
        config.pdf_name,
    )
    run_config_path = config.output_dir / "lr_finder_run_config.json"
    logger.info("Writing LR finder run config to %s", run_config_path)
    run_config_payload = _serialize_config(config)
    run_config_payload.update(
        {
            "valid_records": len(screening_outputs.records),
            "completed_trials": screening_outputs.completed_trials,
            "failed_trials": screening_outputs.failed_trials,
            "architecture_trial_stats": screening_outputs.architecture_trial_stats,
            "source_split_name": screening_outputs.source_split_name,
            "training_dataset_provenance": screening_outputs.training_dataset_provenance,
            "validation_dataset_provenance": screening_outputs.validation_dataset_provenance,
            "generated_at": meta["timestamp"],
            "runtime_environment": collect_runtime_environment(),
            "summary_all_path": str(screening_outputs.summary_all_path),
            "lhs_samples_path": str(screening_outputs.lhs_samples_path),
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
        }
    )
    run_config_path.write_text(json.dumps(run_config_payload, indent=2), encoding="utf-8")
    return LRFinderOutputs(
        tex_path=tex_path,
        pdf_path=pdf_path,
        run_config_path=run_config_path,
        summary_all_path=screening_outputs.summary_all_path,
        lhs_samples_path=screening_outputs.lhs_samples_path,
        valid_records=len(screening_outputs.records),
        completed_trials=screening_outputs.completed_trials,
        failed_trials=screening_outputs.failed_trials,
        architecture_trial_stats=screening_outputs.architecture_trial_stats,
    )
