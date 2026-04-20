from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from helpers.crossfold.config import CrossfoldConfig
from helpers.crossfold.discovery import collect_source_dataset_provenance, load_patch_dataset
from helpers.crossfold.entropy import compute_all_patch_entropies
from helpers.crossfold.logging import configure_crossfold_logging
from helpers.crossfold.normalization import fit_normalizer_on_train_set, save_normalizer_stats
from helpers.crossfold.provenance import (
    build_hdf5_manifest_from_split_dfs,
    write_manifest_and_log_stats,
)
from helpers.crossfold.splitting import create_train_val_test_split_best
from helpers.extraction.master_manifest import STAGE5_STAGE_NAME, MasterManifest


@dataclass(frozen=True)
class CrossfoldRunSummary:
    output_dir: Path
    manifest_rows: int


def run_crossfold_pipeline(config: CrossfoldConfig) -> CrossfoldRunSummary:
    """Run the full Stage 5 split-generation and optional normalization workflow."""

    if config.source_path.suffix.lower() != ".sqlite":
        raise ValueError(
            "Native Stage 5 now requires CROSSFOLD_SOURCE_HDF5_PATH to point to "
            "master_manifest.sqlite."
        )

    output_dir = config.output_run_dir
    if output_dir.exists():
        if config.overwrite_output_dir:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Output directory exists: {output_dir} (set overwrite_output_dir=True)"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_crossfold_logging(config.log_path)
    logging.info("=== Data Preparation (Refactored) ===")
    logging.info("Normalization method: %s", config.normalization_method)
    logging.info("Random state: %s", config.random_state)
    logging.info("Constraints: %s", asdict(config.constraints))
    logging.info("Objective: %s", asdict(config.objective))
    logging.info("Output: %s", output_dir)

    source_dataset_provenance = collect_source_dataset_provenance(config.source_path)
    dataset = load_patch_dataset(config.source_path)
    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=config.random_state,
        constraints=config.constraints,
        objective=config.objective,
    )
    split_data["constraints"]["random_state"] = config.random_state
    run_id = f"{config.normalization_method}_seed_{config.random_state}"
    manifest_df = build_hdf5_manifest_from_split_dfs(
        output_dir=output_dir,
        run_id=run_id,
        normalization_method=config.normalization_method,
        is_normalized=(config.normalization_method != "NOT_NORMALIZED"),
        split_data=split_data,
    )

    normalizer = None
    template_paths: list[str] | None = None
    entropy_df: pd.DataFrame | None = None
    if config.normalization_method != "NOT_NORMALIZED":
        entropy_df = compute_all_patch_entropies(
            df=split_data["train_df"],
            num_workers=config.objective.num_workers,
            chunksize=config.objective.chunksize,
            entropy_thumbnail=config.objective.entropy_thumbnail,
        )
        if config.save_entropy_cache_csv:
            entropy_df.to_csv(output_dir / "entropy_cache.csv", index=False)
            logging.info("Saved entropy cache: %s", output_dir / "entropy_cache.csv")
        normalizer, template_paths = fit_normalizer_on_train_set(
            split_data["train_df"],
            config.normalization_method,
            entropy_df=entropy_df,
        )
        save_normalizer_stats(normalizer, config.normalization_method, output_dir, template_paths)

    split_frames = {
        "TRAIN": split_data["train_df"],
        "VALIDATION": split_data["val_df"],
        "TEST": split_data["test_df"],
    }

    extra: dict[str, object] = {}
    extra["split_selection"] = {
        "method": "optuna_greedy_sample_ratio_stratified",
        "tie_break_priority": ["TEST", "VALIDATION", "TRAIN"],
        "global_cancer_ratio": split_data["constraints"].get("global_cancer_ratio"),
        "optuna_trials": config.objective.optuna_trials,
        "loss_metric": "sum_absolute_split_ratio_delta",
        "final_loss": split_data.get("objective_score"),
    }
    if "verification" in split_data:
        extra["verification"] = split_data["verification"]
    write_manifest_and_log_stats(
        output_dir=output_dir,
        run_id=run_id,
        normalization_method=config.normalization_method,
        is_normalized=(config.normalization_method != "NOT_NORMALIZED"),
        source_hdf5_path=config.source_path,
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=config.calc_checksums,
        source_hdf5_provenance=source_dataset_provenance,
        extra=extra,
    )
    _persist_stage5_split_state(
        master_manifest_path=config.source_path,
        split_frames=split_frames,
        normalization_method=config.normalization_method,
        output_dir=output_dir,
    )
    logging.info("=== DONE ===")
    return CrossfoldRunSummary(
        output_dir=output_dir,
        manifest_rows=len(manifest_df),
    )


def _persist_stage5_split_state(
    *,
    master_manifest_path: Path,
    split_frames: dict[str, pd.DataFrame],
    normalization_method: str,
    output_dir: Path,
) -> None:
    master_manifest = MasterManifest(master_manifest_path)
    run_id = master_manifest.create_run(
        stage_name=STAGE5_STAGE_NAME,
        config_path=output_dir / "run_config.json",
    )
    normalization_artifact_id: int | None = None
    if normalization_method != "NOT_NORMALIZED":
        normalization_artifact_id = master_manifest.create_normalization_artifact(
            run_id=run_id,
            method=normalization_method,
            state_path=output_dir / "normalization_stats.json",
            template_path=output_dir / "normalization_templates",
            fit_scope="TRAIN",
        )

    assignments: list[dict[str, object]] = []
    for split_name, split_df in split_frames.items():
        if split_df.empty:
            continue
        for row in split_df.itertuples(index=False):
            assignments.append(
                {
                    "split": split_name,
                    "filename": row.filename,
                    "patient_id": int(row.patient_id),
                    "label": int(row.label),
                    "source_hdf5_path": str(row.source_hdf5_path),
                    "source_row_index": int(row.source_row_index),
                }
            )
    master_manifest.update_stage5_split_assignments(
        assignments=assignments,
        normalization_method=normalization_method,
        normalization_artifact_id=normalization_artifact_id,
    )
