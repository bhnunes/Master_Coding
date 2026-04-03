from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from helpers.crossfold.config import CrossfoldConfig
from helpers.crossfold.discovery import load_patch_dataset
from helpers.crossfold.entropy import compute_all_patch_entropies, compute_patient_entropy_median
from helpers.crossfold.io import verify_split_hdf5_integrity, write_split_hdf5
from helpers.crossfold.logging import configure_crossfold_logging
from helpers.crossfold.normalization import fit_normalizer_on_train_set, save_normalizer_stats
from helpers.crossfold.provenance import (
    build_hdf5_manifest_from_split_dfs,
    write_manifest_and_log_stats,
)
from helpers.crossfold.splitting import create_train_val_test_split_best
from helpers.provenance import collect_hdf5_provenance


@dataclass(frozen=True)
class CrossfoldRunSummary:
    output_dir: Path
    manifest_rows: int


def run_crossfold_pipeline(config: CrossfoldConfig) -> CrossfoldRunSummary:
    """Run the full Stage 5 split-generation and optional normalization workflow."""

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

    source_hdf5_provenance = collect_hdf5_provenance(config.source_hdf5_path)
    dataset = load_patch_dataset(config.source_hdf5_path)
    entropy_df: pd.DataFrame | None = None
    patient_entropy_df: pd.DataFrame | None = None
    if config.objective.enable_objective:
        entropy_df = compute_all_patch_entropies(
            df=dataset,
            num_workers=config.objective.num_workers,
            chunksize=config.objective.chunksize,
            entropy_thumbnail=config.objective.entropy_thumbnail,
        )
        patient_entropy_df = compute_patient_entropy_median(dataset, entropy_df)
        if config.save_entropy_cache_csv:
            entropy_df.to_csv(output_dir / "entropy_cache.csv", index=False)
            patient_entropy_df.to_csv(output_dir / "patient_entropy_median.csv", index=False)
            logging.info("Saved entropy cache: %s", output_dir / "entropy_cache.csv")

    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=config.random_state,
        constraints=config.constraints,
        objective=config.objective,
        patient_entropy_df=patient_entropy_df,
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
    if config.normalization_method != "NOT_NORMALIZED":
        normalizer, template_paths = fit_normalizer_on_train_set(
            split_data["train_df"],
            config.normalization_method,
            entropy_df=entropy_df,
        )
        save_normalizer_stats(normalizer, config.normalization_method, output_dir, template_paths)

    for split_name, split_df in (
        ("TRAIN", split_data["train_df"]),
        ("VALIDATION", split_data["val_df"]),
        ("TEST", split_data["test_df"]),
    ):
        if split_df.empty:
            continue
        output_path = write_split_hdf5(
            split_df=split_df,
            source_hdf5_path=config.source_hdf5_path,
            output_path=output_dir / f"{split_name}.h5",
            normalizer=normalizer,
            normalization_method=config.normalization_method,
            source_hdf5_provenance=source_hdf5_provenance,
            hdf5_compression=config.hdf5_compression,
            copy_batch_size=config.copy_batch_size,
            overwrite=True,
        )
        verify_split_hdf5_integrity(output_path, split_df)

    extra: dict[str, object] = {}
    if patient_entropy_df is not None and split_data.get("objective_score") is not None:
        extra["objective_definition"] = {
            "metric": config.objective.metric,
            "patient_entropy": "median(patch_entropy)",
            "split_score": (
                f"median(patient_entropy) over {config.objective.score_split.upper()} patients"
            ),
            "maximize": config.objective.maximize,
        }
    write_manifest_and_log_stats(
        output_dir=output_dir,
        run_id=run_id,
        normalization_method=config.normalization_method,
        is_normalized=(config.normalization_method != "NOT_NORMALIZED"),
        source_hdf5_path=config.source_hdf5_path,
        split_data=split_data,
        manifest_df=manifest_df,
        calc_checksums=config.calc_checksums,
        source_hdf5_provenance=source_hdf5_provenance,
        extra=extra,
    )
    logging.info("=== DONE ===")
    return CrossfoldRunSummary(
        output_dir=output_dir,
        manifest_rows=len(manifest_df),
    )
