from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd

from helpers.crossfold.config import CrossfoldConfig, ObjectiveConfig, SplitConstraints
from helpers.crossfold.discovery import load_patch_dataset
from helpers.crossfold.entropy import (
    calculate_image_entropy_from_hdf5_row,
    calculate_image_entropy_from_path,
    compute_all_patch_entropies,
    compute_patient_entropy_median,
)
from helpers.crossfold.io import _normalize_image_array, write_split_hdf5
from helpers.crossfold.pipeline import run_crossfold_pipeline
from helpers.crossfold.splitting import create_train_val_test_split_best
from helpers.provenance import collect_hdf5_provenance


@dataclass(frozen=True)
class BenchmarkCrossfoldConfig:
    source_hdf5_path: Path
    output_dir: Path
    optuna_trials: int
    hdf5_compression: str
    copy_batch_size: int
    chunksize: int
    entropy_thumbnail: int
    num_workers: int
    random_state: int


def _time_call(label: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return {"label": label, "seconds": elapsed, "result": result}


def _baseline_compute_all_patch_entropies(
    df: pd.DataFrame,
    num_workers: int,
    chunksize: int,
    entropy_thumbnail: int = 128,
) -> pd.DataFrame:
    image_paths = df["image_path"].tolist()
    use_hdf5_rows = {"source_hdf5_path", "source_row_index"}.issubset(df.columns)
    worker = partial(
        calculate_image_entropy_from_hdf5_row
        if use_hdf5_rows
        else calculate_image_entropy_from_path,
        thumb=entropy_thumbnail,
    )
    work_items = (
        [
            (str(row.source_hdf5_path), int(row.source_row_index), str(row.image_path))
            for row in df.itertuples(index=False)
        ]
        if use_hdf5_rows
        else image_paths
    )
    if num_workers <= 1:
        results = [worker(item) for item in work_items]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=num_workers) as pool:
            results = list(pool.imap(worker, work_items, chunksize=chunksize))
    return pd.DataFrame(results, columns=["image_path", "entropy"])


def _reduced_trial_split_search(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
) -> dict[str, Any]:
    return create_train_val_test_split_best(
        df=df,
        random_state=random_state,
        constraints=constraints,
        objective=ObjectiveConfig(
            optuna_trials=max(1, objective.optuna_trials // 4),
            num_workers=objective.num_workers,
            chunksize=objective.chunksize,
            entropy_thumbnail=objective.entropy_thumbnail,
        ),
    )


def _baseline_write_split_hdf5(
    *,
    split_df: pd.DataFrame,
    source_hdf5_path: Path,
    output_path: Path,
    normalization_method: str,
) -> Path:
    if split_df.empty:
        raise ValueError("Cannot benchmark empty split")

    payload = {
        "source_hdf5": collect_hdf5_provenance(source_hdf5_path),
        "normalization_method": normalization_method,
        "rows": split_df[["filename", "patient_id", "label", "source_row_index"]].to_dict(
            "records"
        ),
    }
    source_signature = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_split_df = split_df.reset_index(drop=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with (
        h5py.File(source_hdf5_path, "r") as source_handle,
        h5py.File(output_path, "w") as dest_handle,
    ):
        dest_handle.attrs["source_signature"] = source_signature
        dest_handle.attrs["source_hdf5_sha256"] = collect_hdf5_provenance(source_hdf5_path)[
            "sha256"
        ]
        for attr_name in (
            "upstream_source_signature",
            "stage4_cleaning_manifest_path",
            "stage4_cleaning_manifest_sha256",
            "stage4_cleaning_selected_rows",
        ):
            attr_value = source_handle.attrs.get(attr_name)
            if attr_value is not None:
                dest_handle.attrs[attr_name] = attr_value

        first_index = int(ordered_split_df.iloc[0]["source_row_index"])
        first_image = np.asarray(source_handle["images"][first_index])
        first_mask = np.asarray(source_handle["masks"][first_index])
        images = dest_handle.create_dataset(
            "images",
            shape=(len(ordered_split_df),) + first_image.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        masks = dest_handle.create_dataset(
            "masks",
            shape=(len(ordered_split_df),) + first_mask.shape,
            dtype="uint8",
            compression="gzip",
            chunks=True,
        )
        labels = dest_handle.create_dataset("labels", shape=(len(ordered_split_df),), dtype="uint8")
        patient_ids = dest_handle.create_dataset(
            "patient_ids", shape=(len(ordered_split_df),), dtype="int32"
        )
        filenames = dest_handle.create_dataset(
            "filenames", shape=(len(ordered_split_df),), dtype=str_dtype
        )

        for output_index, row in enumerate(ordered_split_df.itertuples(index=False)):
            source_index = int(row.source_row_index)
            image = np.asarray(source_handle["images"][source_index], dtype=np.uint8)
            mask = np.asarray(source_handle["masks"][source_index], dtype=np.uint8)
            images[output_index] = _normalize_image_array(image, None)
            masks[output_index] = mask
            labels[output_index] = int(row.label)
            patient_ids[output_index] = int(row.patient_id)
            filenames[output_index] = str(row.filename)
    return output_path


def _build_benchmark_config(config: BenchmarkCrossfoldConfig) -> CrossfoldConfig:
    return CrossfoldConfig(
        source_path=config.source_hdf5_path,
        overwrite_output_dir=True,
        random_state=config.random_state,
        constraints=SplitConstraints(
            test_patient_count=20,
            validation_patient_count=20,
        ),
        objective=ObjectiveConfig(
            optuna_trials=config.optuna_trials,
            num_workers=config.num_workers,
            chunksize=config.chunksize,
            entropy_thumbnail=config.entropy_thumbnail,
        ),
        hdf5_compression=config.hdf5_compression,
        copy_batch_size=config.copy_batch_size,
        calc_checksums=False,
        save_entropy_cache_csv=False,
        log_folder=config.output_dir / "logs",
        log_file_name="benchmark.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage 5 crossfold hot paths")
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entropy-subset-rows", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=max(1, (mp.cpu_count() or 1) - 1))
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--entropy-thumbnail", type=int, default=128)
    parser.add_argument("--copy-batch-size", type=int, default=256)
    parser.add_argument("--split-trials-fast", type=int, default=25)
    parser.add_argument("--split-trials-full", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_source_dir = output_dir / "source_link"
    benchmark_source_dir.mkdir(parents=True, exist_ok=True)
    benchmark_source = benchmark_source_dir / args.source_hdf5.name
    benchmark_source.symlink_to(args.source_hdf5)

    dataset = load_patch_dataset(benchmark_source)
    entropy_subset = (
        dataset.head(min(args.entropy_subset_rows, len(dataset))).copy().reset_index(drop=True)
    )

    summary: dict[str, Any] = {
        "source_hdf5": str(args.source_hdf5),
        "benchmark_source_hdf5": str(benchmark_source),
        "source_size_bytes": args.source_hdf5.stat().st_size,
        "dataset_rows": int(len(dataset)),
        "dataset_patients": int(dataset["patient_id"].nunique()),
        "entropy_subset_rows": int(len(entropy_subset)),
        "settings": {
            "num_workers": args.num_workers,
            "chunksize": args.chunksize,
            "entropy_thumbnail": args.entropy_thumbnail,
            "copy_batch_size": args.copy_batch_size,
            "split_trials_fast": args.split_trials_fast,
            "split_trials_full": args.split_trials_full,
        },
    }

    baseline_entropy = _time_call(
        "baseline_entropy_subset",
        _baseline_compute_all_patch_entropies,
        entropy_subset,
        args.num_workers,
        args.chunksize,
        args.entropy_thumbnail,
    )
    current_entropy_subset = _time_call(
        "current_entropy_subset",
        compute_all_patch_entropies,
        entropy_subset,
        args.num_workers,
        args.chunksize,
        args.entropy_thumbnail,
    )
    current_entropy_full = _time_call(
        "current_entropy_full",
        compute_all_patch_entropies,
        dataset,
        args.num_workers,
        args.chunksize,
        args.entropy_thumbnail,
    )
    patient_entropy = _time_call(
        "patient_entropy_full",
        compute_patient_entropy_median,
        dataset,
        current_entropy_full["result"],
    )

    fast_config = _build_benchmark_config(
        BenchmarkCrossfoldConfig(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            optuna_trials=args.split_trials_fast,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        )
    )
    full_config = _build_benchmark_config(
        BenchmarkCrossfoldConfig(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            optuna_trials=args.split_trials_full,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        )
    )

    reduced_trial_split = _time_call(
        "split_search_reduced_trials",
        _reduced_trial_split_search,
        dataset,
        42,
        full_config.constraints,
        full_config.objective,
    )
    current_split = _time_call(
        "split_search_full_trials",
        create_train_val_test_split_best,
        dataset,
        42,
        full_config.constraints,
        full_config.objective,
    )

    split_data = current_split["result"]
    train_df = split_data["train_df"]

    provenance_single = _time_call(
        "collect_hdf5_provenance_once", collect_hdf5_provenance, benchmark_source
    )
    provenance_seven = _time_call(
        "collect_hdf5_provenance_seven_times",
        lambda path: [collect_hdf5_provenance(path) for _ in range(7)],
        benchmark_source,
    )

    baseline_write_path = output_dir / "baseline_train.h5"
    current_write_none_path = output_dir / "current_train_none.h5"
    current_write_gzip_path = output_dir / "current_train_gzip.h5"
    baseline_write = _time_call(
        "baseline_write_train",
        _baseline_write_split_hdf5,
        split_df=train_df,
        source_hdf5_path=benchmark_source,
        output_path=baseline_write_path,
        normalization_method="NOT_NORMALIZED",
    )
    current_write_none = _time_call(
        "current_write_train_none",
        write_split_hdf5,
        split_df=train_df,
        source_hdf5_path=benchmark_source,
        output_path=current_write_none_path,
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        source_hdf5_provenance=provenance_single["result"],
        hdf5_compression="NONE",
        copy_batch_size=args.copy_batch_size,
        overwrite=True,
    )
    current_write_gzip = _time_call(
        "current_write_train_gzip",
        write_split_hdf5,
        split_df=train_df,
        source_hdf5_path=benchmark_source,
        output_path=current_write_gzip_path,
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        source_hdf5_provenance=provenance_single["result"],
        hdf5_compression="GZIP",
        copy_batch_size=args.copy_batch_size,
        overwrite=True,
    )

    full_run_fast = _time_call(
        "full_run_reduced_trials",
        run_crossfold_pipeline,
        fast_config,
    )
    full_run_full = _time_call(
        "full_run_full_trials",
        run_crossfold_pipeline,
        full_config,
    )

    summary["timings_seconds"] = {
        baseline_entropy["label"]: baseline_entropy["seconds"],
        current_entropy_subset["label"]: current_entropy_subset["seconds"],
        current_entropy_full["label"]: current_entropy_full["seconds"],
        patient_entropy["label"]: patient_entropy["seconds"],
        reduced_trial_split["label"]: reduced_trial_split["seconds"],
        current_split["label"]: current_split["seconds"],
        provenance_single["label"]: provenance_single["seconds"],
        provenance_seven["label"]: provenance_seven["seconds"],
        baseline_write["label"]: baseline_write["seconds"],
        current_write_none["label"]: current_write_none["seconds"],
        current_write_gzip["label"]: current_write_gzip["seconds"],
        full_run_fast["label"]: full_run_fast["seconds"],
        full_run_full["label"]: full_run_full["seconds"],
    }
    summary["artifacts"] = {
        "baseline_write_path": str(baseline_write_path),
        "current_write_none_path": str(current_write_none_path),
        "current_write_gzip_path": str(current_write_gzip_path),
        "full_run_reduced_trials_output": str(full_run_fast["result"].output_dir),
        "full_run_full_trials_output": str(full_run_full["result"].output_dir),
    }
    summary["benchmark_config"] = asdict(full_config)
    summary["split_selection"] = {
        "method": "optuna_greedy_sample_ratio_stratified",
        "tie_break_priority": ["TEST", "VALIDATION", "TRAIN"],
        "global_cancer_ratio": split_data["constraints"]["global_cancer_ratio"],
        "optuna_trials": full_config.objective.optuna_trials,
        "final_loss": split_data["objective_score"],
        "verification": split_data["verification"],
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
