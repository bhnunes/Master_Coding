from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

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
from helpers.crossfold.splitting import (
    build_patient_table,
    create_train_val_test_split_best,
    decide_split_sizes,
)
from helpers.provenance import collect_hdf5_provenance


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


def _baseline_validation_supports_stage11(
    patient_df: pd.DataFrame,
    val_patients: set[int],
    constraints: SplitConstraints,
    *,
    dataset_has_both_classes: bool,
) -> bool:
    if not constraints.enforce_stage11_validation_sizing:
        return True

    validation_rows = patient_df[patient_df["patient_id"].isin(sorted(val_patients))]
    if len(validation_rows) < constraints.min_validation_patients_for_ensemble:
        return False
    if not dataset_has_both_classes:
        return True

    positive_count = int((validation_rows["patient_label"] == 1).sum())
    negative_count = int((validation_rows["patient_label"] == 0).sum())
    return (
        positive_count >= constraints.min_validation_positive_patients_for_ensemble
        and negative_count >= constraints.min_validation_negative_patients_for_ensemble
    )


def _baseline_score_split_by_patient_entropy_median(
    patient_entropy_df: pd.DataFrame,
    patient_ids: list[int],
) -> float:
    subset = patient_entropy_df[patient_entropy_df["patient_id"].isin(patient_ids)]
    if subset.empty:
        return float("-inf")
    return float(subset["patient_entropy_median"].median())


def _baseline_create_train_val_test_split_best(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
    patient_entropy_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    patient_df = build_patient_table(df)
    _, _, n_test, sizing_meta = decide_split_sizes(patient_df, constraints)
    patient_ids = patient_df["patient_id"].to_numpy()
    labels = patient_df["patient_label"].astype(int).to_numpy()
    dataset_has_both_classes = patient_df["patient_label"].nunique() >= 2
    rng = np.random.default_rng(random_state)

    def image_count(patient_ids_subset: set[int]) -> int:
        return int(
            patient_df[patient_df["patient_id"].isin(sorted(patient_ids_subset))]["n_images"].sum()
        )

    best: tuple[set[int], set[int], set[int], int, int] | None = None
    best_score: float | None = None

    for attempt in range(constraints.max_tries):
        seed = int(rng.integers(0, 2**31 - 1))
        test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_test,
            random_state=seed,
        )
        trainval_index, test_index = next(test_splitter.split(patient_ids, labels))
        trainval_ids = patient_ids[trainval_index]
        trainval_labels = labels[trainval_index]
        try:
            validation_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=sizing_meta["n_val"],
                random_state=seed + 1,
            )
            train_index, val_index = next(validation_splitter.split(trainval_ids, trainval_labels))
        except ValueError:
            continue

        train_patients = set(trainval_ids[train_index])
        val_patients = set(trainval_ids[val_index])
        test_patients = set(patient_ids[test_index])
        if constraints.require_both_classes_if_possible and dataset_has_both_classes:

            def split_has_both(patient_subset: set[int]) -> bool:
                split_labels = set(
                    patient_df[patient_df["patient_id"].isin(sorted(patient_subset))][
                        "patient_label"
                    ].tolist()
                )
                return 0 in split_labels and 1 in split_labels

            if not (
                split_has_both(train_patients)
                and split_has_both(val_patients)
                and split_has_both(test_patients)
            ):
                continue
        if not _baseline_validation_supports_stage11(
            patient_df,
            val_patients,
            constraints,
            dataset_has_both_classes=dataset_has_both_classes,
        ):
            continue
        train_images = image_count(train_patients)
        val_images = image_count(val_patients)
        test_images = image_count(test_patients)
        if constraints.require_train_image_dominance and not (
            train_images > val_images and train_images > test_images
        ):
            continue
        if not objective.enable_objective:
            return {
                "train_patients": sorted(train_patients),
                "val_patients": sorted(val_patients),
                "test_patients": sorted(test_patients),
                "split_seed": seed,
                "split_attempt": attempt + 1,
            }
        assert patient_entropy_df is not None
        score_split = objective.score_split.upper()
        score_ids = (
            sorted(train_patients)
            if score_split == "TRAIN"
            else sorted(val_patients)
            if score_split == "VALIDATION"
            else sorted(test_patients)
        )
        score = _baseline_score_split_by_patient_entropy_median(patient_entropy_df, score_ids)
        if best is None or best_score is None:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
        elif objective.maximize and score > best_score:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
        elif (not objective.maximize) and score < best_score:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
    if best is None:
        raise ValueError("No feasible split found in baseline benchmark")
    return {
        "train_patients": sorted(best[0]),
        "val_patients": sorted(best[1]),
        "test_patients": sorted(best[2]),
        "split_seed": best[3],
        "split_attempt": best[4],
        "objective_score": best_score,
    }


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


def _build_benchmark_config(
    *,
    source_hdf5_path: Path,
    output_dir: Path,
    enable_objective: bool,
    hdf5_compression: str,
    copy_batch_size: int,
    chunksize: int,
    entropy_thumbnail: int,
    num_workers: int,
    random_state: int,
) -> CrossfoldConfig:
    return CrossfoldConfig(
        normalization_method="NOT_NORMALIZED",
        source_hdf5_path=source_hdf5_path,
        overwrite_output_dir=True,
        random_state=random_state,
        constraints=SplitConstraints(
            min_test_patients=2,
            min_val_patients=2,
            min_train_patients=2,
            enforce_stage11_validation_sizing=False,
            test_ratio=0.20,
            val_ratio=0.20,
            require_train_image_dominance=False,
            require_both_classes_if_possible=False,
            max_tries=1000,
            adaptive=True,
        ),
        objective=ObjectiveConfig(
            enable_objective=enable_objective,
            score_split="TRAIN",
            maximize=True,
            num_workers=num_workers,
            chunksize=chunksize,
            entropy_thumbnail=entropy_thumbnail,
        ),
        hdf5_compression=hdf5_compression,
        copy_batch_size=copy_batch_size,
        calc_checksums=False,
        save_entropy_cache_csv=False,
        log_folder=output_dir / "logs",
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

    baseline_split = _time_call(
        "baseline_split_search",
        _baseline_create_train_val_test_split_best,
        dataset,
        42,
        _build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).constraints,
        _build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).objective,
        patient_entropy["result"],
    )
    current_split = _time_call(
        "current_split_search",
        create_train_val_test_split_best,
        dataset,
        42,
        _build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).constraints,
        _build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).objective,
        patient_entropy["result"],
    )

    split_data = create_train_val_test_split_best(
        df=dataset,
        random_state=42,
        constraints=_build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).constraints,
        objective=_build_benchmark_config(
            source_hdf5_path=benchmark_source,
            output_dir=output_dir,
            enable_objective=True,
            hdf5_compression="NONE",
            copy_batch_size=args.copy_batch_size,
            chunksize=args.chunksize,
            entropy_thumbnail=args.entropy_thumbnail,
            num_workers=args.num_workers,
            random_state=42,
        ).objective,
        patient_entropy_df=patient_entropy["result"],
    )
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

    full_config_objective_on = _build_benchmark_config(
        source_hdf5_path=benchmark_source,
        output_dir=output_dir / "full_objective_on",
        enable_objective=True,
        hdf5_compression="NONE",
        copy_batch_size=args.copy_batch_size,
        chunksize=args.chunksize,
        entropy_thumbnail=args.entropy_thumbnail,
        num_workers=args.num_workers,
        random_state=42,
    )
    full_config_objective_off = _build_benchmark_config(
        source_hdf5_path=benchmark_source,
        output_dir=output_dir / "full_objective_off",
        enable_objective=False,
        hdf5_compression="NONE",
        copy_batch_size=args.copy_batch_size,
        chunksize=args.chunksize,
        entropy_thumbnail=args.entropy_thumbnail,
        num_workers=args.num_workers,
        random_state=43,
    )
    full_run_objective_on = _time_call(
        "full_run_objective_on",
        run_crossfold_pipeline,
        full_config_objective_on,
    )
    full_run_objective_off = _time_call(
        "full_run_objective_off",
        run_crossfold_pipeline,
        full_config_objective_off,
    )

    summary["timings_seconds"] = {
        baseline_entropy["label"]: baseline_entropy["seconds"],
        current_entropy_subset["label"]: current_entropy_subset["seconds"],
        current_entropy_full["label"]: current_entropy_full["seconds"],
        patient_entropy["label"]: patient_entropy["seconds"],
        baseline_split["label"]: baseline_split["seconds"],
        current_split["label"]: current_split["seconds"],
        provenance_single["label"]: provenance_single["seconds"],
        provenance_seven["label"]: provenance_seven["seconds"],
        baseline_write["label"]: baseline_write["seconds"],
        current_write_none["label"]: current_write_none["seconds"],
        current_write_gzip["label"]: current_write_gzip["seconds"],
        full_run_objective_on["label"]: full_run_objective_on["seconds"],
        full_run_objective_off["label"]: full_run_objective_off["seconds"],
    }
    summary["artifacts"] = {
        "baseline_write_path": str(baseline_write_path),
        "current_write_none_path": str(current_write_none_path),
        "current_write_gzip_path": str(current_write_gzip_path),
        "full_objective_on_output": str(full_run_objective_on["result"].output_dir),
        "full_objective_off_output": str(full_run_objective_off["result"].output_dir),
    }
    summary["config_objective_on"] = asdict(full_config_objective_on)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
