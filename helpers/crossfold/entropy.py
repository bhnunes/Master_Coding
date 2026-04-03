from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from helpers.crossfold.logging import ProgressReporter


def calculate_image_entropy_from_path(image_path: str, thumb: int = 128) -> tuple[str, float]:
    """Compute fast grayscale Shannon entropy from a PNG path."""

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return image_path, 0.0
        if thumb is not None:
            image = cv2.resize(image, (thumb, thumb), interpolation=cv2.INTER_AREA)
        histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
        total = histogram.sum()
        if total <= 0:
            return image_path, 0.0
        probabilities = histogram / total
        probabilities = probabilities[probabilities > 0]
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        return image_path, entropy
    except Exception:
        return image_path, 0.0


def _calculate_entropy_from_array(image: npt.NDArray[np.uint8], thumb: int = 128) -> float:
    if image.ndim == 3:
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    if thumb is not None:
        image = np.asarray(
            cv2.resize(image, (thumb, thumb), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def calculate_image_entropy_from_hdf5_row(
    payload: tuple[str, int, str],
    thumb: int = 128,
) -> tuple[str, float]:
    source_hdf5_path, row_index, image_key = payload
    try:
        with h5py.File(Path(source_hdf5_path), "r") as handle:
            image_dataset: Any = handle["images"]
            image = np.asarray(image_dataset[row_index], dtype=np.uint8)
        return image_key, _calculate_entropy_from_array(image, thumb=thumb)
    except Exception:
        return image_key, 0.0


def compute_all_patch_entropies(
    df: pd.DataFrame,
    num_workers: int,
    chunksize: int,
    entropy_thumbnail: int = 128,
) -> pd.DataFrame:
    """Compute entropy for each image path in the dataset."""

    image_paths = df["image_path"].tolist()
    logging.info(
        "Computing patch entropies for %s images using num_workers=%s, thumb=%s...",
        len(image_paths),
        num_workers,
        entropy_thumbnail,
    )
    use_hdf5_rows = {"source_hdf5_path", "source_row_index"}.issubset(df.columns)
    if use_hdf5_rows:
        return _compute_all_patch_entropies_from_hdf5(
            df,
            entropy_thumbnail=entropy_thumbnail,
            read_batch_size=chunksize,
        )

    worker = partial(calculate_image_entropy_from_path, thumb=entropy_thumbnail)
    work_items = image_paths
    if num_workers <= 1:
        results = [worker(item) for item in tqdm(work_items, desc="Entropy", leave=False)]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, work_items, chunksize=chunksize),
                    total=len(work_items),
                    desc="Entropy",
                    leave=False,
                )
            )
    return pd.DataFrame(
        {
            "image_path": [result[0] for result in results],
            "entropy": [result[1] for result in results],
        }
    )


def _compute_all_patch_entropies_from_hdf5(
    df: pd.DataFrame,
    *,
    entropy_thumbnail: int,
    read_batch_size: int,
) -> pd.DataFrame:
    indexed = df.reset_index(drop=True).copy()
    indexed["_entropy_position"] = np.arange(len(indexed), dtype=np.int64)
    results: list[tuple[str, float] | None] = [None] * len(indexed)

    grouped = indexed.groupby("source_hdf5_path", sort=False)
    total_rows = len(indexed)
    reporter = ProgressReporter("Stage 5 entropy", total_rows, "rows")
    logging.info(
        "Computing HDF5-backed patch entropies with batched reads from %s source file(s) "
        "using thumb=%s and read_batch_size=%s...",
        grouped.ngroups,
        entropy_thumbnail,
        read_batch_size,
    )
    reporter.log_start(
        f"source_files={grouped.ngroups} | total_rows={total_rows} | thumb={entropy_thumbnail}"
    )
    processed_rows = 0
    for source_hdf5_path, group_df in tqdm(grouped, desc="Entropy", leave=False):
        sorted_group = group_df.sort_values("source_row_index", kind="mergesort").reset_index(
            drop=True
        )
        with h5py.File(Path(str(source_hdf5_path)), "r") as handle:
            image_dataset: Any = handle["images"]
            offset = 0
            while offset < len(sorted_group):
                batch_stop = min(offset + read_batch_size, len(sorted_group))
                batch_df = sorted_group.iloc[offset:batch_stop]
                _compute_entropy_batch(
                    image_dataset=image_dataset,
                    batch_df=batch_df,
                    entropy_thumbnail=entropy_thumbnail,
                    results=results,
                )
                offset = batch_stop
                processed_rows += len(batch_df)
                reporter.log(
                    completed_units=processed_rows,
                    extra_parts=[
                        f"file={Path(str(source_hdf5_path)).name}",
                        f"remaining={total_rows - processed_rows} rows",
                    ],
                )

    finalized = [result for result in results if result is not None]
    return pd.DataFrame(
        {
            "image_path": [result[0] for result in finalized],
            "entropy": [result[1] for result in finalized],
        }
    )


def _compute_entropy_batch(
    *,
    image_dataset: Any,
    batch_df: pd.DataFrame,
    entropy_thumbnail: int,
    results: list[tuple[str, float] | None],
) -> None:
    pending_rows: list[dict[str, Any]] = []
    span_start: int | None = None
    span_end: int | None = None

    def flush_span() -> None:
        nonlocal pending_rows, span_start, span_end
        if not pending_rows or span_start is None or span_end is None:
            return
        batch_images = np.asarray(image_dataset[span_start:span_end], dtype=np.uint8)
        for batch_offset, row in enumerate(pending_rows):
            results[int(row["_entropy_position"])] = (
                str(row["image_path"]),
                _calculate_entropy_from_array(batch_images[batch_offset], thumb=entropy_thumbnail),
            )
        pending_rows = []
        span_start = None
        span_end = None

    for row in batch_df.to_dict("records"):
        source_row_index = int(row["source_row_index"])
        if span_start is None:
            span_start = source_row_index
            span_end = source_row_index + 1
            pending_rows = [row]
            continue
        assert span_end is not None
        if source_row_index == span_end:
            span_end += 1
            pending_rows.append(row)
            continue
        flush_span()
        span_start = source_row_index
        span_end = source_row_index + 1
        pending_rows = [row]

    flush_span()


def compute_patient_entropy_median(df: pd.DataFrame, entropy_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate patch entropy to one median value per patient."""

    merged = df.merge(entropy_df, on="image_path", how="left")
    merged["entropy"] = merged["entropy"].fillna(0.0)
    grouped = merged.groupby("patient_id")["entropy"].median()
    return pd.DataFrame(
        {
            "patient_id": grouped.index.to_numpy(),
            "patient_entropy_median": grouped.to_numpy(dtype=np.float64),
        }
    )


def score_split_by_patient_entropy_median(
    patient_entropy_df: pd.DataFrame,
    patient_ids: list[int],
) -> float:
    """Score one split from the patient-level entropy objective."""

    subset = patient_entropy_df[patient_entropy_df["patient_id"].isin(patient_ids)]
    if subset.empty:
        return float("-inf")
    entropy_values = subset["patient_entropy_median"].to_numpy(dtype=np.float64)
    return float(np.median(entropy_values))
