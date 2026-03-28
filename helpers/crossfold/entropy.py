from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm


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
            image = np.asarray(handle["images"][row_index], dtype=np.uint8)
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
    return pd.DataFrame(results, columns=["image_path", "entropy"])


def compute_patient_entropy_median(df: pd.DataFrame, entropy_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate patch entropy to one median value per patient."""

    merged = df.merge(entropy_df, on="image_path", how="left")
    merged["entropy"] = merged["entropy"].fillna(0.0)
    return (
        merged.groupby("patient_id")["entropy"].median().reset_index(name="patient_entropy_median")
    )


def score_split_by_patient_entropy_median(
    patient_entropy_df: pd.DataFrame,
    patient_ids: list[int],
) -> float:
    """Score one split from the patient-level entropy objective."""

    subset = patient_entropy_df[patient_entropy_df["patient_id"].isin(patient_ids)]
    if subset.empty:
        return float("-inf")
    return float(subset["patient_entropy_median"].median())
