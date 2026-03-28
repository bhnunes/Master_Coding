from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from helpers.crossfold.entropy import (
    calculate_image_entropy_from_path,
    compute_all_patch_entropies,
    compute_patient_entropy_median,
    score_split_by_patient_entropy_median,
)


def test_calculate_image_entropy_from_path_returns_zero_for_missing_file(tmp_path: Path) -> None:
    image_path = tmp_path / "missing.png"

    result_path, entropy = calculate_image_entropy_from_path(str(image_path))

    assert result_path == str(image_path)
    assert entropy == 0.0


def test_compute_patient_entropy_median_and_score_split() -> None:
    dataset = pd.DataFrame(
        [
            {"patient_id": 1, "image_path": "a.png"},
            {"patient_id": 1, "image_path": "b.png"},
            {"patient_id": 2, "image_path": "c.png"},
        ]
    )
    entropy_df = pd.DataFrame(
        [
            {"image_path": "a.png", "entropy": 0.5},
            {"image_path": "b.png", "entropy": 1.5},
            {"image_path": "c.png", "entropy": 3.0},
        ]
    )

    patient_entropy_df = compute_patient_entropy_median(dataset, entropy_df)

    assert patient_entropy_df.to_dict("records") == [
        {"patient_id": 1, "patient_entropy_median": 1.0},
        {"patient_id": 2, "patient_entropy_median": 3.0},
    ]
    assert score_split_by_patient_entropy_median(patient_entropy_df, [1, 2]) == pytest.approx(2.0)


def test_compute_all_patch_entropies_uses_sequential_worker_path(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    cv2.imwrite(str(image_a), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(
        str(image_b),
        np.array([[[0, 0, 0], [255, 255, 255]]] * 8, dtype=np.uint8),
    )
    dataset = pd.DataFrame({"image_path": [str(image_a), str(image_b)]})

    entropy_df = compute_all_patch_entropies(
        dataset, num_workers=1, chunksize=1, entropy_thumbnail=8
    )

    assert entropy_df["image_path"].tolist() == [str(image_a), str(image_b)]
    assert entropy_df["entropy"].iloc[0] == pytest.approx(0.0)
    assert entropy_df["entropy"].iloc[1] > 0.0


def test_score_split_by_patient_entropy_median_returns_negative_infinity_for_empty_split() -> None:
    patient_entropy_df = pd.DataFrame([{"patient_id": 1, "patient_entropy_median": 1.0}])

    assert score_split_by_patient_entropy_median(patient_entropy_df, [99]) == float("-inf")


def test_compute_all_patch_entropies_supports_hdf5_backed_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    variable = np.zeros((8, 8, 3), dtype=np.uint8)
    variable[:, 1::2, :] = 255
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack([np.zeros((8, 8, 3), dtype=np.uint8), variable], axis=0),
        )

    dataset = pd.DataFrame(
        [
            {
                "image_path": f"{source_path}::images[0]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 0,
            },
            {
                "image_path": f"{source_path}::images[1]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 1,
            },
        ]
    )

    entropy_df = compute_all_patch_entropies(
        dataset, num_workers=1, chunksize=1, entropy_thumbnail=8
    )

    assert entropy_df["image_path"].tolist() == [
        f"{source_path}::images[0]",
        f"{source_path}::images[1]",
    ]
    assert entropy_df["entropy"].iloc[0] == pytest.approx(0.0)
    assert entropy_df["entropy"].iloc[1] > 0.0
