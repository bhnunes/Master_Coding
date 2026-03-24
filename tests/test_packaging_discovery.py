from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers.packaging.discovery import discover_patch_pool_samples


def _write_rgb_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask_png(path: Path, *, positive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full((8, 8), 255 if positive else 0, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def test_discover_patch_pool_samples_returns_sorted_pairs(tmp_path: Path) -> None:
    base_dir = tmp_path
    _write_rgb_png(base_dir / "CANCER" / "PATIENT_2_a.png")
    _write_mask_png(base_dir / "CANCER_MASK" / "PATIENT_2_a.png", positive=True)
    _write_rgb_png(base_dir / "NOT_CANCER" / "PATIENT_1_b.png")
    _write_mask_png(base_dir / "NOT_CANCER_MASK" / "PATIENT_1_b.png", positive=False)

    samples = discover_patch_pool_samples(base_dir, r"PATIENT_(\d+)_")

    assert [sample.filename for sample in samples] == ["PATIENT_1_b.png", "PATIENT_2_a.png"]
    assert [sample.label for sample in samples] == [0, 1]
    assert [sample.patient_id for sample in samples] == [1, 2]


def test_discover_patch_pool_samples_fails_on_missing_mask(tmp_path: Path) -> None:
    base_dir = tmp_path
    _write_rgb_png(base_dir / "CANCER" / "PATIENT_7_a.png")

    with pytest.raises(ValueError, match="CANCER"):
        discover_patch_pool_samples(base_dir, r"PATIENT_(\d+)_")


def test_discover_patch_pool_samples_fails_on_invalid_patient_id_filename(tmp_path: Path) -> None:
    base_dir = tmp_path
    _write_rgb_png(base_dir / "CANCER" / "BAD_NAME.png")
    _write_mask_png(base_dir / "CANCER_MASK" / "BAD_NAME.png", positive=True)

    with pytest.raises(ValueError, match="patient id"):
        discover_patch_pool_samples(base_dir, r"PATIENT_(\d+)_")
