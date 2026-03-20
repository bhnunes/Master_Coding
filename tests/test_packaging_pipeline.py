from pathlib import Path

import cv2
import h5py
import numpy as np

from helpers.packaging.config import PackagingConfig
from helpers.packaging.pipeline import run_packaging_pipeline


def _write_rgb_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask_png(path: Path, *, positive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full((8, 8), 255 if positive else 0, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def test_run_packaging_pipeline_writes_train_and_validation_and_skips_missing_test(
    tmp_path: Path,
) -> None:
    _write_rgb_png(tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_train.png")
    _write_mask_png(tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_train.png", positive=True)
    _write_rgb_png(tmp_path / "VALIDATION" / "NOT_CANCER" / "PATIENT_2_val.png")
    _write_mask_png(
        tmp_path / "VALIDATION" / "NOT_CANCER_MASK" / "PATIENT_2_val.png",
        positive=False,
    )

    config = PackagingConfig(
        base_dir=tmp_path,
        output_dir=tmp_path,
        splits=("TRAIN", "VALIDATION", "TEST"),
        img_size=12,
        patient_id_regex=r"PATIENT_(\d+)_",
        overwrite_outputs=True,
    )

    result = run_packaging_pipeline(config)

    assert result == {
        "TRAIN": tmp_path / "TRAIN.h5",
        "VALIDATION": tmp_path / "VALIDATION.h5",
    }
    assert not (tmp_path / "TEST.h5").exists()
    with h5py.File(tmp_path / "TRAIN.h5", "r") as handle:
        assert handle["labels"][0] == 1
    with h5py.File(tmp_path / "VALIDATION.h5", "r") as handle:
        assert handle["labels"][0] == 0
