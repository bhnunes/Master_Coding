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


def test_run_packaging_pipeline_writes_single_source_hdf5_from_cleaned_patch_pool(
    tmp_path: Path,
) -> None:
    _write_rgb_png(tmp_path / "CANCER" / "PATIENT_1_train.png")
    _write_mask_png(tmp_path / "CANCER_MASK" / "PATIENT_1_train.png", positive=True)
    _write_rgb_png(tmp_path / "NOT_CANCER" / "PATIENT_2_val.png")
    _write_mask_png(tmp_path / "NOT_CANCER_MASK" / "PATIENT_2_val.png", positive=False)

    config = PackagingConfig(
        base_dir=tmp_path,
        output_dir=tmp_path,
        output_filename="SOURCE_DATASET.h5",
        img_size=12,
        patient_id_regex=r"PATIENT_(\d+)_",
        overwrite_outputs=True,
    )

    result = run_packaging_pipeline(config)

    assert result == tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(result, "r") as handle:
        assert handle["labels"][:].tolist() == [1, 0]
        assert handle["patient_ids"][:].tolist() == [1, 2]
        assert handle["filenames"][:].tolist() == [b"PATIENT_1_train.png", b"PATIENT_2_val.png"]
        assert handle["source_image_paths"][0].decode("utf-8").endswith("PATIENT_1_train.png")
