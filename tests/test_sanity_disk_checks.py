from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import pytest

from helpers.sanity.disk_checks import (
    check_decode_and_shapes,
    check_manifest_disk_parity,
    check_mask_pixel_values,
)


def test_check_manifest_disk_parity_fails_when_image_and_mask_filenames_differ(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "TRAIN" / "CANCER"
    mask_dir = tmp_path / "TRAIN" / "CANCER_MASK"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "PATIENT_1_PATCH_001.png").write_bytes(b"image")
    (mask_dir / "PATIENT_1_PATCH_002.png").write_bytes(b"mask")
    manifest_df = pd.DataFrame(
        [
            {
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_path_image": "TRAIN/CANCER/PATIENT_1_PATCH_001.png",
                "relative_path_mask": "TRAIN/CANCER_MASK/PATIENT_1_PATCH_001.png",
            }
        ]
    )

    result = check_manifest_disk_parity(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "filename parity" in result.details.lower()


def test_disk_checks_reuse_cached_pair_inspection_between_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_PATCH_001.png"
    mask_path = tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_PATCH_001.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    manifest_df = pd.DataFrame(
        [
            {
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_path_image": "TRAIN/CANCER/PATIENT_1_PATCH_001.png",
                "relative_path_mask": "TRAIN/CANCER_MASK/PATIENT_1_PATCH_001.png",
            }
        ]
    )

    original_imread = cast(Any, cv2.imread)
    read_calls = 0

    def counting_imread(*args: object, **kwargs: object) -> Any:
        nonlocal read_calls
        read_calls += 1
        return original_imread(*args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)

    shape_result = check_decode_and_shapes(manifest_df, tmp_path, "TRAIN", full_scan=True)
    mask_result = check_mask_pixel_values(manifest_df, tmp_path, "TRAIN", full_scan=True)

    assert shape_result.status == "PASS"
    assert mask_result.status == "PASS"
    assert read_calls == 2
