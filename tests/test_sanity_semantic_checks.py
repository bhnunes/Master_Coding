from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from helpers.sanity.semantic_checks import check_mask_label_semantics


def test_check_mask_label_semantics_fails_for_positive_pixels_in_not_cancer(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "PATIENT_1_PATCH_001.png"
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    cv2.imwrite(str(mask_path), mask)
    manifest_df = pd.DataFrame(
        [
            {
                "label": 0,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_path_mask": mask_path.name,
            }
        ]
    )

    result = check_mask_label_semantics(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "not_cancer" in result.details.lower()


def test_check_mask_label_semantics_fails_for_empty_cancer_mask(tmp_path: Path) -> None:
    mask_path = tmp_path / "PATIENT_2_PATCH_001.png"
    cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    manifest_df = pd.DataFrame(
        [
            {
                "label": 1,
                "filename": "PATIENT_2_PATCH_001.png",
                "relative_path_mask": mask_path.name,
            }
        ]
    )

    result = check_mask_label_semantics(manifest_df, tmp_path, "TRAIN")

    assert result.status == "FAIL"
    assert "cancer" in result.details.lower()
