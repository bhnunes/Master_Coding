from pathlib import Path

import pandas as pd

from helpers.sanity.disk_checks import check_manifest_disk_parity


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
