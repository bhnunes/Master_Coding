import json
from pathlib import Path

import pandas as pd

from helpers.sanity.config import SanityConfig
from helpers.sanity.pipeline import run_sanity_pipeline


def test_run_sanity_pipeline_rejects_duplicate_filenames(tmp_path: Path) -> None:
    output_dir = tmp_path
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "label": 1,
                "patient_id": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_path_image": "TRAIN/CANCER/PATIENT_1_PATCH_001.png",
                "relative_path_mask": "TRAIN/CANCER_MASK/PATIENT_1_PATCH_001.png",
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
            {
                "run_id": "run-1",
                "split": "TEST",
                "label": 1,
                "patient_id": 2,
                "filename": "PATIENT_1_PATCH_001.png",
                "relative_path_image": "TEST/CANCER/PATIENT_1_PATCH_001.png",
                "relative_path_mask": "TEST/CANCER_MASK/PATIENT_1_PATCH_001.png",
                "normalization_method": "NOT_NORMALIZED",
                "is_normalized": False,
            },
        ]
    )
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "split_stats.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")

    report = run_sanity_pipeline(
        SanityConfig(
            base_dir=output_dir,
            sample_pairs=10,
            full_mask_scan=False,
            full_shape_scan=False,
            checksum_mode="OFF",
            enforce_filename_uniqueness=True,
            enforce_regex_patient_id_match=True,
            enforce_split_stats_parity=False,
            fail_on_empty_cancer_mask=True,
            fail_on_positive_not_cancer_mask=True,
        )
    )

    assert report.verdict == "SPLITS REJECTED"
    assert report.dataset_checks["Filename Uniqueness"].status == "FAIL"
