import pandas as pd

from helpers.sanity.manifest_checks import check_split_stats_against_manifest


def test_check_split_stats_against_manifest_fails_when_counts_drift() -> None:
    manifest_df = pd.DataFrame(
        [
            {"run_id": "run-1", "split": "TRAIN", "label": 1, "patient_id": 1},
            {"run_id": "run-1", "split": "TRAIN", "label": 0, "patient_id": 2},
        ]
    )
    split_stats_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "split": "TRAIN",
                "n_patients": 2,
                "n_pos_patients": 1,
                "n_neg_patients": 1,
                "n_images": 3,
                "patches_per_patient_mean": 1.0,
                "patches_per_patient_std": 0.0,
                "patches_per_patient_min": 1,
                "patches_per_patient_q1": 1.0,
                "patches_per_patient_median": 1.0,
                "patches_per_patient_q3": 1.0,
                "patches_per_patient_max": 1,
                "n_images_neg": 1,
                "n_images_pos": 1,
            }
        ]
    )

    result = check_split_stats_against_manifest(manifest_df, split_stats_df)

    assert result.status == "FAIL"
    assert "n_images" in result.details
