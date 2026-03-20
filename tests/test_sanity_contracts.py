import pandas as pd

from helpers.sanity.contracts import (
    check_filename_patient_id_consistency,
    check_filename_uniqueness,
)


def test_check_filename_uniqueness_fails_for_duplicate_filenames_across_splits() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "split": "TRAIN",
                "label": 1,
                "patient_id": 10,
                "filename": "PATIENT_10_PATCH_001.png",
            },
            {
                "split": "TEST",
                "label": 0,
                "patient_id": 20,
                "filename": "PATIENT_10_PATCH_001.png",
            },
        ]
    )

    result = check_filename_uniqueness(manifest_df)

    assert result.status == "FAIL"
    assert "duplicate filenames" in result.details.lower()


def test_check_filename_patient_id_consistency_fails_when_filename_id_disagrees() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "patient_id": 10,
                "filename": "PATIENT_999_PATCH_001.png",
            }
        ]
    )

    result = check_filename_patient_id_consistency(manifest_df)

    assert result.status == "FAIL"
    assert "patient_id" in result.details
