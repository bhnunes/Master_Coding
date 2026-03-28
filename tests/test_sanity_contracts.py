import pandas as pd

from helpers.sanity.contracts import (
    check_filename_patient_id_consistency,
    check_filename_uniqueness,
    check_source_reference_contract,
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


def test_check_source_reference_contract_accepts_logical_hdf5_refs() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "source_image_path": "/data/stage2_source.h5::images[7]",
                "source_mask_path": "/data/stage2_source.h5::masks[7]",
            }
        ]
    )

    result = check_source_reference_contract(manifest_df)

    assert result.status == "PASS"
    assert "source references" in result.details.lower()


def test_check_source_reference_contract_rejects_misaligned_logical_hdf5_refs() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "source_image_path": "/data/stage2_source.h5::images[7]",
                "source_mask_path": "/data/stage2_source.h5::masks[8]",
                "filename": "PATIENT_1_PATCH_001.png",
            }
        ]
    )

    result = check_source_reference_contract(manifest_df)

    assert result.status == "FAIL"
    assert "logical hdf5 refs" in result.details.lower()
