from pathlib import Path

from helpers.crossfold.discovery import extract_patient_id, load_patch_dataset


def test_extract_patient_id_returns_integer_for_expected_pattern() -> None:
    assert extract_patient_id("PATIENT_42_PATCH_001.png") == 42
    assert extract_patient_id("unknown.png") is None


def test_load_patch_dataset_only_keeps_valid_image_mask_pairs(tmp_path: Path) -> None:
    cancer_dir = tmp_path / "CANCER"
    cancer_mask_dir = tmp_path / "CANCER_MASK"
    not_cancer_dir = tmp_path / "NOT_CANCER"
    not_cancer_mask_dir = tmp_path / "NOT_CANCER_MASK"
    for path in (cancer_dir, cancer_mask_dir, not_cancer_dir, not_cancer_mask_dir):
        path.mkdir()

    (cancer_dir / "PATIENT_1_PATCH_001.png").write_bytes(b"image")
    (cancer_mask_dir / "PATIENT_1_PATCH_001.png").write_bytes(b"mask")
    (cancer_dir / "PATIENT_2_PATCH_001.png").write_bytes(b"orphan")
    (not_cancer_dir / "PATIENT_3_PATCH_001.png").write_bytes(b"image")
    (not_cancer_mask_dir / "PATIENT_3_PATCH_001.png").write_bytes(b"mask")
    (not_cancer_dir / "bad_name.png").write_bytes(b"image")
    (not_cancer_mask_dir / "bad_name.png").write_bytes(b"mask")

    dataset = load_patch_dataset(tmp_path)

    assert dataset[["patient_id", "label", "filename"]].to_dict("records") == [
        {"patient_id": 1, "label": 1, "filename": "PATIENT_1_PATCH_001.png"},
        {"patient_id": 3, "label": 0, "filename": "PATIENT_3_PATCH_001.png"},
    ]
