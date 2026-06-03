from helpers.extraction.patch_engine import build_patch_filename, sanitize_patch_filename_component


def test_build_patch_filename_is_deterministic() -> None:
    first = build_patch_filename(
        label="CANCER",
        patient_id="42",
        slide_id="slide_a",
        x_coord=128,
        y_coord=256,
    )
    second = build_patch_filename(
        label="CANCER",
        patient_id="42",
        slide_id="slide_a",
        x_coord=128,
        y_coord=256,
    )

    assert first == second
    assert first == "CANCER_PATIENT_42_SLIDE_slide_a_X_128_Y_256.png"


def test_build_patch_filename_sanitizes_components() -> None:
    filename = build_patch_filename(
        label="NOT CANCER",
        patient_id="patient 7",
        slide_id="slide/a:b",
        x_coord=5,
        y_coord=9,
    )

    assert filename == "NOT-CANCER_PATIENT_patient-7_SLIDE_slide-a-b_X_5_Y_9.png"


def test_sanitize_patch_filename_component_falls_back_for_empty_values() -> None:
    assert sanitize_patch_filename_component("   ") == "unknown"
