from pathlib import Path

import pytest

from helpers.image_reader_service import (
    SlideProcessingRequest,
    get_handler_for_files,
    run_slide_processing,
)


def test_get_handler_for_files_rejects_unknown_extension_pair() -> None:
    with pytest.raises(ValueError, match="No handler found"):
        get_handler_for_files("slide.abc", "annotation.xyz")


def test_run_slide_processing_returns_patch_engine_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    artifact_records = [{"filename": "patch.png", "cov_fold": 0.5}]

    def fake_run_extraction(**kwargs: object) -> tuple[int, int, list[dict[str, object]]]:
        captured.update(kwargs)
        return (7, 3, artifact_records)

    monkeypatch.setattr(
        "helpers.image_reader_service.patch_engine.run_extraction", fake_run_extraction
    )

    request = SlideProcessingRequest(
        image_path=tmp_path / "slide.svs",
        annotation_path=tmp_path / "slide.xml",
        cancer_folder=tmp_path / "cancer",
        not_cancer_folder=tmp_path / "not_cancer",
        cancer_mask_folder=tmp_path / "cancer_mask",
        not_cancer_mask_folder=tmp_path / "not_cancer_mask",
        cancer_color="65280",
        not_cancer_color="255",
        patient="100001",
        window_size=224,
        stride=112,
        match_percentage=0.6,
        tissue_percentage=0.3,
        target_level=0,
        num_workers=2,
        use_advanced_artifact_filtering=False,
        artifacts_geojson_path=None,
    )

    result = run_slide_processing(request)

    assert result.status == "COMPLETED"
    assert result.cancer_patches_created == 7
    assert result.not_cancer_patches_created == 3
    assert result.artifact_patch_records == artifact_records
    assert captured["path_Image"] == str(tmp_path / "slide.svs")
    assert captured["patient"] == "100001"
    assert captured["window_size"] == 224
