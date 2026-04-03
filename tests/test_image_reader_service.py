from pathlib import Path

import pytest

from helpers.extraction.image_reader_service import (
    SlideProcessingRequest,
    get_handler_for_files,
    load_slide_runtime_settings,
    run_slide_processing,
)


def test_get_handler_for_files_rejects_unknown_extension_pair() -> None:
    with pytest.raises(ValueError, match="No handler found"):
        get_handler_for_files("slide.abc", "annotation.xyz")


def test_load_slide_runtime_settings_defaults_artifact_filtering_to_true() -> None:
    settings = load_slide_runtime_settings({})

    assert settings.use_advanced_artifact_filtering is True


def test_run_slide_processing_returns_patch_engine_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    artifact_records = [{"filename": "patch.png", "cov_fold": 0.5}]
    hdf5_calls: list[tuple[Path, list[dict[str, object]], dict[str, object]]] = []
    manifest_calls: list[tuple[Path, Path]] = []

    def fake_run_extraction(**kwargs: object) -> tuple[int, int, list[dict[str, object]]]:
        captured.update(kwargs)
        return (7, 3, artifact_records)

    monkeypatch.setattr(
        "helpers.extraction.image_reader_service.patch_engine.run_extraction", fake_run_extraction
    )

    def fake_write_slide_patch_dataset_hdf5(
        output_path: Path, records: list[dict[str, object]], **kwargs: object
    ) -> Path:
        hdf5_calls.append((output_path, records, kwargs))
        return output_path

    monkeypatch.setattr(
        "helpers.extraction.image_reader_service.write_slide_patch_dataset_hdf5",
        fake_write_slide_patch_dataset_hdf5,
    )
    monkeypatch.setattr(
        "helpers.extraction.image_reader_service.update_stage2_shard_manifest",
        lambda manifest_path, shard_path: manifest_calls.append((manifest_path, shard_path)),
    )

    request = SlideProcessingRequest(
        image_path=tmp_path / "slide.svs",
        annotation_path=tmp_path / "slide.xml",
        dataset_tag="TEST",
        patient="100001",
        window_size=224,
        stride=112,
        match_percentage=0.6,
        tissue_percentage=0.3,
        target_level=0,
        num_workers=2,
        use_advanced_artifact_filtering=False,
        hiseg_xml_coord_level=6,
        openslide_cache_bytes=134217728,
        hdf5_compression="gzip",
        preload_scan_area_max_bytes=0,
        artifacts_geojson_path=None,
        profile_output_path=tmp_path / "profile.json",
        hdf5_output_path=tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide.h5",
    )

    result = run_slide_processing(request)

    assert result.status == "COMPLETED"
    assert result.cancer_patches_created == 7
    assert result.not_cancer_patches_created == 3
    assert result.artifact_patch_records == artifact_records
    assert captured["path_Image"] == str(tmp_path / "slide.svs")
    assert captured["patient"] == "100001"
    assert captured["dataset_tag"] == "TEST"
    assert captured["hiseg_xml_coord_level"] == 6
    assert captured["window_size"] == 224
    assert captured["profile_output_path"] == str(tmp_path / "profile.json")
    assert captured["openslide_cache_bytes"] == 134217728
    assert captured["preload_scan_area_max_bytes"] == 0
    assert hdf5_calls == [
        (
            tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide.h5",
            artifact_records,
            {"compression": "gzip"},
        )
    ]
    assert manifest_calls == [
        (
            tmp_path / "PATCHES" / "HDF5_SHARDS" / "manifest.json",
            tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide.h5",
        )
    ]
