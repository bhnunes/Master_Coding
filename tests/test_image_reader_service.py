from pathlib import Path

import pytest

from helpers.extraction.image_reader_service import (
    SlideProcessingRequest,
    get_handler_for_files,
    load_slide_runtime_settings,
    run_slide_processing,
)

CREATED_CANCER_PATCHES = 7
CREATED_NOT_CANCER_PATCHES = 3
HIESD_XML_COORD_LEVEL = 6
WINDOW_SIZE = 224
OPENSLIDE_CACHE_BYTES = 134217728


def test_get_handler_for_files_rejects_unknown_extension_pair() -> None:
    with pytest.raises(ValueError, match="No handler found"):
        get_handler_for_files("slide.abc", "annotation.xyz")


def test_load_slide_runtime_settings_defaults_artifact_filtering_to_true() -> None:
    settings = load_slide_runtime_settings({})

    assert settings.use_advanced_artifact_filtering is True
    assert settings.suppress_native_tiff_warnings is True


def test_load_slide_runtime_settings_allows_native_tiff_warnings() -> None:
    settings = load_slide_runtime_settings({"STAGE2_SUPPRESS_NATIVE_TIFF_WARNINGS": "false"})

    assert settings.suppress_native_tiff_warnings is False


def test_run_slide_processing_returns_patch_engine_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    artifact_records = [{"filename": "patch.png", "cov_fold": 0.5}]
    hdf5_calls: list[tuple[Path, list[dict[str, object]], dict[str, object]]] = []
    manifest_calls: list[tuple[Path, Path]] = []

    def fake_run_extraction(**kwargs: object) -> tuple[int, int, list[dict[str, object]]]:
        captured.update(kwargs)
        return (CREATED_CANCER_PATCHES, CREATED_NOT_CANCER_PATCHES, artifact_records)

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
        window_size=WINDOW_SIZE,
        stride=112,
        match_percentage=0.6,
        tissue_percentage=0.3,
        target_level=0,
        num_workers=2,
        use_advanced_artifact_filtering=False,
        hiesd_xml_coord_level=HIESD_XML_COORD_LEVEL,
        openslide_cache_bytes=OPENSLIDE_CACHE_BYTES,
        hdf5_compression="gzip",
        preload_scan_area_max_bytes=0,
        suppress_native_tiff_warnings=True,
        artifacts_geojson_path=None,
        profile_output_path=tmp_path / "profile.json",
        hdf5_output_path=tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide.h5",
    )

    result = run_slide_processing(request)

    assert result.status == "COMPLETED"
    assert result.cancer_patches_created == CREATED_CANCER_PATCHES
    assert result.not_cancer_patches_created == CREATED_NOT_CANCER_PATCHES
    assert result.artifact_patch_records == artifact_records
    assert captured["path_Image"] == str(tmp_path / "slide.svs")
    assert captured["patient"] == "100001"
    assert captured["dataset_tag"] == "TEST"
    assert captured["hiesd_xml_coord_level"] == HIESD_XML_COORD_LEVEL
    assert captured["window_size"] == WINDOW_SIZE
    assert captured["profile_output_path"] == str(tmp_path / "profile.json")
    assert captured["openslide_cache_bytes"] == OPENSLIDE_CACHE_BYTES
    assert captured["preload_scan_area_max_bytes"] == 0
    assert captured["suppress_native_tiff_warnings"] is True
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
