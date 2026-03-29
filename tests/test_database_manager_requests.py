import importlib.util
from pathlib import Path

from helpers.extraction.config import DatabaseManagerConfig
from helpers.extraction.image_reader_service import SlideRuntimeSettings
from helpers.extraction.repository import ExtractionCaseRecord

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "stage2_database_manager",
    Path(__file__).resolve().parents[1] / "2_database_manager.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
build_slide_request = _MODULE.build_slide_request


def test_build_slide_request_includes_stage2_hdf5_shard_output(tmp_path: Path) -> None:
    case = ExtractionCaseRecord(
        record_id=1,
        status="TO BE PROCESSED",
        image_path=tmp_path / "slide_a.svs",
        annotation_path=tmp_path / "slide_a.xml",
        cancer_qtd=None,
        non_cancer_qtd=None,
        valid_image=None,
        cancer_color="65280",
        not_cancer_color="255",
        processing_time_minutes=None,
        patient="1001",
        comments="",
        window_size=None,
        stride=None,
        match_percentage=None,
        tissue_percentage=None,
        last_update=None,
        input_signature=None,
        processing_signature=None,
    )
    config = DatabaseManagerConfig(
        tag="TEST",
        database_path=tmp_path / "db.sqlite",
        base_path=tmp_path,
        window_size=224,
        stride=112,
        match_percentage=1.0,
        tissue_percentage=0.3,
        target_level=0,
        num_workers=2,
        load_cases=False,
        use_advanced_artifact_filtering=False,
        activate_sanity_check_geojson=False,
        geojson_path=None,
        copy_wsi_to_local_cache=False,
        local_slide_cache_dir=None,
        log_folder=tmp_path / "logs",
        log_file_name="stage2.log",
    )
    runtime_settings = SlideRuntimeSettings(
        window_size=224,
        stride=112,
        match_percentage=1.0,
        tissue_percentage=0.3,
        target_level=0,
        num_workers=2,
        use_advanced_artifact_filtering=False,
    )

    request = build_slide_request(
        case,
        config,
        runtime_settings,
    )

    assert request.hdf5_output_path == tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_a.h5"
