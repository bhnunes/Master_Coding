from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from helpers.extraction.artifact_lookup import (
    GeoJsonLookup,
    ProcessingSignatureConfig,
    build_processing_signature,
    resolve_geojson_for_slide,
)
from helpers.extraction.config import DatabaseManagerConfig, load_database_manager_config
from helpers.extraction.image_reader_service import (
    SlideProcessingRequest,
    SlideProcessingResult,
    SlideRuntimeSettings,
    load_slide_runtime_settings,
    run_slide_processing,
)
from helpers.extraction.master_manifest import MasterManifest, Stage2SlideRows
from helpers.extraction.repository import (
    CaseUpdate,
    ExtractionCaseRecord,
    ExtractionRepository,
    IngestionOptions,
)
from helpers.extraction.staging import stage_wsi_locally
from helpers.logging_utils import LoggerSettings, configure_root_logger


class Style:
    """A helper class for styling terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    INFO = "i"
    SUCCESS = "+"
    WARNING = "!"
    ERROR = "x"
    ROCKET = ">"
    DB = "#"
    FOLDER = "*"
    CHECK = "?"


@contextmanager
def suppress_console_logging(
    logger: logging.Logger, *, level: int = logging.CRITICAL + 1
) -> Iterator[None]:
    """Temporarily silence console handlers while preserving file logging."""

    console_handlers: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            console_handlers.append((handler, handler.level))
            handler.setLevel(level)

    try:
        yield
    finally:
        for handler, original_level in console_handlers:
            handler.setLevel(original_level)


def ensure_project_is_initialized(
    repository: ExtractionRepository,
    source_folder: Path,
    output_base_path: Path,
    *,
    require_geojson: bool,
) -> tuple[tuple[Path, Path, Path], bool]:
    """Ensure Stage 2 source layout exists and output schema is initialized."""

    folders = repository.get_source_directories(source_folder)
    required_folders = folders if require_geojson else folders[:2]
    setup_needed = any(not folder.exists() for folder in required_folders)
    output_base_path.mkdir(parents=True, exist_ok=True)
    (output_base_path / "PATCHES").mkdir(parents=True, exist_ok=True)
    repository.initialize()
    return folders, setup_needed


def print_project_setup(folders: tuple[Path, Path, Path]) -> None:
    """Print the expected Stage 2 source-folder layout guidance."""

    images_folder, annotations_folder, geojson_folder = folders
    print(f"\n{Style.BLUE}{Style.BOLD}--- PROJECT SETUP ---{Style.RESET}")
    print(f"{Style.YELLOW}{Style.WARNING} Stage 2 source folders are missing.{Style.RESET}")
    print(f"{Style.INFO} Expected image folder: '{images_folder}'.{Style.RESET}")
    print(f"{Style.INFO} Expected annotation folder: '{annotations_folder}'.{Style.RESET}")
    print(f"{Style.INFO} Expected GeoJSON folder: '{geojson_folder}'.{Style.RESET}")
    print(
        f"{Style.YELLOW}{Style.INFO} Please point SOURCE_FOLDER at a directory that contains "
        f"these subfolders.{Style.RESET}"
    )
    print(
        f"{Style.BOLD}   IMPORTANT: Image and annotation files must share the same base name "
        "(e.g., case01.svs and case01.xml)."
        f"{Style.RESET}"
    )


def print_ingestion_summary() -> None:
    """Print the ingestion completion message."""

    print(
        f"\n{Style.GREEN}{Style.SUCCESS} Ingestion complete. Set {Style.BOLD}LOADCASES=False"
        f"{Style.RESET}{Style.GREEN} in .env to start processing.{Style.RESET}"
    )


def resolve_artifacts_geojson(
    case: ExtractionCaseRecord,
    config: DatabaseManagerConfig,
    *,
    geojson_lookup: GeoJsonLookup | None = None,
) -> Path | None:
    """Resolve the optional artifact GeoJSON path for one case."""

    if not config.use_advanced_artifact_filtering or config.geojson_path is None:
        return None
    return resolve_geojson_for_slide(config.geojson_path, case.image_path, lookup=geojson_lookup)


def build_processing_signature_for_case(
    case: ExtractionCaseRecord,
    config: DatabaseManagerConfig,
    runtime_settings: SlideRuntimeSettings,
    *,
    artifacts_geojson_path: Path | None,
) -> str:
    """Build the persisted Stage 2 processing signature for one case."""

    if case.annotation_path is None:
        raise ValueError(f"Case {case.record_id} is missing an annotation path.")
    return build_processing_signature(
        ProcessingSignatureConfig(
            image_path=case.image_path,
            annotation_path=case.annotation_path,
            artifacts_geojson_path=artifacts_geojson_path,
            window_size=config.window_size,
            stride=config.stride,
            match_percentage=config.match_percentage,
            tissue_percentage=config.tissue_percentage,
            target_level=runtime_settings.target_level,
            use_advanced_artifact_filtering=runtime_settings.use_advanced_artifact_filtering,
            hiseg_xml_coord_level=config.hiseg_xml_coord_level,
        )
    )


def build_slide_request(
    case: ExtractionCaseRecord,
    config: DatabaseManagerConfig,
    runtime_settings: SlideRuntimeSettings,
    *,
    image_path: Path | None = None,
    artifacts_geojson_path: Path | None = None,
) -> SlideProcessingRequest:
    """Build the shared slide-processing request for one case."""

    if case.annotation_path is None:
        raise ValueError(f"Case {case.record_id} is missing an annotation path.")
    hdf5_output_path = config.patch_base_path / "HDF5_SHARDS" / f"{Path(case.image_path).stem}.h5"
    return SlideProcessingRequest(
        image_path=image_path or case.image_path,
        annotation_path=case.annotation_path,
        dataset_tag=config.tag,
        patient=case.patient,
        window_size=config.window_size,
        stride=config.stride,
        match_percentage=config.match_percentage,
        tissue_percentage=config.tissue_percentage,
        target_level=runtime_settings.target_level,
        num_workers=runtime_settings.num_workers,
        use_advanced_artifact_filtering=runtime_settings.use_advanced_artifact_filtering,
        hiseg_xml_coord_level=config.hiseg_xml_coord_level,
        openslide_cache_bytes=runtime_settings.openslide_cache_bytes,
        hdf5_compression=runtime_settings.hdf5_compression,
        preload_scan_area_max_bytes=runtime_settings.preload_scan_area_max_bytes,
        artifacts_geojson_path=artifacts_geojson_path,
        hdf5_output_path=hdf5_output_path,
    )


def _configure_stage2_logger(config: DatabaseManagerConfig) -> logging.Logger:
    logger = configure_root_logger(
        config.log_path,
        settings=LoggerSettings(
            logger_level=logging.INFO,
            file_level=logging.INFO,
            console_level=logging.INFO,
            file_mode="a",
            file_pattern="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
            console_pattern="%(message)s",
        ),
    )
    logger.info("Stage 2 log file: %s", config.log_path)
    return logger


def _build_geojson_lookup(config: DatabaseManagerConfig) -> GeoJsonLookup | None:
    if config.geojson_path is None:
        return None
    if not (config.use_advanced_artifact_filtering or config.activate_sanity_check_geojson):
        return None
    return GeoJsonLookup.from_directory(config.geojson_path)


def _raise_for_stale_inputs(repository: ExtractionRepository) -> None:
    stale_cases = repository.list_stale_cases()
    if not stale_cases:
        return

    stale_names = ", ".join(case.image_path.name for case in stale_cases[:5])
    raise ValueError(
        "Stage 2 detected stale extraction inputs for existing cases. "
        "Clear stale extraction outputs and reprocess before continuing. "
        f"Examples: {stale_names}"
    )


def _raise_for_stale_processed_cases(
    repository: ExtractionRepository,
    config: DatabaseManagerConfig,
    runtime_settings: SlideRuntimeSettings,
    *,
    geojson_lookup: GeoJsonLookup | None,
) -> None:
    stale_processed_cases: list[str] = []
    for case in repository.list_completed_cases():
        if case.annotation_path is None or case.processing_signature is None:
            continue

        artifacts_geojson_path = resolve_artifacts_geojson(
            case,
            config,
            geojson_lookup=geojson_lookup,
        )
        expected_processing_signature = build_processing_signature_for_case(
            case,
            config,
            runtime_settings,
            artifacts_geojson_path=artifacts_geojson_path,
        )
        if expected_processing_signature != case.processing_signature:
            stale_processed_cases.append(case.image_path.name)

    if not stale_processed_cases:
        return

    examples = ", ".join(stale_processed_cases[:5])
    raise ValueError(
        "Stage 2 detected completed slides whose processing inputs or settings changed. "
        "Clear stale extraction outputs and reprocess before continuing. "
        f"Examples: {examples}"
    )


def _process_pending_case(
    case: ExtractionCaseRecord,
    *,
    config: DatabaseManagerConfig,
    repository: ExtractionRepository,
    runtime_settings: SlideRuntimeSettings,
    geojson_lookup: GeoJsonLookup | None,
    master_manifest: MasterManifest,
) -> SlideProcessingResult:
    repository.mark_processing(case.record_id)
    started_at = time.perf_counter()
    artifacts_geojson_path = resolve_artifacts_geojson(
        case,
        config,
        geojson_lookup=geojson_lookup,
    )
    processing_signature = build_processing_signature_for_case(
        case,
        config,
        runtime_settings,
        artifacts_geojson_path=artifacts_geojson_path,
    )
    slide_request = build_slide_request(
        case,
        config,
        runtime_settings,
        artifacts_geojson_path=artifacts_geojson_path,
    )
    with stage_wsi_locally(
        source_path=case.image_path,
        cache_dir=(config.local_slide_cache_dir if config.copy_wsi_to_local_cache else None),
        patient_id=case.patient,
    ) as staged_image_path:
        result = run_slide_processing(
            build_slide_request(
                case,
                config,
                runtime_settings,
                image_path=staged_image_path,
                artifacts_geojson_path=artifacts_geojson_path,
            )
        )
    if slide_request.hdf5_output_path is not None:
        master_manifest.replace_stage2_slide_rows(
            Stage2SlideRows(
                source_hdf5_path=slide_request.hdf5_output_path,
                records=result.artifact_patch_records,
                source_slide_path=case.image_path,
                annotation_path=case.annotation_path,
                artifacts_geojson_path=artifacts_geojson_path,
                stage2_case_record_id=case.record_id,
                stage2_processing_signature=processing_signature,
                stage2_status=result.status,
            )
        )
    elapsed_minutes = (time.perf_counter() - started_at) / 60
    repository.update_case(
        case.record_id,
        CaseUpdate(
            cancer_qtd=result.cancer_patches_created,
            non_cancer_qtd=result.not_cancer_patches_created,
            exec_time_minutes=elapsed_minutes,
            comments=result.comments[-240:],
            status=result.status,
            window_size=config.window_size,
            stride=config.stride,
            match_percentage=config.match_percentage,
            tissue_percentage=config.tissue_percentage,
            processing_signature=processing_signature,
        ),
    )
    return result


def _run_pending_processing(
    cases_to_process: list[ExtractionCaseRecord],
    *,
    logger: logging.Logger,
    config: DatabaseManagerConfig,
    repository: ExtractionRepository,
    runtime_settings: SlideRuntimeSettings,
    geojson_lookup: GeoJsonLookup | None,
    master_manifest: MasterManifest,
) -> None:
    print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.ROCKET} STARTING PROCESSING ---{Style.RESET}")
    with suppress_console_logging(logger):
        with tqdm(
            total=len(cases_to_process), desc=f"{Style.CYAN}Processing WSI slides{Style.RESET}"
        ) as progress_bar:
            for case in cases_to_process:
                progress_bar.set_description(
                    f"Processing: {Style.CYAN}{case.image_path.name}{Style.RESET}"
                )
                if case.annotation_path is None:
                    progress_bar.update(1)
                    continue

                result = _process_pending_case(
                    case,
                    config=config,
                    repository=repository,
                    runtime_settings=runtime_settings,
                    geojson_lookup=geojson_lookup,
                    master_manifest=master_manifest,
                )
                progress_bar.update(1)
                progress_bar.set_postfix(
                    cancer=result.cancer_patches_created,
                    non_cancer=result.not_cancer_patches_created,
                    artifact_rows=len(result.artifact_patch_records),
                    status=result.status,
                )


def main_process() -> None:
    """Main orchestration function."""

    load_dotenv(override=True)
    config = load_database_manager_config(os.environ)
    logger = _configure_stage2_logger(config)
    repository = ExtractionRepository(database_path=config.database_path, tag=config.tag)
    master_manifest = MasterManifest(config.master_manifest_path)
    master_manifest.initialize()
    folders, setup_needed = ensure_project_is_initialized(
        repository,
        config.source_folder,
        config.base_path,
        require_geojson=(
            config.use_advanced_artifact_filtering or config.activate_sanity_check_geojson
        ),
    )
    if setup_needed:
        print_project_setup(folders)
        return

    if config.load_cases:
        logger.info("Starting Stage 2 ingestion for tag=%s", config.tag)
        print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.CHECK} INGESTION PROCESS ---{Style.RESET}")
        repository.ingest_new_cases(
            IngestionOptions(
                source_folder=config.source_folder,
                activate_sanity_check=config.activate_sanity_check_geojson,
                use_advanced_filtering=config.use_advanced_artifact_filtering,
                geojson_path=config.geojson_path,
            )
        )
        print_ingestion_summary()
        return

    runtime_settings = load_slide_runtime_settings(os.environ)
    geojson_lookup = _build_geojson_lookup(config)
    logger.info("Starting Stage 2 processing for tag=%s", config.tag)
    _raise_for_stale_inputs(repository)
    _raise_for_stale_processed_cases(
        repository,
        config,
        runtime_settings,
        geojson_lookup=geojson_lookup,
    )
    cases_to_process = repository.list_pending_cases()
    if not cases_to_process:
        print(f"\n{Style.INFO} No cases to process with status 'TO BE PROCESSED'.")
        return

    _run_pending_processing(
        cases_to_process,
        logger=logger,
        config=config,
        repository=repository,
        runtime_settings=runtime_settings,
        geojson_lookup=geojson_lookup,
        master_manifest=master_manifest,
    )


if __name__ == "__main__":
    main_process()
