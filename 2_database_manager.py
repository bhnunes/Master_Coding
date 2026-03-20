from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from helpers.extraction_artifact_index import ArtifactIndexWriter, ArtifactPatchRecord
from helpers.extraction_config import DatabaseManagerConfig, load_database_manager_config
from helpers.extraction_repository import CaseUpdate, ExtractionCaseRecord, ExtractionRepository
from helpers.image_reader_service import (
    SlideProcessingRequest,
    SlideRuntimeSettings,
    load_slide_runtime_settings,
    run_slide_processing,
)


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


def ensure_project_is_initialized(
    repository: ExtractionRepository, base_path: Path
) -> tuple[tuple[Path, Path, Path], bool]:
    """Ensure project folders and database schema exist."""

    images_folder = base_path / f"IMAGES_{repository.tag}"
    annotations_folder = base_path / f"ANNOTATIONS_{repository.tag}"
    geojson_folder = base_path / f"GEOJSON_{repository.tag}"
    setup_needed = any(
        not folder.exists() for folder in (images_folder, annotations_folder, geojson_folder)
    )
    folders = repository.ensure_case_directories(base_path)
    repository.initialize()
    return folders, setup_needed


def print_project_setup(folders: tuple[Path, Path, Path]) -> None:
    """Print the one-time project setup guidance."""

    images_folder, annotations_folder, geojson_folder = folders
    print(f"\n{Style.BLUE}{Style.BOLD}--- PROJECT SETUP ---{Style.RESET}")
    print(f"{Style.GREEN}{Style.SUCCESS} '{images_folder}' created.{Style.RESET}")
    print(f"{Style.GREEN}{Style.SUCCESS} '{annotations_folder}' created.{Style.RESET}")
    print(f"{Style.GREEN}{Style.SUCCESS} '{geojson_folder}' created.{Style.RESET}")
    print(
        f"{Style.YELLOW}{Style.INFO} Please move your images and annotations to these "
        f"folders.{Style.RESET}"
    )
    print(
        f"{Style.BOLD}   IMPORTANT: Image and annotation files must share the same base name "
        "(e.g., case01.svs and case01.xml)."
        f"{Style.RESET}"
    )


def print_ingestion_summary(table_name: str, svs_files_added: bool) -> None:
    """Print the ingestion completion messages."""

    print(
        f"\n{Style.GREEN}{Style.SUCCESS} Ingestion complete. Set {Style.BOLD}LOADCASES=False"
        f"{Style.RESET}{Style.GREEN} in .env to start processing.{Style.RESET}"
    )
    if not svs_files_added:
        return

    print(f"\n{Style.YELLOW}{Style.BOLD}SVS ACTION REQUIRED{Style.RESET}")
    print("SVS files require manual annotation color setup in the database.")
    print(
        f"Example: {Style.CYAN}UPDATE {table_name} SET CANCER_COLOR = '65280' WHERE ...;"
        f"{Style.RESET}"
    )


def collect_problematic_svs_files(cases: list[ExtractionCaseRecord]) -> list[str]:
    """Return SVS/XML cases that are missing color assignments."""

    problematic_files: list[str] = []
    for case in cases:
        if case.annotation_path is None:
            continue
        is_svs_xml = (
            case.image_path.suffix.lower() == ".svs"
            and case.annotation_path.suffix.lower() == ".xml"
        )
        colors_missing = not case.cancer_color or not case.not_cancer_color
        if is_svs_xml and colors_missing:
            problematic_files.append(case.image_path.name)
    return problematic_files


def ensure_patch_output_folders(base_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create patch output folders and return them."""

    patch_base_path = base_path / "PATCHES"
    cancer_folder = patch_base_path / "CANCER"
    not_cancer_folder = patch_base_path / "NOT_CANCER"
    cancer_mask_folder = patch_base_path / "CANCER_MASK"
    not_cancer_mask_folder = patch_base_path / "NOT_CANCER_MASK"
    for folder in (
        cancer_folder,
        not_cancer_folder,
        cancer_mask_folder,
        not_cancer_mask_folder,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return cancer_folder, not_cancer_folder, cancer_mask_folder, not_cancer_mask_folder


def resolve_artifacts_geojson(
    case: ExtractionCaseRecord, config: DatabaseManagerConfig
) -> Path | None:
    """Resolve the optional artifact GeoJSON path for one case."""

    if not config.use_advanced_artifact_filtering or config.geojson_path is None:
        return None
    candidate = config.geojson_path / f"{case.image_path.stem}.geojson"
    return candidate if candidate.exists() else None


def build_slide_request(
    case: ExtractionCaseRecord,
    config: DatabaseManagerConfig,
    runtime_settings: SlideRuntimeSettings,
    patch_folders: tuple[Path, Path, Path, Path],
) -> SlideProcessingRequest:
    """Build the shared slide-processing request for one case."""

    cancer_folder, not_cancer_folder, cancer_mask_folder, not_cancer_mask_folder = patch_folders
    if case.annotation_path is None:
        raise ValueError(f"Case {case.record_id} is missing an annotation path.")
    return SlideProcessingRequest(
        image_path=case.image_path,
        annotation_path=case.annotation_path,
        cancer_folder=cancer_folder,
        not_cancer_folder=not_cancer_folder,
        cancer_mask_folder=cancer_mask_folder,
        not_cancer_mask_folder=not_cancer_mask_folder,
        cancer_color=case.cancer_color or "NA",
        not_cancer_color=case.not_cancer_color or "NA",
        patient=case.patient,
        window_size=config.window_size,
        stride=config.stride,
        match_percentage=config.match_percentage,
        tissue_percentage=config.tissue_percentage,
        target_level=runtime_settings.target_level,
        num_workers=runtime_settings.num_workers,
        use_advanced_artifact_filtering=runtime_settings.use_advanced_artifact_filtering,
        artifacts_geojson_path=resolve_artifacts_geojson(case, config),
    )


def main_process() -> None:
    """Main orchestration function."""

    load_dotenv(override=True)
    config = load_database_manager_config(os.environ)
    repository = ExtractionRepository(database_path=config.database_path, tag=config.tag)
    folders, setup_needed = ensure_project_is_initialized(repository, config.base_path)
    if setup_needed:
        print_project_setup(folders)
        return

    if config.load_cases:
        print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.CHECK} INGESTION PROCESS ---{Style.RESET}")
        svs_added = repository.ingest_new_cases(
            base_path=config.base_path,
            activate_sanity_check=config.activate_sanity_check_geojson,
            use_advanced_filtering=config.use_advanced_artifact_filtering,
            geojson_path=config.geojson_path,
        )
        print_ingestion_summary(config.table_name, svs_added)
        return

    runtime_settings = load_slide_runtime_settings(os.environ)
    cases_to_process = repository.list_pending_cases()
    if not cases_to_process:
        print(f"\n{Style.INFO} No cases to process with status 'TO BE PROCESSED'.")
        return

    problematic_svs_files = collect_problematic_svs_files(cases_to_process)
    if problematic_svs_files:
        error_message = (
            f"\n{Style.RED}{Style.ERROR}{Style.BOLD} PROCESSING HALTED: Missing SVS annotation "
            f"colors.{Style.RESET}\n"
            f"{Style.YELLOW}The following .svs files are marked 'TO BE PROCESSED' but do not "
            f"have "
            f"'CANCER_COLOR' and/or 'NOT_CANCER_COLOR' set in the database.{Style.RESET}\n\n"
            + "\n".join(f"  - {name}" for name in problematic_svs_files)
            + f"\n\nPlease run an UPDATE query on the '{Style.CYAN}{config.table_name}"
            f"{Style.RESET}' table to set these values before proceeding."
        )
        raise ValueError(error_message)

    print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.ROCKET} STARTING PROCESSING ---{Style.RESET}")
    patch_folders = ensure_patch_output_folders(config.base_path)
    artifact_index_writer = ArtifactIndexWriter(
        config.patch_base_path / "artifact_patch_index.parquet"
    )

    try:
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

                repository.mark_processing(case.record_id)
                started_at = time.perf_counter()
                result = run_slide_processing(
                    build_slide_request(case, config, runtime_settings, patch_folders)
                )
                artifact_index_writer.append_records(
                    [ArtifactPatchRecord(**record) for record in result.artifact_patch_records]
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
                    ),
                )
                progress_bar.update(1)
                progress_bar.set_postfix(
                    cancer=result.cancer_patches_created,
                    non_cancer=result.not_cancer_patches_created,
                    artifact_rows=len(result.artifact_patch_records),
                    status=result.status,
                )
    finally:
        artifact_index_writer.close()


if __name__ == "__main__":
    main_process()
