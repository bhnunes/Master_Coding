from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from helpers.extraction.artifact_lookup import GeoJsonLookup, resolve_geojson_for_slide
from helpers.provenance import build_file_metadata_fingerprint, hash_json_payload

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionCaseRecord:
    """One extraction case tracked in the Stage 2 database."""

    record_id: int
    status: str
    image_path: Path
    annotation_path: Path | None
    cancer_qtd: int | None
    non_cancer_qtd: int | None
    valid_image: int | None
    processing_time_minutes: float | None
    patient: str
    comments: str
    window_size: int | None
    stride: int | None
    match_percentage: str | None
    tissue_percentage: str | None
    last_update: str | None
    input_signature: str | None
    processing_signature: str | None


@dataclass(frozen=True)
class CaseUpdate:
    """Persisted processing result for one extraction case."""

    cancer_qtd: int
    non_cancer_qtd: int
    exec_time_minutes: float
    comments: str
    status: str
    window_size: int
    stride: int
    match_percentage: float
    tissue_percentage: float
    processing_signature: str | None = None


@dataclass(frozen=True)
class IngestionOptions:
    source_folder: Path
    activate_sanity_check: bool
    use_advanced_filtering: bool
    geojson_path: Path | None


@dataclass(frozen=True)
class ExistingCaseIndex:
    rows: Sequence[sqlite3.Row]
    by_basename: dict[str, sqlite3.Row]
    next_patient_id: int


@dataclass(frozen=True)
class IngestionDecision:
    image_path: Path
    annotation_path: str | None
    input_signature: str
    status: str
    comments: str
    stale_row_id: int | None = None


@dataclass(frozen=True)
class IngestionResources:
    images_folder: Path
    image_files: list[Path]
    image_listing_elapsed: float
    run_geojson_check: bool
    geojson_lookup: GeoJsonLookup | None
    geojson_lookup_elapsed: float
    annotation_lookup: dict[str, str]
    annotation_lookup_elapsed: float


@dataclass(frozen=True)
class IngestionScanSummary:
    payload: list[tuple[str, str | None, str, str, str, str]]
    scanned_cases: int
    stale_updates: int
    failed_cases: int
    signature_scan_elapsed: float


class ExtractionRepository:
    """SQLite-backed repository for Stage 2 case ingestion and processing."""

    def __init__(self, database_path: Path, tag: str) -> None:
        self.database_path = database_path
        self.tag = tag
        self.table_name = f"DATABASE_{tag}"

    def get_source_directories(self, source_folder: Path) -> tuple[Path, Path, Path]:
        """Return the Stage 2 source folder layout under one dataset root."""

        return (
            source_folder / "IMAGES",
            source_folder / "ANNOTATIONS",
            source_folder / "GEOJSON",
        )

    def initialize(self) -> None:
        """Create the Stage 2 database table when needed."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    STATUS TEXT DEFAULT 'TO BE PROCESSED',
                    IMAGEPATH TEXT NOT NULL,
                    ANNOTATIONPATH TEXT,
                    LastUpdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CANCER_QTD INTEGER,
                    NON_CANCER_QTD INTEGER,
                    VALID_IMAGE INTEGER,
                    PROCESSINGTIME_MINUTES REAL,
                    PATIENT TEXT NOT NULL UNIQUE,
                    COMMENTS TEXT,
                    WINDOW_SIZE INTEGER,
                    STRIDE INTEGER,
                    MATCH_PERCENTAGE TEXT,
                    TISSUE_PERCENTAGE TEXT
                )
                """
            )
            self._ensure_column(connection, "INPUT_SIGNATURE", "TEXT")
            self._ensure_column(connection, "PROCESSING_SIGNATURE", "TEXT")
            connection.commit()

    def _ensure_column(
        self, connection: sqlite3.Connection, column_name: str, column_sql: str
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({self.table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE {self.table_name} ADD COLUMN {column_name} {column_sql}"
            )

    def _build_input_signature(
        self,
        image_path: Path,
        annotation_path: Path | None,
    ) -> str:
        return hash_json_payload(
            {
                "image": build_file_metadata_fingerprint(image_path),
                "annotation": (
                    build_file_metadata_fingerprint(annotation_path)
                    if annotation_path is not None and annotation_path.exists()
                    else None
                ),
            }
        )

    def ingest_new_cases(self, options: IngestionOptions) -> None:
        """Scan case folders and insert unseen cases into the database."""

        ingestion_started_at = time.perf_counter()
        resources = self._prepare_ingestion_resources(options)

        with self._connect() as connection:
            existing_query_started_at = time.perf_counter()
            existing_index = self._load_existing_case_index(connection)
            existing_query_elapsed = time.perf_counter() - existing_query_started_at
            scan_summary = self._scan_ingestion_decisions(
                connection=connection,
                options=options,
                resources=resources,
                existing_index=existing_index,
            )

            write_elapsed = 0.0
            if scan_summary.payload:
                write_started_at = time.perf_counter()
                connection.executemany(
                    f"""
                    INSERT INTO {self.table_name}
                    (IMAGEPATH, ANNOTATIONPATH, PATIENT, STATUS, COMMENTS, INPUT_SIGNATURE)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    scan_summary.payload,
                )
                write_elapsed = time.perf_counter() - write_started_at
            commit_started_at = time.perf_counter()
            connection.commit()
            commit_elapsed = time.perf_counter() - commit_started_at

        total_elapsed = time.perf_counter() - ingestion_started_at
        LOGGER.info(
            "Stage 2 ingestion timing: slides=%d existing=%d inserted=%d stale_updates=%d "
            "failed=%d list_images=%.3fs index_annotations=%.3fs index_geojson=%.3fs "
            "query_existing=%.3fs "
            "scan_signatures=%.3fs write_rows=%.3fs commit=%.3fs total=%.3fs",
            scan_summary.scanned_cases,
            len(existing_index.rows),
            len(scan_summary.payload),
            scan_summary.stale_updates,
            scan_summary.failed_cases,
            resources.image_listing_elapsed,
            resources.annotation_lookup_elapsed,
            resources.geojson_lookup_elapsed,
            existing_query_elapsed,
            scan_summary.signature_scan_elapsed,
            write_elapsed,
            commit_elapsed,
            total_elapsed,
        )

    def _prepare_ingestion_resources(self, options: IngestionOptions) -> IngestionResources:
        images_folder, annotations_folder, _ = self.get_source_directories(options.source_folder)
        image_listing_started_at = time.perf_counter()
        image_files = sorted(path for path in images_folder.iterdir() if path.is_file())
        image_listing_elapsed = time.perf_counter() - image_listing_started_at
        if not image_files:
            raise FileNotFoundError(
                f"The directory '{images_folder}' is empty. Please add images to process."
            )

        run_geojson_check = options.activate_sanity_check and options.use_advanced_filtering
        geojson_lookup = None
        geojson_lookup_elapsed = 0.0
        if run_geojson_check:
            if options.geojson_path is None or not options.geojson_path.is_dir():
                raise FileNotFoundError(
                    "GeoJSON sanity check is active, but the source GEOJSON folder "
                    f"('{options.geojson_path}') is invalid."
                )
            geojson_lookup_started_at = time.perf_counter()
            geojson_lookup = GeoJsonLookup.from_directory(options.geojson_path)
            geojson_lookup_elapsed = time.perf_counter() - geojson_lookup_started_at

        annotation_lookup_started_at = time.perf_counter()
        annotation_lookup = {
            path.stem: str(path) for path in sorted(annotations_folder.iterdir()) if path.is_file()
        }
        annotation_lookup_elapsed = time.perf_counter() - annotation_lookup_started_at
        return IngestionResources(
            images_folder=images_folder,
            image_files=image_files,
            image_listing_elapsed=image_listing_elapsed,
            run_geojson_check=run_geojson_check,
            geojson_lookup=geojson_lookup,
            geojson_lookup_elapsed=geojson_lookup_elapsed,
            annotation_lookup=annotation_lookup,
            annotation_lookup_elapsed=annotation_lookup_elapsed,
        )

    def _scan_ingestion_decisions(
        self,
        *,
        connection: sqlite3.Connection,
        options: IngestionOptions,
        resources: IngestionResources,
        existing_index: ExistingCaseIndex,
    ) -> IngestionScanSummary:
        payload: list[tuple[str, str | None, str, str, str, str]] = []
        scanned_cases = 0
        stale_updates = 0
        failed_cases = 0
        next_patient_id = existing_index.next_patient_id
        signature_scan_started_at = time.perf_counter()
        for image_path in resources.image_files:
            scanned_cases += 1
            decision = self._build_ingestion_decision(
                image_path=image_path,
                annotation_lookup=resources.annotation_lookup,
                run_geojson_check=resources.run_geojson_check,
                geojson_path=options.geojson_path,
                geojson_lookup=resources.geojson_lookup,
                existing_row=existing_index.by_basename.get(image_path.name),
            )
            if decision.stale_row_id is not None:
                self._mark_case_stale(
                    connection,
                    decision.stale_row_id,
                    decision.input_signature,
                )
                stale_updates += 1
                continue
            if decision.status == "FAILED":
                failed_cases += 1

            payload.append(
                (
                    str(decision.image_path),
                    decision.annotation_path,
                    str(next_patient_id),
                    decision.status,
                    decision.comments,
                    decision.input_signature,
                )
            )
            next_patient_id += 1
        return IngestionScanSummary(
            payload=payload,
            scanned_cases=scanned_cases,
            stale_updates=stale_updates,
            failed_cases=failed_cases,
            signature_scan_elapsed=time.perf_counter() - signature_scan_started_at,
        )

    def _load_existing_case_index(self, connection: sqlite3.Connection) -> ExistingCaseIndex:
        rows = connection.execute(
            f"SELECT ID, IMAGEPATH, PATIENT, INPUT_SIGNATURE FROM {self.table_name}"
        ).fetchall()
        by_basename = {Path(str(row["IMAGEPATH"])).name: row for row in rows}
        existing_patients = {
            int(str(row["PATIENT"])) for row in rows if str(row["PATIENT"]).isdigit()
        }
        next_patient_id = max(existing_patients) + 1 if existing_patients else 100001
        return ExistingCaseIndex(
            rows=rows,
            by_basename=by_basename,
            next_patient_id=next_patient_id,
        )

    def _build_ingestion_decision(
        self,
        *,
        image_path: Path,
        annotation_lookup: dict[str, str],
        run_geojson_check: bool,
        geojson_path: Path | None,
        geojson_lookup: GeoJsonLookup | None,
        existing_row: sqlite3.Row | None,
    ) -> IngestionDecision:
        annotation_path = annotation_lookup.get(image_path.stem)
        input_signature = self._build_input_signature(
            image_path,
            Path(annotation_path) if annotation_path is not None else None,
        )
        stale_row_id = self._stale_row_id(existing_row, input_signature)
        if stale_row_id is not None:
            return IngestionDecision(
                image_path=image_path,
                annotation_path=annotation_path,
                input_signature=input_signature,
                status="STALE",
                comments=self._stale_comment(),
                stale_row_id=stale_row_id,
            )
        if existing_row is not None:
            return IngestionDecision(
                image_path=image_path,
                annotation_path=annotation_path,
                input_signature=input_signature,
                status="EXISTING",
                comments="",
            )
        status, comments = self._resolve_new_case_state(
            image_path=image_path,
            annotation_path=annotation_path,
            run_geojson_check=run_geojson_check,
            geojson_path=geojson_path,
            geojson_lookup=geojson_lookup,
        )
        return IngestionDecision(
            image_path=image_path,
            annotation_path=annotation_path,
            input_signature=input_signature,
            status=status,
            comments=comments,
        )

    def _stale_row_id(self, existing_row: sqlite3.Row | None, input_signature: str) -> int | None:
        if existing_row is None:
            return None
        previous_signature = str(existing_row["INPUT_SIGNATURE"] or "")
        if previous_signature and previous_signature != input_signature:
            return int(existing_row["ID"])
        return None

    def _mark_case_stale(
        self,
        connection: sqlite3.Connection,
        row_id: int,
        input_signature: str,
    ) -> None:
        connection.execute(
            f"""
            UPDATE {self.table_name}
            SET STATUS = 'STALE',
                COMMENTS = ?,
                INPUT_SIGNATURE = ?,
                LastUpdate = CURRENT_TIMESTAMP
            WHERE ID = ?
            """,
            (self._stale_comment(), input_signature, row_id),
        )

    def _stale_comment(self) -> str:
        return (
            "Input files changed for an existing case. "
            "Clear stale patch outputs and reprocess this slide."
        )

    def _resolve_new_case_state(
        self,
        *,
        image_path: Path,
        annotation_path: str | None,
        run_geojson_check: bool,
        geojson_path: Path | None,
        geojson_lookup: GeoJsonLookup | None,
    ) -> tuple[str, str]:
        if annotation_path is None:
            return "FAILED", "The equivalent annotation file could not be found."
        if not run_geojson_check:
            return "TO BE PROCESSED", ""
        assert geojson_path is not None
        try:
            resolved_geojson = resolve_geojson_for_slide(
                geojson_path,
                image_path,
                lookup=geojson_lookup,
            )
        except ValueError as error:
            return "FAILED", f"GeoJSON Sanity Check Failed: {error}"
        if resolved_geojson is None:
            return (
                "FAILED",
                "GeoJSON Sanity Check Failed: The equivalent GeoJSON file was not found.",
            )
        return "TO BE PROCESSED", ""

    def list_pending_cases(self) -> list[ExtractionCaseRecord]:
        """Return all cases still waiting for processing, ordered deterministically."""

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE STATUS = 'TO BE PROCESSED' ORDER BY ID ASC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_stale_cases(self) -> list[ExtractionCaseRecord]:
        """Return all cases explicitly marked stale due to changed inputs."""

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE STATUS = 'STALE' ORDER BY ID ASC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_cases_by_ids(self, case_ids: Sequence[int]) -> list[ExtractionCaseRecord]:
        """Return the requested cases ordered by the provided identifier sequence."""

        ordered_ids = [int(case_id) for case_id in case_ids]
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE ID IN ({placeholders})",
                ordered_ids,
            ).fetchall()
        rows_by_id = {int(row["ID"]): row for row in rows}
        return [
            self._row_to_record(rows_by_id[case_id])
            for case_id in ordered_ids
            if case_id in rows_by_id
        ]

    def list_completed_cases(self) -> list[ExtractionCaseRecord]:
        """Return all completed cases to validate processing lineage."""

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE STATUS = 'COMPLETED' ORDER BY ID ASC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def mark_processing(self, case_id: int) -> None:
        """Mark one case as currently being processed."""

        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE {self.table_name}
                SET STATUS = 'PROCESSING', LastUpdate = CURRENT_TIMESTAMP
                WHERE ID = ?
                """,
                (case_id,),
            )
            connection.commit()

    def update_case(self, case_id: int, update: CaseUpdate) -> None:
        """Persist the result of one case processing run."""

        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE {self.table_name}
                SET CANCER_QTD = ?,
                    NON_CANCER_QTD = ?,
                    PROCESSINGTIME_MINUTES = ?,
                    COMMENTS = ?,
                    STATUS = ?,
                    WINDOW_SIZE = ?,
                    STRIDE = ?,
                    MATCH_PERCENTAGE = ?,
                    TISSUE_PERCENTAGE = ?,
                    PROCESSING_SIGNATURE = ?,
                    LastUpdate = CURRENT_TIMESTAMP
                WHERE ID = ?
                """,
                (
                    update.cancer_qtd,
                    update.non_cancer_qtd,
                    update.exec_time_minutes,
                    update.comments,
                    update.status,
                    update.window_size,
                    update.stride,
                    str(update.match_percentage),
                    str(update.tissue_percentage),
                    update.processing_signature,
                    case_id,
                ),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def _row_to_record(self, row: sqlite3.Row) -> ExtractionCaseRecord:
        return ExtractionCaseRecord(
            record_id=int(row["ID"]),
            status=str(row["STATUS"]),
            image_path=Path(str(row["IMAGEPATH"])),
            annotation_path=Path(str(row["ANNOTATIONPATH"])) if row["ANNOTATIONPATH"] else None,
            cancer_qtd=int(row["CANCER_QTD"]) if row["CANCER_QTD"] is not None else None,
            non_cancer_qtd=int(row["NON_CANCER_QTD"])
            if row["NON_CANCER_QTD"] is not None
            else None,
            valid_image=int(row["VALID_IMAGE"]) if row["VALID_IMAGE"] is not None else None,
            processing_time_minutes=float(row["PROCESSINGTIME_MINUTES"])
            if row["PROCESSINGTIME_MINUTES"] is not None
            else None,
            patient=str(row["PATIENT"]),
            comments=str(row["COMMENTS"] or ""),
            window_size=int(row["WINDOW_SIZE"]) if row["WINDOW_SIZE"] is not None else None,
            stride=int(row["STRIDE"]) if row["STRIDE"] is not None else None,
            match_percentage=str(row["MATCH_PERCENTAGE"]) if row["MATCH_PERCENTAGE"] else None,
            tissue_percentage=str(row["TISSUE_PERCENTAGE"]) if row["TISSUE_PERCENTAGE"] else None,
            last_update=str(row["LastUpdate"]) if row["LastUpdate"] else None,
            input_signature=str(row["INPUT_SIGNATURE"]) if row["INPUT_SIGNATURE"] else None,
            processing_signature=(
                str(row["PROCESSING_SIGNATURE"]) if row["PROCESSING_SIGNATURE"] else None
            ),
        )
