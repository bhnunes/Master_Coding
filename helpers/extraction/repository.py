from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from helpers.extraction.artifact_lookup import resolve_geojson_for_slide
from helpers.provenance import hash_file_sha256, hash_json_payload


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
    cancer_color: str | None
    not_cancer_color: str | None
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
                    CANCER_COLOR TEXT,
                    NOT_CANCER_COLOR TEXT,
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
                "image_path": str(image_path),
                "image_sha256": hash_file_sha256(image_path),
                "annotation_path": str(annotation_path) if annotation_path is not None else None,
                "annotation_sha256": (
                    hash_file_sha256(annotation_path)
                    if annotation_path is not None and annotation_path.exists()
                    else None
                ),
            }
        )

    def ingest_new_cases(
        self,
        source_folder: Path,
        activate_sanity_check: bool,
        use_advanced_filtering: bool,
        geojson_path: Path | None,
    ) -> bool:
        """Scan case folders and insert unseen cases into the database."""

        images_folder, annotations_folder, _ = self.get_source_directories(source_folder)
        image_files = sorted(path for path in images_folder.iterdir() if path.is_file())
        if not image_files:
            raise FileNotFoundError(
                f"The directory '{images_folder}' is empty. Please add images to process."
            )

        run_geojson_check = activate_sanity_check and use_advanced_filtering
        if run_geojson_check:
            if geojson_path is None or not geojson_path.is_dir():
                raise FileNotFoundError(
                    "GeoJSON sanity check is active, but the source GEOJSON folder "
                    f"('{geojson_path}') is invalid."
                )

        annotation_lookup = {
            path.stem: str(path) for path in sorted(annotations_folder.iterdir()) if path.is_file()
        }

        svs_files_added = False
        with self._connect() as connection:
            existing_data = connection.execute(
                f"SELECT ID, IMAGEPATH, PATIENT, INPUT_SIGNATURE FROM {self.table_name}"
            ).fetchall()
            existing_by_basename = {Path(str(row["IMAGEPATH"])).name: row for row in existing_data}
            existing_patients = {
                int(str(row["PATIENT"])) for row in existing_data if str(row["PATIENT"]).isdigit()
            }

            next_patient_id = max(existing_patients) + 1 if existing_patients else 100001
            payload: list[tuple[str, str | None, str, str, str, str]] = []
            for image_path in image_files:
                annotation_path = annotation_lookup.get(image_path.stem)
                input_signature = self._build_input_signature(
                    image_path,
                    Path(annotation_path) if annotation_path is not None else None,
                )
                existing_row = existing_by_basename.get(image_path.name)
                if existing_row is not None:
                    previous_signature = str(existing_row["INPUT_SIGNATURE"] or "")
                    if previous_signature and previous_signature != input_signature:
                        connection.execute(
                            f"""
                            UPDATE {self.table_name}
                            SET STATUS = 'STALE',
                                COMMENTS = ?,
                                INPUT_SIGNATURE = ?,
                                LastUpdate = CURRENT_TIMESTAMP
                            WHERE ID = ?
                            """,
                            (
                                "Input files changed for an existing case. "
                                "Clear stale patch outputs and reprocess this slide.",
                                input_signature,
                                int(existing_row["ID"]),
                            ),
                        )
                    continue

                if image_path.suffix.lower() == ".svs":
                    svs_files_added = True

                status = "TO BE PROCESSED"
                comments = ""
                if annotation_path is None:
                    status = "FAILED"
                    comments = "The equivalent annotation file could not be found."
                elif run_geojson_check:
                    assert geojson_path is not None
                    try:
                        resolved_geojson = resolve_geojson_for_slide(geojson_path, image_path)
                    except ValueError as error:
                        status = "FAILED"
                        comments = f"GeoJSON Sanity Check Failed: {error}"
                    else:
                        if resolved_geojson is None:
                            status = "FAILED"
                            comments = (
                                "GeoJSON Sanity Check Failed: "
                                "The equivalent GeoJSON file was not found."
                            )

                payload.append(
                    (
                        str(image_path),
                        annotation_path,
                        str(next_patient_id),
                        status,
                        comments,
                        input_signature,
                    )
                )
                next_patient_id += 1

            if payload:
                connection.executemany(
                    f"""
                    INSERT INTO {self.table_name}
                    (IMAGEPATH, ANNOTATIONPATH, PATIENT, STATUS, COMMENTS, INPUT_SIGNATURE)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
            connection.commit()

        return svs_files_added

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
            cancer_color=str(row["CANCER_COLOR"]) if row["CANCER_COLOR"] else None,
            not_cancer_color=str(row["NOT_CANCER_COLOR"]) if row["NOT_CANCER_COLOR"] else None,
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
