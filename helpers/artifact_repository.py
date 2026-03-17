from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactRecord:
    """One slide tracked by the artifact detection database."""

    record_id: int
    image_name: str
    zip_member_path: str
    geojson_processed: bool
    comments: str
    status: str
    geojson_path: str | None
    error_type: str | None
    processing_time_seconds: float | None
    last_update: str


class ArtifactRepository:
    """SQLite-backed progress tracking for artifact detection."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create the database schema if needed."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_detection (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Image_Name TEXT NOT NULL UNIQUE,
                    Zip_Member_Path TEXT NOT NULL,
                    GeoJSON_Processed INTEGER NOT NULL DEFAULT 0,
                    GeoJSON_Path TEXT,
                    Status TEXT NOT NULL DEFAULT 'PENDING',
                    Error_Type TEXT,
                    Comments TEXT NOT NULL DEFAULT '',
                    Processing_Time_Seconds REAL,
                    LastUpdate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def sync_members(self, members: list[str]) -> None:
        """Insert new slide members into the repository without duplication."""

        payload = [(member, member) for member in members]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO artifact_detection (Image_Name, Zip_Member_Path)
                VALUES (?, ?)
                """,
                payload,
            )
            connection.commit()

    def list_pending(self) -> list[ArtifactRecord]:
        """Return slides that still need processing."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM artifact_detection
                WHERE GeoJSON_Processed = 0
                ORDER BY ID ASC
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> list[ArtifactRecord]:
        """Return all artifact detection records."""

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifact_detection ORDER BY ID ASC").fetchall()
        return [self._row_to_record(row) for row in rows]

    def mark_processing(self, record_id: int) -> None:
        """Mark a slide as being processed."""

        self._update_status(record_id, status="PROCESSING")

    def mark_success(
        self,
        record_id: int,
        geojson_path: str,
        processing_time_seconds: float | None = None,
    ) -> None:
        """Mark a slide as successfully processed."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE artifact_detection
                SET GeoJSON_Processed = 1,
                    GeoJSON_Path = ?,
                    Status = 'PROCESSED',
                    Error_Type = NULL,
                    Comments = '',
                    Processing_Time_Seconds = ?,
                    LastUpdate = CURRENT_TIMESTAMP
                WHERE ID = ?
                """,
                (geojson_path, processing_time_seconds, record_id),
            )
            connection.commit()

    def mark_failure(
        self,
        record_id: int,
        error_type: str,
        comments: str,
        processing_time_seconds: float | None = None,
    ) -> None:
        """Mark a slide as failed and store the error details."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE artifact_detection
                SET GeoJSON_Processed = 0,
                    Status = 'FAILED',
                    Error_Type = ?,
                    Comments = ?,
                    Processing_Time_Seconds = ?,
                    LastUpdate = CURRENT_TIMESTAMP
                WHERE ID = ?
                """,
                (error_type, comments, processing_time_seconds, record_id),
            )
            connection.commit()

    def _update_status(self, record_id: int, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE artifact_detection
                SET Status = ?, LastUpdate = CURRENT_TIMESTAMP
                WHERE ID = ?
                """,
                (status, record_id),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def _row_to_record(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            record_id=int(row["ID"]),
            image_name=str(row["Image_Name"]),
            zip_member_path=str(row["Zip_Member_Path"]),
            geojson_processed=bool(row["GeoJSON_Processed"]),
            comments=str(row["Comments"]),
            status=str(row["Status"]),
            geojson_path=str(row["GeoJSON_Path"]) if row["GeoJSON_Path"] else None,
            error_type=str(row["Error_Type"]) if row["Error_Type"] else None,
            processing_time_seconds=float(row["Processing_Time_Seconds"])
            if row["Processing_Time_Seconds"] is not None
            else None,
            last_update=str(row["LastUpdate"]),
        )
