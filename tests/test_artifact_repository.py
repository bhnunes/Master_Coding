from pathlib import Path
from sqlite3 import Row
from typing import cast
from unittest.mock import patch

import pytest

from helpers.artifact.repository import ArtifactRepository

RECORD_ID = 7
PROCESSING_TIME_SECONDS = 3.5
SUCCESS_PROCESSING_TIME_SECONDS = 1.25


def test_repository_initializes_schema_and_tracks_pending_members(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    repository.initialize()
    repository.sync_members(["case_1.svs", "nested/case_2.ndpi"])

    pending = repository.list_pending()

    assert [record.image_name for record in pending] == ["case_1.svs", "nested/case_2.ndpi"]


def test_repository_marks_success_and_failure(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members(["case_1.svs", "case_2.ndpi"])

    first, second = repository.list_pending()
    repository.mark_success(first.record_id, "/tmp/case_1.geojson")
    repository.mark_failure(second.record_id, "RuntimeError", "boom")
    records = repository.list_all()

    assert records[0].geojson_processed is True
    assert records[0].status == "PROCESSED"
    assert records[1].geojson_processed is False
    assert records[1].status == "FAILED"
    assert records[1].comments == "boom"


def test_repository_requeues_processed_member_when_signature_changes(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members([("nested/case_1.svs", "sig-a")])

    record = repository.list_pending()[0]
    repository.mark_success(record.record_id, "/tmp/case_1.geojson")
    repository.sync_members([("nested/case_1.svs", "sig-b")])

    refreshed = repository.list_pending()[0]
    assert refreshed.zip_member_path == "nested/case_1.svs"
    assert refreshed.geojson_processed is False
    assert refreshed.status == "PENDING"
    assert (
        refreshed.comments
        == "Zip member content changed; artifact GeoJSON must be regenerated."
    )
    assert refreshed.error_type == "STALE_INPUT"
    assert refreshed.geojson_path is None
    assert refreshed.processing_time_seconds is None
    assert refreshed.member_signature == "sig-b"


def test_repository_initialize_creates_parent_directory_and_signature_column(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "deeper" / "artifact_detection.db"
    repository = ArtifactRepository(database_path)

    repository.initialize()

    assert database_path.parent.is_dir()
    with repository._connect() as connection:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(artifact_detection)").fetchall()
        }

    assert "Member_Signature" in columns


def test_repository_initialize_preserves_member_signature_text_type(tmp_path: Path) -> None:
    database_path = tmp_path / "artifact_detection.db"
    repository = ArtifactRepository(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TABLE artifact_detection (
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

    repository.initialize()

    with repository._connect() as connection:
        column_types = {
            str(row["name"]): str(row["type"])
            for row in connection.execute("PRAGMA table_info(artifact_detection)").fetchall()
        }

    assert column_types["Member_Signature"] == "TEXT"


def test_repository_initialize_adds_member_signature_to_legacy_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "artifact_detection.db"
    repository = ArtifactRepository(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TABLE artifact_detection (
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

    repository.initialize()

    with repository._connect() as connection:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(artifact_detection)").fetchall()
        }

    assert "Member_Signature" in columns


def test_repository_initialize_is_idempotent_after_member_signature_exists(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    repository.initialize()
    repository.initialize()

    with repository._connect() as connection:
        member_signature_columns = [
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(artifact_detection)").fetchall()
            if str(row["name"]) == "Member_Signature"
        ]

    assert member_signature_columns == ["Member_Signature"]


def test_repository_sync_members_uses_member_name_as_default_signature(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()

    repository.sync_members(["nested/case_1.svs"])

    record = repository.list_pending()[0]
    assert record.image_name == "nested/case_1.svs"
    assert record.member_signature == "nested/case_1.svs"


def test_repository_sync_members_keeps_existing_row_when_signature_is_unchanged(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members([("nested/case_1.svs", "sig-a")])
    record = repository.list_pending()[0]
    repository.mark_success(
        record.record_id,
        "/tmp/case_1.geojson",
        processing_time_seconds=SUCCESS_PROCESSING_TIME_SECONDS,
    )

    repository.sync_members([("nested/case_1.svs", "sig-a")])

    records = repository.list_all()
    assert len(records) == 1
    assert records[0].status == "PROCESSED"
    assert records[0].geojson_processed is True
    assert records[0].geojson_path == "/tmp/case_1.geojson"
    assert records[0].processing_time_seconds == SUCCESS_PROCESSING_TIME_SECONDS
    assert records[0].member_signature == "sig-a"


def test_repository_mark_processing_sets_status_without_removing_pending_state(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members(["case_1.svs"])
    record = repository.list_pending()[0]

    repository.mark_processing(record.record_id)

    refreshed = repository.list_pending()[0]
    assert refreshed.record_id == record.record_id
    assert refreshed.status == "PROCESSING"
    assert refreshed.geojson_processed is False


def test_repository_mark_success_clears_error_fields_and_sets_processing_time(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members(["case_1.svs"])
    record = repository.list_pending()[0]
    repository.mark_failure(record.record_id, "RuntimeError", "boom", processing_time_seconds=2.0)

    repository.mark_success(
        record.record_id,
        "/tmp/case_1.geojson",
        processing_time_seconds=SUCCESS_PROCESSING_TIME_SECONDS,
    )

    refreshed = repository.list_all()[0]
    assert refreshed.status == "PROCESSED"
    assert refreshed.geojson_processed is True
    assert refreshed.error_type is None
    assert refreshed.comments == ""
    assert refreshed.geojson_path == "/tmp/case_1.geojson"
    assert refreshed.processing_time_seconds == SUCCESS_PROCESSING_TIME_SECONDS


def test_repository_mark_failure_records_error_fields_and_time(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members(["case_1.svs"])
    record = repository.list_pending()[0]

    repository.mark_failure(
        record.record_id,
        "RuntimeError",
        "boom",
        processing_time_seconds=PROCESSING_TIME_SECONDS,
    )

    refreshed = repository.list_all()[0]
    assert refreshed.status == "FAILED"
    assert refreshed.geojson_processed is False
    assert refreshed.error_type == "RuntimeError"
    assert refreshed.comments == "boom"
    assert refreshed.processing_time_seconds == PROCESSING_TIME_SECONDS


def test_repository_list_pending_excludes_processed_rows_and_preserves_order(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")
    repository.initialize()
    repository.sync_members(["case_1.svs", "case_2.ndpi", "case_3.tif"])
    first, second, third = repository.list_pending()
    repository.mark_success(second.record_id, "/tmp/case_2.geojson")

    pending = repository.list_pending()

    assert [record.record_id for record in pending] == [first.record_id, third.record_id]
    assert [record.image_name for record in pending] == ["case_1.svs", "case_3.tif"]


def test_repository_connect_uses_sqlite_row_factory(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    with repository._connect() as connection:
        row = connection.execute("SELECT 1 AS value").fetchone()

    assert isinstance(row, Row)
    assert row["value"] == 1


def test_repository_connect_uses_expected_timeout(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    with patch("helpers.artifact.repository.sqlite3.connect") as connect:
        connection = connect.return_value
        repository._connect()

    connect.assert_called_once_with(repository.database_path, timeout=20)
    assert connection.row_factory is Row


def test_row_to_record_requires_exact_repository_column_names(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    record = repository._row_to_record(
        cast(
            Row,
            {
                "ID": RECORD_ID,
                "Image_Name": "nested/case_1.svs",
                "Zip_Member_Path": "nested/case_1.svs",
                "GeoJSON_Processed": 1,
                "Comments": "processed",
                "Status": "PROCESSED",
                "GeoJSON_Path": "/tmp/case_1.geojson",
                "Error_Type": None,
                "Processing_Time_Seconds": PROCESSING_TIME_SECONDS,
                "LastUpdate": "2026-04-21 12:34:56",
                "Member_Signature": "sig-a",
            },
        )
    )

    assert record.record_id == RECORD_ID
    assert record.image_name == "nested/case_1.svs"
    assert record.zip_member_path == "nested/case_1.svs"
    assert record.geojson_processed is True
    assert record.comments == "processed"
    assert record.status == "PROCESSED"
    assert record.geojson_path == "/tmp/case_1.geojson"
    assert record.error_type is None
    assert record.processing_time_seconds == PROCESSING_TIME_SECONDS
    assert record.last_update == "2026-04-21 12:34:56"
    assert record.member_signature == "sig-a"


def test_row_to_record_sets_nullable_fields_to_none(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    record = repository._row_to_record(
        cast(
            Row,
            {
                "ID": 8,
                "Image_Name": "case_2.ndpi",
                "Zip_Member_Path": "case_2.ndpi",
                "GeoJSON_Processed": 0,
                "Comments": "",
                "Status": "PENDING",
                "GeoJSON_Path": None,
                "Error_Type": None,
                "Processing_Time_Seconds": None,
                "LastUpdate": "2026-04-21 12:35:00",
                "Member_Signature": None,
            },
        )
    )

    assert record.geojson_processed is False
    assert record.geojson_path is None
    assert record.error_type is None
    assert record.processing_time_seconds is None
    assert record.member_signature is None


def test_row_to_record_rejects_wrong_column_casing(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    with pytest.raises(KeyError):
        repository._row_to_record(
            cast(
                Row,
                {
                    "ID": 1,
                    "Image_Name": "case_1.svs",
                    "Zip_Member_Path": "case_1.svs",
                    "GeoJSON_Processed": 0,
                    "Comments": "",
                    "Status": "PENDING",
                    "GeoJSON_Path": None,
                    "Error_Type": None,
                    "Processing_Time_Seconds": None,
                    "LASTUPDATE": "2026-04-21 12:35:00",
                    "Member_Signature": None,
                },
            )
        )


def test_row_to_record_reads_error_type_from_exact_repository_column_name(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "artifact_detection.db")

    record = repository._row_to_record(
        cast(
            Row,
            {
                "ID": 1,
                "Image_Name": "case_1.svs",
                "Zip_Member_Path": "case_1.svs",
                "GeoJSON_Processed": 0,
                "Comments": "boom",
                "Status": "FAILED",
                "GeoJSON_Path": None,
                "Error_Type": "RuntimeError",
                "Processing_Time_Seconds": None,
                "LastUpdate": "2026-04-21 12:35:00",
                "Member_Signature": None,
            },
        )
    )

    assert record.error_type == "RuntimeError"
