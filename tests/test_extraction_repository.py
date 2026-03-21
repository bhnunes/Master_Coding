import sqlite3
from pathlib import Path

from helpers.extraction.repository import ExtractionRepository


def test_repository_ingests_cases_and_marks_geojson_mismatches(tmp_path: Path) -> None:
    base_path = tmp_path / "project"
    repository = ExtractionRepository(database_path=tmp_path / "database.db", tag="TEST")

    images_dir, annotations_dir, geojson_dir = repository.ensure_case_directories(base_path)
    repository.initialize()

    (images_dir / "case_a.svs").write_text("slide")
    (annotations_dir / "case_a.xml").write_text("annotation")
    (images_dir / "case_b.svs").write_text("slide")
    (annotations_dir / "case_b.xml").write_text("annotation")
    (geojson_dir / "case_a.geojson").write_text("{}")

    svs_added = repository.ingest_new_cases(
        base_path=base_path,
        activate_sanity_check=True,
        use_advanced_filtering=True,
        geojson_path=geojson_dir,
    )

    assert svs_added is True

    with sqlite3.connect(tmp_path / "database.db") as connection:
        rows = connection.execute(
            "SELECT IMAGEPATH, STATUS, COMMENTS, PATIENT FROM DATABASE_TEST ORDER BY ID"
        ).fetchall()

    assert rows == [
        (
            str(images_dir / "case_a.svs"),
            "TO BE PROCESSED",
            "",
            "100001",
        ),
        (
            str(images_dir / "case_b.svs"),
            "FAILED",
            "GeoJSON Sanity Check Failed: The equivalent GeoJSON file was not found.",
            "100002",
        ),
    ]


def test_repository_lists_pending_cases_in_id_order(tmp_path: Path) -> None:
    repository = ExtractionRepository(database_path=tmp_path / "database.db", tag="TEST")
    repository.initialize()

    with sqlite3.connect(tmp_path / "database.db") as connection:
        connection.execute(
            "INSERT INTO DATABASE_TEST "
            "(IMAGEPATH, ANNOTATIONPATH, PATIENT, STATUS, COMMENTS) VALUES (?, ?, ?, ?, ?)",
            ("/tmp/b.svs", "/tmp/b.xml", "100002", "TO BE PROCESSED", ""),
        )
        connection.execute(
            "INSERT INTO DATABASE_TEST "
            "(IMAGEPATH, ANNOTATIONPATH, PATIENT, STATUS, COMMENTS) VALUES (?, ?, ?, ?, ?)",
            ("/tmp/a.svs", "/tmp/a.xml", "100001", "TO BE PROCESSED", ""),
        )
        connection.commit()

    pending_ids = [case.record_id for case in repository.list_pending_cases()]
    assert pending_ids == [1, 2]


def test_repository_marks_existing_case_stale_when_inputs_change(tmp_path: Path) -> None:
    base_path = tmp_path / "project"
    repository = ExtractionRepository(database_path=tmp_path / "database.db", tag="TEST")

    images_dir, annotations_dir, _geojson_dir = repository.ensure_case_directories(base_path)
    repository.initialize()

    image_path = images_dir / "case_a.svs"
    annotation_path = annotations_dir / "case_a.xml"
    image_path.write_text("slide-v1")
    annotation_path.write_text("annotation-v1")

    repository.ingest_new_cases(
        base_path=base_path,
        activate_sanity_check=False,
        use_advanced_filtering=False,
        geojson_path=None,
    )

    annotation_path.write_text("annotation-v2")
    repository.ingest_new_cases(
        base_path=base_path,
        activate_sanity_check=False,
        use_advanced_filtering=False,
        geojson_path=None,
    )

    with sqlite3.connect(tmp_path / "database.db") as connection:
        row = connection.execute(
            "SELECT STATUS, COMMENTS FROM DATABASE_TEST WHERE IMAGEPATH = ?",
            (str(image_path),),
        ).fetchone()

    assert row == (
        "STALE",
        "Input files changed for an existing case. "
        "Clear stale patch outputs and reprocess this slide.",
    )
