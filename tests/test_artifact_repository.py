from pathlib import Path

from helpers.artifact.repository import ArtifactRepository


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
    assert "must be regenerated" in refreshed.comments
