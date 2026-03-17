from pathlib import Path

from helpers.artifact_pipeline import ArtifactDetectionPipeline
from helpers.artifact_repository import ArtifactRecord


class FakeZipSource:
    def __init__(self) -> None:
        self.extracted: list[str] = []

    def list_slide_members(self) -> list[str]:
        return ["case_1.svs", "case_2.ndpi"]

    def extract_member(self, member_name: str, destination: Path) -> Path:
        self.extracted.append(member_name)
        path = destination / Path(member_name).name
        path.write_text("slide")
        return path


class FakeRepository:
    def __init__(self) -> None:
        self.records = [
            ArtifactRecord(
                record_id=1,
                image_name="case_1.svs",
                zip_member_path="case_1.svs",
                geojson_processed=False,
                comments="",
                status="PENDING",
                geojson_path=None,
                error_type=None,
                processing_time_seconds=None,
                last_update="",
            ),
            ArtifactRecord(
                record_id=2,
                image_name="case_2.ndpi",
                zip_member_path="case_2.ndpi",
                geojson_processed=False,
                comments="",
                status="PENDING",
                geojson_path=None,
                error_type=None,
                processing_time_seconds=None,
                last_update="",
            ),
        ]
        self.synced_members: list[str] = []
        self.success_ids: list[int] = []
        self.failures: list[tuple[int, str, str]] = []

    def initialize(self) -> None:
        return None

    def sync_members(self, members: list[str]) -> None:
        self.synced_members = members

    def list_pending(self) -> list[ArtifactRecord]:
        return self.records

    def mark_processing(self, record_id: int) -> None:
        return None

    def mark_success(
        self, record_id: int, geojson_path: str, processing_time_seconds: float | None = None
    ) -> None:
        self.success_ids.append(record_id)

    def mark_failure(
        self,
        record_id: int,
        error_type: str,
        comments: str,
        processing_time_seconds: float | None = None,
    ) -> None:
        self.failures.append((record_id, error_type, comments))


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def process_slide(
        self, slide_path: Path, zip_member_name: str, geojson_output_path: Path
    ) -> None:
        self.calls.append(zip_member_name)
        if zip_member_name == "case_2.ndpi":
            raise RuntimeError("broken slide")
        geojson_output_path.write_text("{}")


class FakeLogger:
    def info(self, message: str, *args: object) -> None:
        return None

    def warning(self, message: str, *args: object) -> None:
        return None

    def error(self, message: str, *args: object) -> None:
        return None

    def exception(self, message: str, *args: object) -> None:
        return None


def test_pipeline_processes_all_pending_records_and_continues_after_failure(tmp_path: Path) -> None:
    repository = FakeRepository()
    processor = FakeProcessor()
    pipeline = ArtifactDetectionPipeline(
        zip_source=FakeZipSource(),
        repository=repository,
        processor=processor,
        temp_root=tmp_path / "temp",
        geojson_output=tmp_path / "geojson",
        logger=FakeLogger(),
    )

    summary = pipeline.run()

    assert repository.synced_members == ["case_1.svs", "case_2.ndpi"]
    assert repository.success_ids == [1]
    assert repository.failures == [(2, "RuntimeError", "broken slide")]
    assert summary.processed == 1
    assert summary.failed == 1
