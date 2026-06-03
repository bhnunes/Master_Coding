from pathlib import Path

from helpers.artifact.pipeline import ArtifactDetectionPipeline
from helpers.artifact.repository import ArtifactRecord


class FakeZipSource:
    def __init__(self) -> None:
        self.extracted: list[str] = []

    def list_slide_members(self) -> list[str]:
        return ["case_1.svs", "case_2.ndpi"]

    def list_slide_members_with_signatures(self) -> list[tuple[str, str]]:
        return [("case_1.svs", "sig-1"), ("case_2.ndpi", "sig-2")]

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
                member_signature="sig-1",
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
                member_signature="sig-2",
            ),
        ]
        self.synced_members: list[str] = []
        self.success_ids: list[int] = []
        self.failures: list[tuple[int, str, str]] = []

    def initialize(self) -> None:
        return None

    def sync_members(self, members: list[tuple[str, str]]) -> None:
        self.synced_members = [member for member, _signature in members]

    def list_pending(self) -> list[ArtifactRecord]:
        return self.records

    def mark_processing(self, _record_id: int) -> None:
        return None

    def mark_success(
        self, record_id: int, geojson_path: str, processing_time_seconds: float | None = None
    ) -> None:
        del geojson_path, processing_time_seconds
        self.success_ids.append(record_id)

    def mark_failure(
        self,
        record_id: int,
        error_type: str,
        comments: str,
        processing_time_seconds: float | None = None,
    ) -> None:
        del processing_time_seconds
        self.failures.append((record_id, error_type, comments))


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def process_slide(
        self, _slide_path: Path, zip_member_name: str, geojson_output_path: Path
    ) -> None:
        self.calls.append(zip_member_name)
        if zip_member_name == "case_2.ndpi":
            raise RuntimeError("broken slide")
        geojson_output_path.write_text("{}")


class FakeLogger:
    def info(self, _message: str, *_args: object) -> None:
        return None

    def warning(self, _message: str, *_args: object) -> None:
        return None

    def error(self, _message: str, *_args: object) -> None:
        return None

    def exception(self, _message: str, *_args: object) -> None:
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
