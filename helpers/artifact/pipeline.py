from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from helpers.artifact.paths import cleanup_directory, create_slide_temp_dir, geojson_output_path
from helpers.artifact.repository import ArtifactRecord


@dataclass(frozen=True)
class ArtifactPipelineSummary:
    """Execution summary for one artifact detection run."""

    discovered: int
    processed: int
    failed: int


class ArtifactZipSource(Protocol):
    def list_slide_members(self) -> list[str]: ...

    def list_slide_members_with_signatures(self) -> list[tuple[str, str]]: ...

    def extract_member(self, member_name: str, destination: Path) -> Path: ...


class ArtifactRepositoryProtocol(Protocol):
    def initialize(self) -> None: ...

    def sync_members(self, members: list[tuple[str, str]]) -> None: ...

    def list_pending(self) -> list[ArtifactRecord]: ...

    def mark_processing(self, record_id: int) -> None: ...

    def mark_success(
        self,
        record_id: int,
        geojson_path: str,
        processing_time_seconds: float | None = None,
    ) -> None: ...

    def mark_failure(
        self,
        record_id: int,
        error_type: str,
        comments: str,
        processing_time_seconds: float | None = None,
    ) -> None: ...


class ArtifactProcessorProtocol(Protocol):
    def process_slide(
        self, slide_path: Path, zip_member_name: str, geojson_output_path: Path
    ) -> None: ...


class ArtifactLogger(Protocol):
    def info(self, *args: Any, **kwargs: Any) -> None: ...

    def exception(self, *args: Any, **kwargs: Any) -> None: ...


class ArtifactDetectionPipeline:
    """Coordinate zip ingestion, per-slide processing, and persistence."""

    def __init__(
        self,
        zip_source: ArtifactZipSource,
        repository: ArtifactRepositoryProtocol,
        processor: ArtifactProcessorProtocol,
        temp_root: Path,
        geojson_output: Path,
        logger: ArtifactLogger,
    ) -> None:
        self.zip_source = zip_source
        self.repository = repository
        self.processor = processor
        self.temp_root = temp_root
        self.geojson_output = geojson_output
        self.logger = logger

    def run(self) -> ArtifactPipelineSummary:
        """Execute artifact detection for every pending slide."""

        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.geojson_output.mkdir(parents=True, exist_ok=True)

        self.repository.initialize()
        members_with_signatures = self.zip_source.list_slide_members_with_signatures()
        self.repository.sync_members(members_with_signatures)
        members = [member_name for member_name, _signature in members_with_signatures]
        pending_records = self.repository.list_pending()

        processed = 0
        failed = 0
        for record in pending_records:
            slide_temp_dir = create_slide_temp_dir(self.temp_root, record.zip_member_path)
            output_path = geojson_output_path(self.geojson_output, record.zip_member_path)
            start_time = time.perf_counter()
            try:
                self.repository.mark_processing(record.record_id)
                slide_path = self.zip_source.extract_member(record.zip_member_path, slide_temp_dir)
                self.processor.process_slide(slide_path, record.zip_member_path, output_path)
                self.repository.mark_success(
                    record.record_id,
                    str(output_path),
                    processing_time_seconds=time.perf_counter() - start_time,
                )
                processed += 1
            except Exception as error:
                failed += 1
                self.repository.mark_failure(
                    record.record_id,
                    type(error).__name__,
                    str(error),
                    processing_time_seconds=time.perf_counter() - start_time,
                )
                self.logger.exception("Failed to process %s", record.image_name)
            finally:
                cleanup_directory(slide_temp_dir)

        return ArtifactPipelineSummary(
            discovered=len(members),
            processed=processed,
            failed=failed,
        )
