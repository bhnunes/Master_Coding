from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ARTIFACT_INDEX_COLUMNS = (
    "filename",
    "label",
    "patient_id",
    "slide_id",
    "cov_fold",
    "cov_penmarking",
    "cov_oof",
    "cov_darkspot_foreign",
    "cov_edge_airbubble",
)

ARTIFACT_INDEX_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("label", pa.int8()),
        ("patient_id", pa.string()),
        ("slide_id", pa.string()),
        ("cov_fold", pa.float32()),
        ("cov_penmarking", pa.float32()),
        ("cov_oof", pa.float32()),
        ("cov_darkspot_foreign", pa.float32()),
        ("cov_edge_airbubble", pa.float32()),
    ]
)


@dataclass(frozen=True)
class ArtifactPatchRecord:
    """Stable artifact metadata stored for each saved patch."""

    filename: str
    label: int
    patient_id: str
    slide_id: str
    cov_fold: float
    cov_penmarking: float
    cov_oof: float
    cov_darkspot_foreign: float
    cov_edge_airbubble: float


class ArtifactIndexWriter:
    """Append-only Parquet writer for Stage 2 artifact metadata."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def append_records(self, records: list[ArtifactPatchRecord]) -> None:
        if not records:
            return
        table = pa.Table.from_pylist(
            [asdict(record) for record in records], schema=ARTIFACT_INDEX_SCHEMA
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.output_path, ARTIFACT_INDEX_SCHEMA)
        self._writer.write_table(table)
        self._rows_written += len(records)

    def close(self) -> None:
        if self._writer is None:
            empty_table = pa.Table.from_pylist([], schema=ARTIFACT_INDEX_SCHEMA)
            pq.write_table(empty_table, self.output_path)
            return
        self._writer.close()
        self._writer = None
