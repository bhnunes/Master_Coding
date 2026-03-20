from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from helpers.extraction.artifact_index import ArtifactIndexWriter, ArtifactPatchRecord


def test_artifact_index_writer_writes_expected_schema_and_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "artifact_patch_index.parquet"
    writer = ArtifactIndexWriter(output_path)

    writer.append_records(
        [
            ArtifactPatchRecord(
                filename="CANCER_PATIENT_101_0_0_0001.png",
                label=1,
                patient_id="101",
                slide_id="slide_a",
                cov_fold=0.25,
                cov_penmarking=0.0,
                cov_oof=0.1,
                cov_darkspot_foreign=0.0,
                cov_edge_airbubble=0.05,
            )
        ]
    )
    writer.close()

    table = pq.read_table(output_path)

    assert table.column_names == [
        "filename",
        "label",
        "patient_id",
        "slide_id",
        "cov_fold",
        "cov_penmarking",
        "cov_oof",
        "cov_darkspot_foreign",
        "cov_edge_airbubble",
    ]
    assert table.num_rows == 1
    assert table["filename"].to_pylist() == ["CANCER_PATIENT_101_0_0_0001.png"]
    assert table["cov_fold"].to_pylist() == [0.25]


def test_artifact_index_writer_creates_empty_file_with_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "artifact_patch_index.parquet"
    writer = ArtifactIndexWriter(output_path)

    writer.close()

    table = pq.read_table(output_path)

    assert table.num_rows == 0
    assert "cov_edge_airbubble" in table.column_names
