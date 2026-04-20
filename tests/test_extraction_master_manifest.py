import sqlite3
from pathlib import Path

import h5py
import numpy as np

from helpers.extraction.hdf5_storage import write_slide_patch_dataset_hdf5
from helpers.extraction.master_manifest import MasterManifest


def _write_stage2_shard(tmp_path: Path, *, name: str, records: list[dict[str, object]]) -> Path:
    output_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / f"{name}.h5"
    result = write_slide_patch_dataset_hdf5(output_path=output_path, records=records)
    assert result == output_path
    return output_path


def test_master_manifest_replaces_stage2_slide_rows_with_canonical_source_identity(
    tmp_path: Path,
) -> None:
    records = [
        {
            "filename": "cancer.png",
            "label": 1,
            "patient_id": "1001",
            "slide_id": "slide_a",
            "cov_fold": 0.2,
            "cov_penmarking": 0.0,
            "cov_oof": 0.0,
            "cov_darkspot_foreign": 0.1,
            "cov_edge_airbubble": 0.0,
            "_image_array": np.full((4, 4, 3), 120, dtype=np.uint8),
            "_mask_array": np.ones((4, 4), dtype=np.uint8),
        },
        {
            "filename": "not_cancer.png",
            "label": 0,
            "patient_id": "1001",
            "slide_id": "slide_a",
            "cov_fold": 0.0,
            "cov_penmarking": 0.3,
            "cov_oof": 0.4,
            "cov_darkspot_foreign": 0.0,
            "cov_edge_airbubble": 0.5,
            "_image_array": np.full((4, 4, 3), 10, dtype=np.uint8),
            "_mask_array": np.zeros((4, 4), dtype=np.uint8),
        },
    ]
    shard_path = _write_stage2_shard(tmp_path, name="slide_a", records=records)
    manifest = MasterManifest(tmp_path / "master_manifest.sqlite")

    manifest.replace_stage2_slide_rows(
        source_hdf5_path=shard_path,
        records=records,
        source_slide_path=tmp_path / "source" / "IMAGES" / "slide_a.svs",
        annotation_path=tmp_path / "source" / "ANNOTATIONS" / "slide_a.xml",
        artifacts_geojson_path=tmp_path / "source" / "GEOJSON" / "slide_a.geojson",
        stage2_case_record_id=17,
        stage2_processing_signature="proc-sig",
        stage2_status="COMPLETED",
    )

    with sqlite3.connect(tmp_path / "master_manifest.sqlite") as connection:
        rows = connection.execute(
            """
            SELECT
                p.source_hdf5_path,
                p.source_row_index,
                p.filename,
                p.patient_id,
                p.label,
                p.slide_id,
                p.source_signature,
                p.source_image_path,
                p.source_mask_path,
                p.source_slide_path,
                p.annotation_path,
                p.artifacts_geojson_path,
                p.stage2_case_record_id,
                p.stage2_processing_signature,
                p.stage2_status,
                p.cov_fold,
                p.cov_penmarking,
                p.cov_oof,
                p.cov_darkspot_foreign,
                p.cov_edge_airbubble,
                s.last_updated_stage_name
            FROM patches p
            INNER JOIN patch_stage_state s ON s.patch_id = p.patch_id
            ORDER BY p.source_row_index ASC
            """
        ).fetchall()

    with h5py.File(shard_path, "r") as handle:
        source_signature = str(handle.attrs["source_signature"])

    assert rows == [
        (
            str(shard_path),
            0,
            "cancer.png",
            1001,
            1,
            "slide_a",
            source_signature,
            f"{shard_path}::images[0]",
            f"{shard_path}::masks[0]",
            str(tmp_path / "source" / "IMAGES" / "slide_a.svs"),
            str(tmp_path / "source" / "ANNOTATIONS" / "slide_a.xml"),
            str(tmp_path / "source" / "GEOJSON" / "slide_a.geojson"),
            17,
            "proc-sig",
            "COMPLETED",
            0.2,
            0.0,
            0.0,
            0.1,
            0.0,
            "STAGE2",
        ),
        (
            str(shard_path),
            1,
            "not_cancer.png",
            1001,
            0,
            "slide_a",
            source_signature,
            f"{shard_path}::images[1]",
            f"{shard_path}::masks[1]",
            str(tmp_path / "source" / "IMAGES" / "slide_a.svs"),
            str(tmp_path / "source" / "ANNOTATIONS" / "slide_a.xml"),
            str(tmp_path / "source" / "GEOJSON" / "slide_a.geojson"),
            17,
            "proc-sig",
            "COMPLETED",
            0.0,
            0.3,
            0.4,
            0.0,
            0.5,
            "STAGE2",
        ),
    ]


def test_master_manifest_removes_stale_rows_for_rewritten_stage2_shard(tmp_path: Path) -> None:
    first_records = [
        {
            "filename": "old.png",
            "label": 1,
            "patient_id": "1002",
            "slide_id": "slide_b",
            "_image_array": np.full((4, 4, 3), 55, dtype=np.uint8),
            "_mask_array": np.ones((4, 4), dtype=np.uint8),
        }
    ]
    second_records = [
        {
            "filename": "new.png",
            "label": 0,
            "patient_id": "1002",
            "slide_id": "slide_b",
            "_image_array": np.full((4, 4, 3), 77, dtype=np.uint8),
            "_mask_array": np.zeros((4, 4), dtype=np.uint8),
        }
    ]
    shard_path = _write_stage2_shard(tmp_path, name="slide_b", records=first_records)
    manifest = MasterManifest(tmp_path / "master_manifest.sqlite")

    manifest.replace_stage2_slide_rows(
        source_hdf5_path=shard_path,
        records=first_records,
        source_slide_path=tmp_path / "slide_b.svs",
        annotation_path=tmp_path / "slide_b.xml",
        artifacts_geojson_path=None,
        stage2_case_record_id=19,
        stage2_processing_signature="first",
        stage2_status="COMPLETED",
    )

    shard_path = _write_stage2_shard(tmp_path, name="slide_b", records=second_records)
    manifest.replace_stage2_slide_rows(
        source_hdf5_path=shard_path,
        records=second_records,
        source_slide_path=tmp_path / "slide_b.svs",
        annotation_path=tmp_path / "slide_b.xml",
        artifacts_geojson_path=None,
        stage2_case_record_id=19,
        stage2_processing_signature="second",
        stage2_status="COMPLETED",
    )

    with sqlite3.connect(tmp_path / "master_manifest.sqlite") as connection:
        rows = connection.execute(
            "SELECT filename, source_row_index, stage2_processing_signature FROM patches"
        ).fetchall()

    assert rows == [("new.png", 0, "second")]
