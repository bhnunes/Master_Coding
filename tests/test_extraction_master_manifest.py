import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers.extraction.hdf5_storage import write_slide_patch_dataset_hdf5
from helpers.extraction.manifest_paths import (
    build_hdf5_dataset_ref,
    to_manifest_path_ref,
    to_source_path_ref,
)
from helpers.extraction.master_manifest import MasterManifest, Stage2SlideRows


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
    source_root = tmp_path / "source"
    manifest = MasterManifest(tmp_path / "master_manifest.sqlite", source_root=source_root)

    manifest.replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=records,
            source_slide_path=source_root / "IMAGES" / "slide_a.svs",
            annotation_path=source_root / "ANNOTATIONS" / "slide_a.xml",
            artifacts_geojson_path=source_root / "GEOJSON" / "slide_a.geojson",
            stage2_case_record_id=17,
            stage2_processing_signature="proc-sig",
            stage2_status="COMPLETED",
        )
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
            to_manifest_path_ref(shard_path, manifest_path=tmp_path / "master_manifest.sqlite"),
            0,
            "cancer.png",
            1001,
            1,
            "slide_a",
            source_signature,
            build_hdf5_dataset_ref("images", 0),
            build_hdf5_dataset_ref("masks", 0),
            to_source_path_ref(source_root / "IMAGES" / "slide_a.svs", source_root=source_root),
            to_source_path_ref(
                source_root / "ANNOTATIONS" / "slide_a.xml",
                source_root=source_root,
            ),
            to_source_path_ref(
                source_root / "GEOJSON" / "slide_a.geojson",
                source_root=source_root,
            ),
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
            to_manifest_path_ref(shard_path, manifest_path=tmp_path / "master_manifest.sqlite"),
            1,
            "not_cancer.png",
            1001,
            0,
            "slide_a",
            source_signature,
            build_hdf5_dataset_ref("images", 1),
            build_hdf5_dataset_ref("masks", 1),
            to_source_path_ref(source_root / "IMAGES" / "slide_a.svs", source_root=source_root),
            to_source_path_ref(
                source_root / "ANNOTATIONS" / "slide_a.xml",
                source_root=source_root,
            ),
            to_source_path_ref(
                source_root / "GEOJSON" / "slide_a.geojson",
                source_root=source_root,
            ),
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
    source_root = tmp_path / "source"
    manifest = MasterManifest(tmp_path / "master_manifest.sqlite", source_root=source_root)

    manifest.replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=first_records,
            source_slide_path=source_root / "IMAGES" / "slide_b.svs",
            annotation_path=source_root / "ANNOTATIONS" / "slide_b.xml",
            artifacts_geojson_path=None,
            stage2_case_record_id=19,
            stage2_processing_signature="first",
            stage2_status="COMPLETED",
        )
    )

    shard_path = _write_stage2_shard(tmp_path, name="slide_b", records=second_records)
    manifest.replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=second_records,
            source_slide_path=source_root / "IMAGES" / "slide_b.svs",
            annotation_path=source_root / "ANNOTATIONS" / "slide_b.xml",
            artifacts_geojson_path=None,
            stage2_case_record_id=19,
            stage2_processing_signature="second",
            stage2_status="COMPLETED",
        )
    )

    with sqlite3.connect(tmp_path / "master_manifest.sqlite") as connection:
        rows = connection.execute(
            "SELECT filename, source_row_index, stage2_processing_signature FROM patches"
        ).fetchall()

    assert rows == [("new.png", 0, "second")]


def test_master_manifest_batches_stage3_3_stage4_and_stage6_updates(tmp_path: Path) -> None:
    records = [
        {
            "filename": "a.png",
            "label": 1,
            "patient_id": "2001",
            "slide_id": "slide_c",
            "_image_array": np.full((4, 4, 3), 20, dtype=np.uint8),
            "_mask_array": np.ones((4, 4), dtype=np.uint8),
        },
        {
            "filename": "b.png",
            "label": 0,
            "patient_id": "2002",
            "slide_id": "slide_d",
            "_image_array": np.full((4, 4, 3), 40, dtype=np.uint8),
            "_mask_array": np.zeros((4, 4), dtype=np.uint8),
        },
    ]
    shard_path = _write_stage2_shard(tmp_path, name="batched_updates", records=records)
    source_root = tmp_path / "source"
    manifest = MasterManifest(tmp_path / "master_manifest.sqlite", source_root=source_root)

    manifest.replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=records,
            source_slide_path=source_root / "IMAGES" / "batched_updates.svs",
            annotation_path=None,
            artifacts_geojson_path=None,
            stage2_case_record_id=23,
            stage2_processing_signature="stage2-batch",
            stage2_status="COMPLETED",
        )
    )

    with h5py.File(shard_path, "r") as handle:
        source_signature = str(handle.attrs["source_signature"])

    manifest.update_stage3_3_cleaning_decisions(
        decisions=[
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 0,
                "filename": "a.png",
                "patient_id": 2001,
                "slide_id": "slide_c",
                "source_hdf5_sha256": source_signature,
                "decision": "accepted",
                "contamination_rate": 0.1,
            },
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 1,
                "filename": "b.png",
                "patient_id": 2002,
                "slide_id": "slide_d",
                "source_hdf5_sha256": source_signature,
                "decision": "rejected",
                "contamination_rate": 0.8,
            },
        ]
    )
    manifest.update_stage4_split_assignments(
        assignments=[
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 0,
                "filename": "a.png",
                "patient_id": 2001,
                "label": 1,
                "split": "TRAIN",
            },
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 1,
                "filename": "b.png",
                "patient_id": 2002,
                "label": 0,
                "split": "VALIDATION",
            },
        ],
        normalization_method="NOT_NORMALIZED",
        normalization_artifact_id=None,
    )
    manifest.update_stage6_sampling_decisions(
        decisions=[
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 0,
                "filename": "a.png",
                "patient_id": 2001,
                "label": 1,
                "sampling_decision": "protected_kept",
                "is_stage7_selected": True,
            },
            {
                "source_hdf5_path": str(shard_path),
                "source_row_index": 1,
                "filename": "b.png",
                "patient_id": 2002,
                "label": 0,
                "sampling_decision": "rejected_reducible",
                "is_stage7_selected": False,
            },
        ]
    )

    with sqlite3.connect(tmp_path / "master_manifest.sqlite") as connection:
        rows = connection.execute(
            "SELECT cleaning_decision, contamination_rate, split, normalization_method, "
            "sampling_decision, is_stage4_accepted, is_stage7_selected, last_updated_stage_name "
            "FROM patch_stage_state ORDER BY patch_id ASC"
        ).fetchall()

    assert rows == [
        ("accepted", 0.1, "TRAIN", "NOT_NORMALIZED", "protected_kept", 1, 1, "STAGE6"),
        (
            "rejected",
            0.8,
            "VALIDATION",
            "NOT_NORMALIZED",
            "rejected_reducible",
            0,
            0,
            "STAGE6",
        ),
    ]


def test_list_stage2_patch_records_rejects_legacy_absolute_manifest_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "legacy.h5"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.attrs["source_signature"] = "legacy-sig"

    with sqlite3.connect(manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                slide_id TEXT,
                source_signature TEXT,
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL,
                source_slide_path TEXT NOT NULL,
                annotation_path TEXT,
                artifacts_geojson_path TEXT,
                stage2_case_record_id INTEGER NOT NULL,
                stage2_processing_signature TEXT,
                stage2_status TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_hdf5_path, source_row_index)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO patches (
                source_hdf5_path,
                source_row_index,
                filename,
                patient_id,
                label,
                slide_id,
                source_signature,
                source_image_path,
                source_mask_path,
                source_slide_path,
                stage2_case_record_id,
                stage2_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(shard_path),
                0,
                "legacy.png",
                1,
                1,
                "slide_1",
                "legacy-sig",
                "HDF5::images[0]",
                "HDF5::masks[0]",
                "SOURCE::IMAGES/legacy.svs",
                1,
                "COMPLETED",
            ),
        )
        connection.commit()

    with pytest.raises(ValueError, match="Legacy absolute-path manifest values are not supported"):
        MasterManifest(manifest_path).list_stage2_patch_records()
