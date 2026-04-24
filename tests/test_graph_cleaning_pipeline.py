from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from helpers.extraction.master_manifest import MasterManifest, Stage2SlideRows
from helpers.graph import cleaning_pipeline
from helpers.graph.cleaning_pipeline import (
    GraphCleaningPipelineConfig,
    SourceCandidateRecord,
    run_graph_cleaning_pipeline,
)
from helpers.graph.contamination import GraphContaminationParameters, calculate_roi_contamination

HDF5_BATCH_COUNT = 2
CANCER_ONLY_EVALUATED_COUNT = 2


def _pipeline_config(
    *,
    master_manifest_path: Path,
    output_base_dir: Path,
    graph_params: GraphContaminationParameters,
    logger_name: str,
    scorer: Any = calculate_roi_contamination,
) -> GraphCleaningPipelineConfig:
    return GraphCleaningPipelineConfig(
        master_manifest_path=master_manifest_path,
        output_base_dir=output_base_dir,
        graph_params=graph_params,
        tau=0.24,
        num_workers=1,
        logger=logging.getLogger(logger_name),
        scorer=scorer,
    )


def _write_graph_cleaning_shard(
    shard_path: Path,
    filenames: list[str],
    patient_ids: list[int],
    *,
    slide_ids: list[str] | None = None,
    source_signature: str | None = None,
) -> None:
    with h5py.File(shard_path, "w") as handle:
        row_count = len(filenames)
        handle.create_dataset("images", data=np.zeros((row_count, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((row_count, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.ones((row_count,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array(patient_ids, dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([f"{name}.png".encode() for name in filenames]),
        )
        if slide_ids is not None:
            handle.create_dataset(
                "slide_ids",
                data=np.array([slide.encode() for slide in slide_ids]),
            )
        if source_signature is not None:
            handle.attrs["source_signature"] = source_signature


def _write_manifest_for_shard(
    *,
    master_manifest_path: Path,
    shard_path: Path,
    filenames: list[str],
    patient_ids: list[int],
    labels: list[int] | None = None,
    slide_ids: list[str] | None = None,
) -> None:
    resolved_labels = labels if labels is not None else ([1] * len(filenames))
    MasterManifest(master_manifest_path).replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=[
                {
                    "filename": f"{filename}.png",
                    "patient_id": patient_id,
                    "label": label,
                    "slide_id": slide_id,
                }
                for filename, patient_id, label, slide_id in zip(
                    filenames,
                    patient_ids,
                    resolved_labels,
                    slide_ids if slide_ids is not None else [shard_path.stem] * len(filenames),
                    strict=True,
                )
            ],
            source_slide_path=shard_path.with_suffix(".svs"),
            annotation_path=None,
            artifacts_geojson_path=None,
            stage2_case_record_id=1,
            stage2_processing_signature="stage2-test",
            stage2_status="completed",
        )
    )


def _read_shard_snapshot(shard_path: Path) -> dict[str, object]:
    with h5py.File(shard_path, "r") as handle:
        return {
            "images_shape": tuple(handle["images"].shape),
            "masks_shape": tuple(handle["masks"].shape),
            "labels": handle["labels"][:].tolist(),
            "patient_ids": handle["patient_ids"][:].tolist(),
            "filenames": [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["filenames"][:]
            ],
            "source_signature": handle.attrs.get("source_signature"),
        }


def test_run_graph_cleaning_pipeline_reports_empty_source_dataset(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    output_dir = tmp_path / "output"
    MasterManifest(master_manifest_path).initialize()

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(
                bg_intensity_thresh=198,
                k=386.0,
                min_size=200,
                erosion_px=0,
            ),
            logger_name="test_graph_cleaning_empty",
        )
    )

    assert summary.total_images == 0
    assert summary.accepted == 0
    assert summary.rejected == 0
    assert summary.skipped == 0


def test_run_graph_cleaning_pipeline_writes_decision_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    names = ["accepted_PATIENT_1", "rejected_PATIENT_2"]
    _write_graph_cleaning_shard(
        shard_path,
        names,
        [1, 2],
        slide_ids=["slide_a", "slide_b"],
        source_signature="stage3-signature",
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=names,
        patient_ids=[1, 2],
        slide_ids=["slide_a", "slide_b"],
    )

    scores = {0: 0.10, 1: 0.50}

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        row_index = int(str(image_path).rsplit("[", maxsplit=1)[1][:-1])
        return scores[row_index]

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(
                bg_intensity_thresh=198,
                k=386.0,
                min_size=200,
                erosion_px=0,
            ),
            logger_name="test_graph_cleaning_hdf5",
            scorer=fake_scorer,
        )
    )

    assert summary.accepted == 1
    assert summary.rejected == 1
    assert summary.accepted_manifest_path == output_dir / "accepted_manifest.csv"
    assert summary.rejected_manifest_path == output_dir / "rejected_manifest.csv"

    with (output_dir / "accepted_manifest.csv").open(encoding="utf-8", newline="") as handle:
        accepted_rows = list(csv.DictReader(handle))
    with (output_dir / "rejected_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))

    assert accepted_rows[0]["filename"] == "accepted_PATIENT_1.png"
    assert accepted_rows[0]["source_hdf5_sha256"] != ""
    assert accepted_rows[0]["source_row_index"] == "0"
    assert rejected_rows[0]["filename"] == "rejected_PATIENT_2.png"
    assert rejected_rows[0]["source_hdf5_sha256"] != ""
    assert rejected_rows[0]["source_row_index"] == "1"


def test_run_graph_cleaning_pipeline_does_not_rewrite_source_hdf5_shards(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    _write_graph_cleaning_shard(
        shard_path,
        ["accepted_PATIENT_1", "rejected_PATIENT_2"],
        [1, 2],
        slide_ids=["slide_a", "slide_b"],
        source_signature="stage3-signature",
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=["accepted_PATIENT_1", "rejected_PATIENT_2"],
        patient_ids=[1, 2],
        slide_ids=["slide_a", "slide_b"],
    )
    before_snapshot = _read_shard_snapshot(shard_path)

    scores = {0: 0.10, 1: 0.50}

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        row_index = int(str(image_path).rsplit("[", maxsplit=1)[1][:-1])
        return scores[row_index]

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
            logger_name="test_graph_cleaning_read_only_shards",
            scorer=fake_scorer,
        )
    )

    after_snapshot = _read_shard_snapshot(shard_path)

    assert summary.total_images == CANCER_ONLY_EVALUATED_COUNT
    assert before_snapshot == after_snapshot


def test_run_graph_cleaning_pipeline_uses_only_cancer_manifest_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    names = ["cancer_a", "cancer_b", "not_cancer_c"]
    _write_graph_cleaning_shard(
        shard_path,
        names,
        [1, 2, 3],
        slide_ids=["slide_a", "slide_b", "slide_c"],
        source_signature="stage3-signature",
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=names,
        patient_ids=[1, 2, 3],
        labels=[1, 1, 0],
        slide_ids=["slide_a", "slide_b", "slide_c"],
    )

    scores = {0: 0.10, 1: 0.50}

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        row_index = int(str(image_path).rsplit("[", maxsplit=1)[1][:-1])
        return scores[row_index]

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
            logger_name="test_graph_cleaning_cancer_only",
            scorer=fake_scorer,
        )
    )

    assert summary.total_images == CANCER_ONLY_EVALUATED_COUNT
    assert summary.accepted == 1
    assert summary.rejected == 1
    with (output_dir / "accepted_manifest.csv").open(encoding="utf-8", newline="") as handle:
        accepted_rows = list(csv.DictReader(handle))
    with (output_dir / "rejected_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))

    assert [row["filename"] for row in accepted_rows] == ["cancer_a.png"]
    assert [row["filename"] for row in rejected_rows] == ["cancer_b.png"]
    assert all(row["filename"] != "not_cancer_c.png" for row in accepted_rows + rejected_rows)


def test_list_manifest_candidates_uses_source_signature_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "sample.h5"
    _write_graph_cleaning_shard(
        shard_path,
        ["sample"],
        [7],
        source_signature="known-signature",
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=["sample"],
        patient_ids=[7],
    )

    def fail_hash(path: Path) -> str:
        raise AssertionError(f"hash_file_sha256 should not be called for {path}")

    monkeypatch.setattr(cleaning_pipeline, "hash_file_sha256", fail_hash)

    candidates = cleaning_pipeline._list_manifest_candidates(
        master_manifest_path, logging.getLogger("test_graph_cleaning_source_signature")
    )

    assert len(candidates) == 1
    assert candidates[0].source_hdf5_sha256 == "known-signature"


def test_list_manifest_candidates_excludes_non_cancer_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "sample.h5"
    _write_graph_cleaning_shard(
        shard_path,
        ["cancer_sample", "not_cancer_sample"],
        [7, 8],
        source_signature="known-signature",
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=["cancer_sample", "not_cancer_sample"],
        patient_ids=[7, 8],
        labels=[1, 0],
    )

    candidates = cleaning_pipeline._list_manifest_candidates(
        master_manifest_path, logging.getLogger("test_graph_cleaning_cancer_candidates")
    )

    assert [candidate.filename for candidate in candidates] == ["cancer_sample.png"]


def test_process_hdf5_candidates_batches_contiguous_hdf5_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cleaning_pipeline, "_HDF5_SCORING_BATCH_SIZE", 64)
    batch_starts: list[tuple[str, int, int]] = []

    def fake_load_graph_source_batch(
        source_hdf5_path: Path | str,
        start_index: int,
        end_index: int,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        batch_starts.append((str(source_hdf5_path), start_index, end_index))
        batch_size = end_index - start_index
        images = np.zeros((batch_size, 2, 2, 3), dtype=np.uint8)
        for offset in range(batch_size):
            images[offset, 0, 0, 0] = start_index + offset
        masks = cast(
            npt.NDArray[np.uint8],
            np.asarray(np.ones((batch_size, 2, 2), dtype=np.uint8) * 255, dtype=np.uint8),
        )
        return images, masks

    def fake_calculate_roi_contamination_from_arrays(
        image: npt.NDArray[np.uint8],
        roi_mask: npt.NDArray[np.uint8],
        params: GraphContaminationParameters,
        *,
        logger: logging.Logger | None = None,
        base_name: str = "preloaded_record",
    ) -> float:
        del roi_mask, params, logger, base_name
        return float(image[0, 0, 0]) / 100.0

    monkeypatch.setattr(cleaning_pipeline, "load_graph_source_batch", fake_load_graph_source_batch)
    monkeypatch.setattr(
        cleaning_pipeline,
        "calculate_roi_contamination_from_arrays",
        fake_calculate_roi_contamination_from_arrays,
    )

    candidates = [
        SourceCandidateRecord(
            filename=f"sample_{index}.png",
            image_path=f"fake.h5::images[{index}]",
            mask_path=f"fake.h5::masks[{index}]",
            source_hdf5_path="fake.h5",
            source_hdf5_sha256="sig",
            source_row_index=index,
        )
        for index in range(70)
    ]

    decisions = cleaning_pipeline._process_hdf5_candidates(
        candidates=candidates,
        graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
        tau=0.5,
        scorer=calculate_roi_contamination,
        progress_factory=None,
    )

    assert len(batch_starts) == HDF5_BATCH_COUNT
    assert batch_starts == [("fake.h5", 0, 64), ("fake.h5", 64, 70)]
    assert decisions[0].decision == "accepted"
    assert decisions[-1].decision == "rejected"
    assert decisions[-1].contamination_rate == pytest.approx(0.69)


def test_run_graph_cleaning_pipeline_updates_master_manifest_state(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_cleaning_shard(shard_path, ["a", "b"], [1, 2], source_signature="stage2-sig")

    with sqlite3.connect(master_manifest_path) as connection:
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
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        for row_index, filename in enumerate(("a.png", "b.png")):
            cursor = connection.execute(
                """
                INSERT INTO patches (
                    source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                    source_signature, source_image_path, source_mask_path, source_slide_path,
                    stage2_case_record_id, stage2_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(shard_path),
                    row_index,
                    filename,
                    row_index + 1,
                    1,
                    f"slide_{row_index}",
                    "stage2-sig",
                    f"{shard_path}::images[{row_index}]",
                    f"{shard_path}::masks[{row_index}]",
                    str(shard_path),
                    1,
                    "COMPLETED",
                ),
            )
            connection.execute(
                "INSERT INTO patch_stage_state (patch_id, last_updated_stage_name) VALUES (?, ?)",
                ((cursor.lastrowid or 0), "STAGE2"),
            )
        connection.commit()

    def fake_sqlite_scorer(
        image_path: Path | str,
        mask_path: Path | str,
        params: GraphContaminationParameters,
    ) -> float:
        del mask_path, params
        return 0.1 if str(image_path).endswith("[0]") else 0.5

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
            logger_name="test_graph_cleaning_sqlite",
            scorer=fake_sqlite_scorer,
        )
    )

    assert summary.accepted == 1
    assert summary.rejected == 1
    with sqlite3.connect(master_manifest_path) as connection:
        rows = connection.execute(
            "SELECT cleaning_decision, contamination_rate, is_stage4_accepted, "
            "last_updated_stage_name "
            "FROM patch_stage_state ORDER BY patch_id ASC"
        ).fetchall()

    assert rows == [
        ("accepted", 0.1, 1, "STAGE3_3"),
        ("rejected", 0.5, 0, "STAGE3_3"),
    ]


def test_run_graph_cleaning_pipeline_updates_only_canonical_cancer_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_cleaning_shard(
        shard_path,
        ["cancer_accept", "cancer_reject", "not_cancer_ignore"],
        [1, 2, 3],
        source_signature="stage2-sig",
    )

    with sqlite3.connect(master_manifest_path) as connection:
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
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        patch_rows = [
            (0, "cancer_accept.png", 1, 1, "slide_0"),
            (1, "cancer_reject.png", 2, 1, "slide_1"),
            (2, "not_cancer_ignore.png", 3, 0, "slide_2"),
        ]
        for source_row_index, filename, patient_id, label, slide_id in patch_rows:
            cursor = connection.execute(
                """
                INSERT INTO patches (
                    source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                    source_signature, source_image_path, source_mask_path, source_slide_path,
                    stage2_case_record_id, stage2_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(shard_path),
                    source_row_index,
                    filename,
                    patient_id,
                    label,
                    slide_id,
                    "stage2-sig",
                    f"{shard_path}::images[{source_row_index}]",
                    f"{shard_path}::masks[{source_row_index}]",
                    str(shard_path),
                    1,
                    "COMPLETED",
                ),
            )
            connection.execute(
                "INSERT INTO patch_stage_state (patch_id, last_updated_stage_name) VALUES (?, ?)",
                ((cursor.lastrowid or 0), "STAGE2"),
            )
        connection.commit()

    def fake_sqlite_scorer(
        image_path: Path | str,
        mask_path: Path | str,
        params: GraphContaminationParameters,
    ) -> float:
        del mask_path, params
        row_index = int(str(image_path).rsplit("[", maxsplit=1)[1][:-1])
        return 0.1 if row_index == 0 else 0.5

    summary = run_graph_cleaning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            output_base_dir=output_dir,
            graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
            logger_name="test_graph_cleaning_canonical_cancer_rows",
            scorer=fake_sqlite_scorer,
        )
    )

    assert summary.total_images == CANCER_ONLY_EVALUATED_COUNT
    with sqlite3.connect(master_manifest_path) as connection:
        rows = connection.execute(
            "SELECT p.source_row_index, p.filename, s.cleaning_decision, s.contamination_rate, "
            "s.is_stage4_accepted, s.last_updated_stage_name "
            "FROM patches AS p "
            "JOIN patch_stage_state AS s ON s.patch_id = p.patch_id "
            "ORDER BY p.source_row_index ASC"
        ).fetchall()

    assert rows == [
        (0, "cancer_accept.png", "accepted", 0.1, 1, "STAGE3_3"),
        (1, "cancer_reject.png", "rejected", 0.5, 0, "STAGE3_3"),
        (2, "not_cancer_ignore.png", None, None, None, "STAGE2"),
    ]


def test_run_graph_cleaning_pipeline_rejects_stale_master_manifest_provenance(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_cleaning_shard(shard_path, ["a"], [1], source_signature="current-signature")

    with sqlite3.connect(master_manifest_path) as connection:
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
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor = connection.execute(
            """
            INSERT INTO patches (
                source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                source_signature, source_image_path, source_mask_path, source_slide_path,
                stage2_case_record_id, stage2_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(shard_path),
                0,
                "a.png",
                1,
                1,
                "slide_0",
                "stale-signature",
                f"{shard_path}::images[0]",
                f"{shard_path}::masks[0]",
                str(shard_path),
                1,
                "COMPLETED",
            ),
        )
        connection.execute(
            "INSERT INTO patch_stage_state (patch_id, last_updated_stage_name) VALUES (?, ?)",
            ((cursor.lastrowid or 0), "STAGE2"),
        )
        connection.commit()

    def fake_sqlite_scorer(
        image_path: Path | str,
        mask_path: Path | str,
        params: GraphContaminationParameters,
    ) -> float:
        del image_path, mask_path, params
        return 0.1

    def fake_candidate_list(
        master_manifest_path: Path,
        logger: logging.Logger,
    ) -> list[SourceCandidateRecord]:
        del master_manifest_path, logger
        return [
            SourceCandidateRecord(
                filename="a.png",
                image_path=f"{shard_path}::images[0]",
                mask_path=f"{shard_path}::masks[0]",
                patient_id="1",
                slide_id="slide_0",
                source_hdf5_path=str(shard_path),
                source_hdf5_sha256="current-signature",
                source_row_index=0,
            )
        ]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cleaning_pipeline, "_list_manifest_candidates", fake_candidate_list)
    try:
        with pytest.raises(ValueError, match="source_signature does not match"):
            run_graph_cleaning_pipeline(
                _pipeline_config(
                    master_manifest_path=master_manifest_path,
                    output_base_dir=output_dir,
                    graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
                    logger_name="test_graph_cleaning_sqlite_stale_provenance",
                    scorer=fake_sqlite_scorer,
                )
            )
    finally:
        monkeypatch.undo()


def test_run_graph_cleaning_pipeline_rejects_mismatched_filename_join(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    shard_path = source_path / "batch_a.h5"
    output_dir = tmp_path / "output"
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_cleaning_shard(shard_path, ["actual"], [1], source_signature="stage2-sig")

    with sqlite3.connect(master_manifest_path) as connection:
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
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor = connection.execute(
            """
            INSERT INTO patches (
                source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                source_signature, source_image_path, source_mask_path, source_slide_path,
                stage2_case_record_id, stage2_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(shard_path),
                0,
                "expected.png",
                1,
                1,
                "slide_0",
                "stage2-sig",
                f"{shard_path}::images[0]",
                f"{shard_path}::masks[0]",
                str(shard_path),
                1,
                "COMPLETED",
            ),
        )
        connection.execute(
            "INSERT INTO patch_stage_state (patch_id, last_updated_stage_name) VALUES (?, ?)",
            ((cursor.lastrowid or 0), "STAGE2"),
        )
        connection.commit()

    def fake_candidate_list(
        master_manifest_path: Path,
        logger: logging.Logger,
    ) -> list[SourceCandidateRecord]:
        del master_manifest_path, logger
        return [
            SourceCandidateRecord(
                filename="actual.png",
                image_path=f"{shard_path}::images[0]",
                mask_path=f"{shard_path}::masks[0]",
                patient_id="1",
                slide_id="slide_0",
                source_hdf5_path=str(shard_path),
                source_hdf5_sha256="stage2-sig",
                source_row_index=0,
            )
        ]

    def fake_sqlite_scorer(
        image_path: Path | str,
        mask_path: Path | str,
        params: GraphContaminationParameters,
    ) -> float:
        del image_path, mask_path, params
        return 0.1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cleaning_pipeline, "_list_manifest_candidates", fake_candidate_list)
    try:
        with pytest.raises(ValueError, match="filename does not match"):
            run_graph_cleaning_pipeline(
                _pipeline_config(
                    master_manifest_path=master_manifest_path,
                    output_base_dir=output_dir,
                    graph_params=GraphContaminationParameters(198, 386.0, 200, 0),
                    logger_name="test_graph_cleaning_sqlite_filename_mismatch",
                    scorer=fake_sqlite_scorer,
                )
            )
    finally:
        monkeypatch.undo()
