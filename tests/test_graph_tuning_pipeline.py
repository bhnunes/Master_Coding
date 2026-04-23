from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from helpers.extraction.master_manifest import MasterManifest, Stage2SlideRows
from helpers.graph.contamination import GraphContaminationParameters
from helpers.graph.parameter_store import GraphCleaningParameterArtifact
from helpers.graph.tuning_pipeline import (
    GraphTuningPipelineConfig,
    GraphTuningResult,
    GraphTuningSummary,
    LabeledSourceRecord,
    _build_grouped_cv_splits,
    _build_objective,
    _split_records,
    build_graph_cleaning_parameter_artifact,
    build_recommendation_message,
    collect_review_labels,
    infer_group_id_from_stem,
    resolve_labeled_source_records,
    run_graph_tuning_pipeline,
    select_best_contamination_threshold,
)
from helpers.optimization_sampling.sampling import ImageMaskPair

RESOLVED_RECORD_COUNT = 1
BEST_THRESHOLD = 0.5
BEST_GRAPH_K = 386.0


def _pipeline_config(
    *,
    master_manifest_path: Path,
    review_base_dir: Path,
    logger_name: str,
    scorer: Any,
    optimizer: Any,
) -> GraphTuningPipelineConfig:
    return GraphTuningPipelineConfig(
        master_manifest_path=master_manifest_path,
        review_base_dir=review_base_dir,
        test_set_size=0.5,
        n_splits_inner_cv=2,
        n_bayesian_calls=10,
        n_initial_points=4,
        random_state=42,
        bg_intensity_range=(100, 250),
        k_range=(100, 500),
        min_size_range=(10, 200),
        erosion_range=(0, 10),
        logger=logging.getLogger(logger_name),
        scorer=scorer,
        optimizer=optimizer,
        progress_factory=lambda iterable, **_: iterable,
    )


def _write_graph_tuning_shard(
    shard_path: Path,
    filenames: list[str],
    patient_ids: list[int],
    labels: list[int] | None = None,
) -> None:
    with h5py.File(shard_path, "w") as handle:
        row_count = len(filenames)
        handle.create_dataset("images", data=np.zeros((row_count, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((row_count, 4, 4), dtype=np.uint8))
        handle.create_dataset(
            "labels",
            data=np.array(labels if labels is not None else ([1] * row_count), dtype=np.uint8),
        )
        handle.create_dataset("patient_ids", data=np.array(patient_ids, dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([f"{name}.png".encode() for name in filenames]),
        )
        handle.attrs["source_signature"] = f"signature-{shard_path.stem}"


def _write_manifest_for_shard(
    *,
    master_manifest_path: Path,
    shard_path: Path,
    filenames: list[str],
    patient_ids: list[int],
    labels: list[int] | None = None,
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
                }
                for filename, patient_id, label in zip(
                    filenames,
                    patient_ids,
                    resolved_labels,
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


def _resolve_shard_name(ref: Path | str, names_by_shard: dict[str, list[str]]) -> str:
    ref_text = str(ref)
    row_index = int(ref_text.rsplit("[", maxsplit=1)[1][:-1])
    shard_name = Path(ref_text.split("::", maxsplit=1)[0]).name
    return names_by_shard[shard_name][row_index]


def test_collect_review_labels_reads_approved_and_rejected_files(tmp_path: Path) -> None:
    approved_dir = tmp_path / "APPROVED"
    rejected_dir = tmp_path / "REJECTED"
    approved_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)
    (approved_dir / "case_a.png").write_bytes(b"x")
    (rejected_dir / "case_b.png").write_bytes(b"x")

    labels = collect_review_labels(tmp_path)

    assert labels == {"case_a": "Approved", "case_b": "Rejected"}


def test_collect_review_labels_rejects_empty_review_folders(tmp_path: Path) -> None:
    (tmp_path / "APPROVED").mkdir(parents=True)
    (tmp_path / "REJECTED").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Nothing to tune"):
        collect_review_labels(tmp_path)


def test_resolve_labeled_source_records_keeps_only_pairs_present_in_source(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_tuning_shard(source_path / "patient_1.h5", ["case_a"], [1])
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "patient_1.h5",
        filenames=["case_a"],
        patient_ids=[1],
    )

    logger = logging.getLogger("test_graph_tuning")
    records = resolve_labeled_source_records(
        labels_by_stem={"case_a": "Approved", "case_b": "Rejected"},
        master_manifest_path=master_manifest_path,
        logger=logger,
    )

    assert [(record.pair.stem, record.label, record.group_id) for record in records] == [
        ("case_a", "Approved", "case_a")
    ]


def test_resolve_labeled_source_records_supports_master_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_tuning_shard(source_path / "patient_7.h5", ["CANCER_PATIENT_7_PATCH_001"], [7])
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "patient_7.h5",
        filenames=["CANCER_PATIENT_7_PATCH_001"],
        patient_ids=[7],
    )

    records = resolve_labeled_source_records(
        labels_by_stem={"CANCER_PATIENT_7_PATCH_001": "Approved"},
        master_manifest_path=master_manifest_path,
        logger=logging.getLogger("test_graph_tuning"),
    )

    assert len(records) == RESOLVED_RECORD_COUNT
    assert str(records[0].pair.image_path).endswith("::images[0]")


def test_resolve_labeled_source_records_skips_reviewed_non_cancer_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_graph_tuning_shard(
        source_path / "patient_7.h5",
        ["CANCER_PATIENT_7_PATCH_001", "NOT_CANCER_PATIENT_7_PATCH_002"],
        [7, 7],
        labels=[1, 0],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "patient_7.h5",
        filenames=["CANCER_PATIENT_7_PATCH_001", "NOT_CANCER_PATIENT_7_PATCH_002"],
        patient_ids=[7, 7],
        labels=[1, 0],
    )

    records = resolve_labeled_source_records(
        labels_by_stem={
            "CANCER_PATIENT_7_PATCH_001": "Approved",
            "NOT_CANCER_PATIENT_7_PATCH_002": "Rejected",
        },
        master_manifest_path=master_manifest_path,
        logger=logging.getLogger("test_graph_tuning"),
    )

    assert [record.pair.stem for record in records] == ["CANCER_PATIENT_7_PATCH_001"]


def test_infer_group_id_from_stem_extracts_patient_from_legacy_and_new_patterns() -> None:
    assert infer_group_id_from_stem("CANCER_PATIENT_42_128_256_1234_20260101") == "42"
    assert infer_group_id_from_stem("CANCER_PATIENT_42_SLIDE_slide-7_X_128_Y_256") == "42"
    assert infer_group_id_from_stem("unstructured_stem") == "unstructured_stem"


def test_select_best_contamination_threshold_optimizes_rejected_f1() -> None:
    rates = [0.1, 0.2, 0.8, 0.9]
    labels = ["Approved", "Approved", "Rejected", "Rejected"]

    tau = select_best_contamination_threshold(rates, labels, thresholds=np.array([0.15, 0.5, 0.85]))

    assert tau == pytest.approx(BEST_THRESHOLD)


def test_run_graph_tuning_pipeline_uses_optimizer_result_and_returns_summary(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    review_dir = tmp_path / "review"
    approved_dir = review_dir / "APPROVED"
    rejected_dir = review_dir / "REJECTED"
    approved_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)

    source_names = (
        "a1_PATIENT_1",
        "a2_PATIENT_2",
        "a3_PATIENT_5",
        "a4_PATIENT_6",
        "r1_PATIENT_3",
        "r2_PATIENT_4",
        "r3_PATIENT_7",
        "r4_PATIENT_8",
    )
    _write_graph_tuning_shard(
        source_path / "batch_a.h5",
        list(source_names[:4]),
        [1, 2, 5, 6],
    )
    _write_graph_tuning_shard(
        source_path / "batch_b.h5",
        list(source_names[4:]),
        [3, 4, 7, 8],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "batch_a.h5",
        filenames=list(source_names[:4]),
        patient_ids=[1, 2, 5, 6],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "batch_b.h5",
        filenames=list(source_names[4:]),
        patient_ids=[3, 4, 7, 8],
    )
    for name in ("a1_PATIENT_1", "a2_PATIENT_2", "a3_PATIENT_5", "a4_PATIENT_6"):
        (approved_dir / f"{name}.png").write_bytes(b"overlay")
    for name in ("r1_PATIENT_3", "r2_PATIENT_4", "r3_PATIENT_7", "r4_PATIENT_8"):
        (rejected_dir / f"{name}.png").write_bytes(b"overlay")

    score_map = {
        "a1_PATIENT_1": 0.10,
        "a2_PATIENT_2": 0.20,
        "a3_PATIENT_5": 0.12,
        "a4_PATIENT_6": 0.18,
        "r1_PATIENT_3": 0.80,
        "r2_PATIENT_4": 0.90,
        "r3_PATIENT_7": 0.82,
        "r4_PATIENT_8": 0.88,
    }
    names_by_shard = {
        "batch_a.h5": list(source_names[:4]),
        "batch_b.h5": list(source_names[4:]),
    }

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        return score_map[_resolve_shard_name(image_path, names_by_shard)]

    def fake_optimizer(*, objective: object, search_space: list[object]) -> GraphTuningResult:
        del objective, search_space
        return GraphTuningResult(
            best_score=1.0,
            best_params=GraphContaminationParameters(
                bg_intensity_thresh=198,
                k=386.0,
                min_size=200,
                erosion_px=0,
            ),
        )

    summary = run_graph_tuning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            review_base_dir=review_dir,
            logger_name="test_graph_pipeline",
            scorer=fake_scorer,
            optimizer=fake_optimizer,
        )
    )

    assert isinstance(summary, GraphTuningSummary)
    assert summary.best_params.k == BEST_GRAPH_K
    assert summary.final_tau == pytest.approx(0.21, abs=0.05)
    assert summary.best_cross_validated_f1 == 1.0


def test_run_graph_tuning_pipeline_accepts_master_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    review_dir = tmp_path / "review"
    approved_dir = review_dir / "APPROVED"
    rejected_dir = review_dir / "REJECTED"
    approved_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)

    names = [
        "a1_PATIENT_1",
        "a2_PATIENT_2",
        "a3_PATIENT_5",
        "a4_PATIENT_6",
        "r1_PATIENT_3",
        "r2_PATIENT_4",
        "r3_PATIENT_7",
        "r4_PATIENT_8",
    ]
    _write_graph_tuning_shard(source_path / "batch_a.h5", names[:4], [1, 2, 5, 6])
    _write_graph_tuning_shard(source_path / "batch_b.h5", names[4:], [3, 4, 7, 8])
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "batch_a.h5",
        filenames=names[:4],
        patient_ids=[1, 2, 5, 6],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "batch_b.h5",
        filenames=names[4:],
        patient_ids=[3, 4, 7, 8],
    )

    for name in names[:4]:
        (approved_dir / f"{name}.png").write_bytes(b"overlay")
    for name in names[4:]:
        (rejected_dir / f"{name}.png").write_bytes(b"overlay")

    score_map = {
        name: value
        for name, value in zip(names, [0.1, 0.2, 0.12, 0.18, 0.8, 0.9, 0.82, 0.88], strict=True)
    }
    names_by_shard = {
        "batch_a.h5": names[:4],
        "batch_b.h5": names[4:],
    }

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        return score_map[_resolve_shard_name(image_path, names_by_shard)]

    def fake_optimizer(*, objective: object, search_space: list[object]) -> GraphTuningResult:
        del objective, search_space
        return GraphTuningResult(
            best_score=1.0,
            best_params=GraphContaminationParameters(
                bg_intensity_thresh=198,
                k=BEST_GRAPH_K,
                min_size=200,
                erosion_px=0,
            ),
        )

    summary = run_graph_tuning_pipeline(
        _pipeline_config(
            master_manifest_path=master_manifest_path,
            review_base_dir=review_dir,
            logger_name="test_graph_pipeline_hdf5",
            scorer=fake_scorer,
            optimizer=fake_optimizer,
        )
    )

    assert summary.best_params.k == BEST_GRAPH_K
    assert summary.final_tau == pytest.approx(0.21, abs=0.05)


def test_split_records_keeps_patient_groups_disjoint() -> None:
    records = [
        _make_record("CANCER_PATIENT_1_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_1_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_2_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_2_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_3_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_3_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_4_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_4_patch_b", "Rejected"),
    ]

    train_records, test_records = _split_records(records, test_set_size=0.25, random_state=7)

    assert {record.group_id for record in train_records}.isdisjoint(
        {record.group_id for record in test_records}
    )


def test_build_grouped_cv_splits_keeps_groups_disjoint_between_train_and_validation() -> None:
    records = [
        _make_record("CANCER_PATIENT_1_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_1_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_2_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_2_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_3_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_3_patch_b", "Rejected"),
        _make_record("CANCER_PATIENT_4_patch_a", "Approved"),
        _make_record("CANCER_PATIENT_4_patch_b", "Rejected"),
    ]
    labels = np.array([record.label for record in records])
    groups = np.array([record.group_id for record in records], dtype=object)

    splits = _build_grouped_cv_splits(
        labels=labels,
        groups=groups,
        n_splits_inner_cv=2,
        random_state=11,
    )

    assert splits
    for train_indices, validation_indices in splits:
        train_groups = {groups[index] for index in train_indices.tolist()}
        validation_groups = {groups[index] for index in validation_indices.tolist()}
        assert train_groups.isdisjoint(validation_groups)


def test_build_objective_scores_each_record_once_per_parameter_set() -> None:
    records = [
        _make_record("a_PATIENT_1", "Approved"),
        _make_record("b_PATIENT_2", "Approved"),
        _make_record("c_PATIENT_3", "Approved"),
        _make_record("d_PATIENT_4", "Approved"),
        _make_record("e_PATIENT_5", "Rejected"),
        _make_record("f_PATIENT_6", "Rejected"),
        _make_record("g_PATIENT_7", "Rejected"),
        _make_record("h_PATIENT_8", "Rejected"),
    ]
    scored_paths: list[str] = []

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        scored_paths.append(str(image_path))
        return 0.9 if any(token in str(image_path) for token in ("e_", "f_", "g_", "h_")) else 0.1

    objective = _build_objective(
        train_records=records,
        n_splits_inner_cv=2,
        random_state=42,
        scorer=fake_scorer,
    )

    score = objective([198, 386, 200, 0])

    assert score <= 0.0
    assert len(scored_paths) == len(records)
    assert set(scored_paths) == {str(record.pair.image_path) for record in records}


def test_build_recommendation_message_includes_final_tau() -> None:
    summary = GraphTuningSummary(
        total_labeled_pairs=10,
        training_pairs=8,
        test_pairs=2,
        best_cross_validated_f1=0.9,
        best_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        final_tau=0.24,
    )

    message = build_recommendation_message(summary)

    assert "Contamination Rate Threshold (tau):    0.24" in message


def test_build_graph_cleaning_parameter_artifact_maps_summary_to_artifact() -> None:
    summary = GraphTuningSummary(
        total_labeled_pairs=10,
        training_pairs=8,
        test_pairs=2,
        best_cross_validated_f1=0.9,
        best_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        final_tau=0.24,
    )

    artifact = build_graph_cleaning_parameter_artifact(summary, random_state=42)

    assert artifact == GraphCleaningParameterArtifact(
        graph_params=summary.best_params,
        tau=0.24,
        best_cross_validated_f1=0.9,
        total_labeled_pairs=10,
        training_pairs=8,
        test_pairs=2,
        random_state=42,
        generated_by="3_2_tune_graph_method.py",
    )


def _make_record(stem: str, label: str) -> LabeledSourceRecord:
    image_path = Path(f"/tmp/{stem}.png")
    mask_path = Path(f"/tmp/{stem}.png")

    return LabeledSourceRecord(
        pair=ImageMaskPair(
            stem=stem,
            label=1,
            image_path=image_path,
            mask_path=mask_path,
            output_name=f"{stem}.png",
        ),
        label=label,
        group_id=infer_group_id_from_stem(stem),
    )
