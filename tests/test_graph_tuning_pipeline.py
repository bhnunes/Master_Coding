from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from helpers.graph_contamination import GraphContaminationParameters
from helpers.graph_tuning_pipeline import (
    GraphTuningResult,
    GraphTuningSummary,
    build_recommendation_message,
    collect_review_labels,
    resolve_labeled_source_records,
    run_graph_tuning_pipeline,
    select_best_contamination_threshold,
)


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
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "case_a.png").write_bytes(b"image")
    (mask_dir / "case_a.png").write_bytes(b"mask")

    logger = logging.getLogger("test_graph_tuning")
    records = resolve_labeled_source_records(
        labels_by_stem={"case_a": "Approved", "case_b": "Rejected"},
        source_image_folder=image_dir,
        source_mask_folder=mask_dir,
        logger=logger,
    )

    assert [(record.pair.stem, record.label) for record in records] == [("case_a", "Approved")]


def test_select_best_contamination_threshold_optimizes_rejected_f1() -> None:
    rates = [0.1, 0.2, 0.8, 0.9]
    labels = ["Approved", "Approved", "Rejected", "Rejected"]

    tau = select_best_contamination_threshold(rates, labels, thresholds=np.array([0.15, 0.5, 0.85]))

    assert tau == pytest.approx(0.5)


def test_run_graph_tuning_pipeline_uses_optimizer_result_and_returns_summary(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    review_dir = tmp_path / "review"
    approved_dir = review_dir / "APPROVED"
    rejected_dir = review_dir / "REJECTED"
    image_dir.mkdir()
    mask_dir.mkdir()
    approved_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)

    for name in ("a1", "a2", "r1", "r2"):
        (image_dir / f"{name}.png").write_bytes(b"image")
        (mask_dir / f"{name}.png").write_bytes(b"mask")
    for name in ("a1", "a2"):
        (approved_dir / f"{name}.png").write_bytes(b"overlay")
    for name in ("r1", "r2"):
        (rejected_dir / f"{name}.png").write_bytes(b"overlay")

    score_map = {
        "a1": 0.10,
        "a2": 0.20,
        "r1": 0.80,
        "r2": 0.90,
    }

    def fake_scorer(
        image_path: Path, mask_path: Path, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        return score_map[image_path.stem]

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
        source_image_folder=image_dir,
        source_mask_folder=mask_dir,
        review_base_dir=review_dir,
        test_set_size=0.5,
        n_splits_inner_cv=2,
        n_bayesian_calls=10,
        n_initial_points=4,
        random_state=42,
        bg_intensity_range=(100, 250),
        k_range=(100, 500),
        min_size_range=(10, 200),
        erosion_range=(0, 10),
        logger=logging.getLogger("test_graph_pipeline"),
        scorer=fake_scorer,
        optimizer=fake_optimizer,
        progress_factory=lambda iterable, **_: iterable,
    )

    assert isinstance(summary, GraphTuningSummary)
    assert summary.best_params.k == 386.0
    assert summary.final_tau == pytest.approx(0.21, abs=0.05)
    assert summary.best_cross_validated_f1 == 1.0


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
