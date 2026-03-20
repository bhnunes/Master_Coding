from __future__ import annotations

import logging
from pathlib import Path

from helpers.graph_cleaning_pipeline import GraphCleaningSummary, run_graph_cleaning_pipeline
from helpers.graph_contamination import GraphContaminationParameters


def test_run_graph_cleaning_pipeline_moves_only_rejected_pairs(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    mask_dir.mkdir()

    for name in ("accepted", "rejected"):
        (image_dir / f"{name}.png").write_bytes(b"image")
        (mask_dir / f"{name}.png").write_bytes(b"mask")

    scores = {"accepted": 0.10, "rejected": 0.50}

    def fake_scorer(
        image_path: Path, mask_path: Path, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        return scores[image_path.stem]

    summary = run_graph_cleaning_pipeline(
        source_image_dir=image_dir,
        source_mask_dir=mask_dir,
        output_base_dir=output_dir,
        graph_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        tau=0.24,
        num_workers=1,
        logger=logging.getLogger("test_graph_cleaning"),
        scorer=fake_scorer,
    )

    assert isinstance(summary, GraphCleaningSummary)
    assert summary.accepted == 1
    assert summary.rejected == 1
    assert summary.skipped == 0
    assert (image_dir / "accepted.png").exists()
    assert (mask_dir / "accepted.png").exists()
    assert (output_dir / "REJECTED_IMAGES" / "rejected.png").exists()
    assert (output_dir / "REJECTED_MASKS" / "rejected.png").exists()


def test_run_graph_cleaning_pipeline_skips_missing_masks(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "sample.png").write_bytes(b"image")

    summary = run_graph_cleaning_pipeline(
        source_image_dir=image_dir,
        source_mask_dir=mask_dir,
        output_base_dir=output_dir,
        graph_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        tau=0.24,
        num_workers=1,
        logger=logging.getLogger("test_graph_cleaning_missing_mask"),
    )

    assert summary.accepted == 0
    assert summary.rejected == 0
    assert summary.skipped == 1


def test_run_graph_cleaning_pipeline_reports_empty_source_folder(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    mask_dir.mkdir()

    summary = run_graph_cleaning_pipeline(
        source_image_dir=image_dir,
        source_mask_dir=mask_dir,
        output_base_dir=output_dir,
        graph_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        tau=0.24,
        num_workers=1,
        logger=logging.getLogger("test_graph_cleaning_empty"),
    )

    assert summary.total_images == 0
    assert summary.accepted == 0
    assert summary.rejected == 0
    assert summary.skipped == 0
