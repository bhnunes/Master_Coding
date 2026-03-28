from __future__ import annotations

import csv
import logging
from pathlib import Path

import h5py
import numpy as np

from helpers.graph.cleaning_pipeline import run_graph_cleaning_pipeline
from helpers.graph.contamination import GraphContaminationParameters


def test_run_graph_cleaning_pipeline_reports_empty_source_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    output_dir = tmp_path / "output"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((0, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((0, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.zeros((0,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.zeros((0,), dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([], dtype="S32"))

    summary = run_graph_cleaning_pipeline(
        source_hdf5_path=source_path,
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


def test_run_graph_cleaning_pipeline_writes_hdf5_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    output_dir = tmp_path / "output"
    names = ["accepted_PATIENT_1", "rejected_PATIENT_2"]
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.ones((2,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 2], dtype=np.int32))
        handle.create_dataset(
            "filenames", data=np.array([f"{name}.png".encode() for name in names])
        )
        handle.create_dataset("slide_ids", data=np.array([b"slide_a", b"slide_b"]))

    scores = {0: 0.10, 1: 0.50}

    def fake_scorer(
        image_path: Path | str, mask_path: Path | str, params: GraphContaminationParameters
    ) -> float | None:
        del mask_path, params
        row_index = int(str(image_path).rsplit("[", maxsplit=1)[1][:-1])
        return scores[row_index]

    summary = run_graph_cleaning_pipeline(
        source_hdf5_path=source_path,
        output_base_dir=output_dir,
        graph_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        tau=0.24,
        num_workers=1,
        logger=logging.getLogger("test_graph_cleaning_hdf5"),
        scorer=fake_scorer,
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
    assert accepted_rows[0]["source_row_index"] == "0"
    assert rejected_rows[0]["filename"] == "rejected_PATIENT_2.png"
    assert rejected_rows[0]["source_row_index"] == "1"
