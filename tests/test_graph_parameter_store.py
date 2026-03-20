from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.graph_contamination import GraphContaminationParameters
from helpers.graph_parameter_store import (
    GraphCleaningParameterArtifact,
    load_graph_cleaning_parameter_artifact,
    save_graph_cleaning_parameter_artifact,
)


def test_save_and_load_graph_cleaning_parameter_artifact_round_trip(tmp_path: Path) -> None:
    artifact = GraphCleaningParameterArtifact(
        graph_params=GraphContaminationParameters(
            bg_intensity_thresh=198,
            k=386.0,
            min_size=200,
            erosion_px=0,
        ),
        tau=0.24,
        best_cross_validated_f1=0.91,
        total_labeled_pairs=412,
        training_pairs=329,
        test_pairs=83,
        random_state=42,
        generated_by="4_2_tune_graph_method.py",
    )
    output_path = tmp_path / "graph_cleaning_params.json"

    save_graph_cleaning_parameter_artifact(output_path, artifact)
    loaded = load_graph_cleaning_parameter_artifact(output_path)

    assert loaded == artifact


def test_load_graph_cleaning_parameter_artifact_rejects_invalid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "graph_cleaning_params.json"
    output_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_graph_cleaning_parameter_artifact(output_path)


def test_load_graph_cleaning_parameter_artifact_requires_expected_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "graph_cleaning_params.json"
    output_path.write_text(json.dumps({"tau": 0.24}), encoding="utf-8")

    with pytest.raises(ValueError, match="graph_params"):
        load_graph_cleaning_parameter_artifact(output_path)
