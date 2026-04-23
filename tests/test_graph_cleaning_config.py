import json
from pathlib import Path

import pytest

from helpers.graph.cleaning_config import load_graph_cleaning_config
from helpers.graph.contamination import GraphContaminationParameters

NUM_WORKERS = 3
TAU = 0.5
BG_INTENSITY_THRESHOLD = 111
GRAPH_SCALE = 222.0
MIN_COMPONENT_SIZE = 33
EROSION_PIXELS = 4


def _write_artifact(artifact_path: Path) -> None:
    artifact_path.write_text(
        json.dumps(
            {
                "graph_params": {
                    "bg_intensity_thresh": 111,
                    "k": 222.0,
                    "min_size": 33,
                    "erosion_px": 4,
                },
                "tau": 0.5,
                "best_cross_validated_f1": 0.9,
                "total_labeled_pairs": 10,
                "training_pairs": 8,
                "test_pairs": 2,
                "random_state": 42,
                "generated_by": "3_2_tune_graph_method.py",
            }
        ),
        encoding="utf-8",
    )


def test_load_graph_cleaning_config_reads_expected_environment(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")
    artifact_path = tmp_path / "graph_cleaning_params.json"
    _write_artifact(artifact_path)
    config = load_graph_cleaning_config(
        {
            "GRAPH_CLEANING_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "GRAPH_CLEANING_OUTPUT_BASE_DIR": str(tmp_path / "output"),
            "GRAPH_CLEANING_LOG_FOLDER": str(tmp_path / "logs"),
            "GRAPH_CLEANING_LOG_FILE": "cleaning.log",
            "GRAPH_CLEANING_PARAMS_PATH": str(artifact_path),
            "GRAPH_CLEANING_NUM_WORKERS": "3",
        }
    )

    assert config.master_manifest_path == master_manifest_path
    assert config.output_base_dir == tmp_path / "output"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "cleaning.log"
    assert config.params_path == artifact_path
    assert config.num_workers == NUM_WORKERS
    assert config.tau == TAU
    assert config.graph_params == GraphContaminationParameters(
        bg_intensity_thresh=BG_INTENSITY_THRESHOLD,
        k=GRAPH_SCALE,
        min_size=MIN_COMPONENT_SIZE,
        erosion_px=EROSION_PIXELS,
    )


def test_load_graph_cleaning_config_requires_master_manifest() -> None:
    with pytest.raises(ValueError, match="GRAPH_CLEANING_MASTER_MANIFEST_PATH"):
        load_graph_cleaning_config({})


def test_load_graph_cleaning_config_requires_parameter_artifact(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    with pytest.raises(ValueError, match="GRAPH_CLEANING_PARAMS_PATH"):
        load_graph_cleaning_config(
            {
                "GRAPH_CLEANING_MASTER_MANIFEST_PATH": str(master_manifest_path),
                "GRAPH_CLEANING_OUTPUT_BASE_DIR": str(tmp_path / "output"),
            }
        )


def test_load_graph_cleaning_config_reads_parameter_artifact(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")
    artifact_path = tmp_path / "graph_cleaning_params.json"
    _write_artifact(artifact_path)

    config = load_graph_cleaning_config(
        {
            "GRAPH_CLEANING_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "GRAPH_CLEANING_OUTPUT_BASE_DIR": str(tmp_path / "output"),
            "GRAPH_CLEANING_PARAMS_PATH": str(artifact_path),
        }
    )

    assert config.params_path == artifact_path
    assert config.tau == TAU
    assert config.graph_params == GraphContaminationParameters(
        bg_intensity_thresh=BG_INTENSITY_THRESHOLD,
        k=GRAPH_SCALE,
        min_size=MIN_COMPONENT_SIZE,
        erosion_px=EROSION_PIXELS,
    )


def test_load_graph_cleaning_config_requires_master_manifest_file(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "PATCHES"
    master_manifest_path.mkdir()
    artifact_path = tmp_path / "graph_cleaning_params.json"
    _write_artifact(artifact_path)

    with pytest.raises(ValueError, match="must point to the Stage 2 master_manifest.sqlite file"):
        load_graph_cleaning_config(
            {
                "GRAPH_CLEANING_MASTER_MANIFEST_PATH": str(master_manifest_path),
                "GRAPH_CLEANING_OUTPUT_BASE_DIR": str(tmp_path / "output"),
                "GRAPH_CLEANING_PARAMS_PATH": str(artifact_path),
            }
        )
