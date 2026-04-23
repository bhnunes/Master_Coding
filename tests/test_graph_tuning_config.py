from pathlib import Path

import pytest

from helpers.graph.tuning_config import load_graph_tuning_config

TEST_SET_SIZE = 0.25
INNER_CV_SPLITS = 4
BAYESIAN_CALLS = 60
INITIAL_POINTS = 12
RANDOM_STATE = 99
BG_INTENSITY_RANGE = (120, 220)
K_RANGE = (110, 510)
MIN_SIZE_RANGE = (20, 180)
EROSION_RANGE = (1, 8)


def test_load_graph_tuning_config_reads_expected_environment(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")
    config = load_graph_tuning_config(
        {
            "GRAPH_TUNING_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "GRAPH_TUNING_BASE_DIR": str(tmp_path / "review"),
            "GRAPH_TUNING_LOG_FOLDER": str(tmp_path / "logs"),
            "GRAPH_TUNING_LOG_FILE": "graph.log",
            "GRAPH_TUNING_TEST_SET_SIZE": "0.25",
            "GRAPH_TUNING_N_SPLITS_INNER_CV": "4",
            "GRAPH_TUNING_N_BAYESIAN_CALLS": "60",
            "GRAPH_TUNING_N_INITIAL_POINTS": "12",
            "GRAPH_TUNING_RANDOM_STATE": "99",
            "GRAPH_TUNING_BG_INTENSITY_MIN": "120",
            "GRAPH_TUNING_BG_INTENSITY_MAX": "220",
            "GRAPH_TUNING_K_MIN": "110",
            "GRAPH_TUNING_K_MAX": "510",
            "GRAPH_TUNING_MIN_SIZE_MIN": "20",
            "GRAPH_TUNING_MIN_SIZE_MAX": "180",
            "GRAPH_TUNING_EROSION_MIN": "1",
            "GRAPH_TUNING_EROSION_MAX": "8",
        }
    )

    assert config.master_manifest_path == master_manifest_path
    assert config.review_base_dir == tmp_path / "review"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "graph.log"
    assert config.test_set_size == TEST_SET_SIZE
    assert config.n_splits_inner_cv == INNER_CV_SPLITS
    assert config.n_bayesian_calls == BAYESIAN_CALLS
    assert config.n_initial_points == INITIAL_POINTS
    assert config.random_state == RANDOM_STATE
    assert config.bg_intensity_range == BG_INTENSITY_RANGE
    assert config.k_range == K_RANGE
    assert config.min_size_range == MIN_SIZE_RANGE
    assert config.erosion_range == EROSION_RANGE


def test_load_graph_tuning_config_requires_master_manifest() -> None:
    with pytest.raises(ValueError, match="GRAPH_TUNING_MASTER_MANIFEST_PATH"):
        load_graph_tuning_config({})


def test_load_graph_tuning_config_requires_master_manifest_file(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "PATCHES"
    master_manifest_path.mkdir()

    with pytest.raises(ValueError, match="must point to the Stage 2 master_manifest.sqlite file"):
        load_graph_tuning_config(
            {
                "GRAPH_TUNING_MASTER_MANIFEST_PATH": str(master_manifest_path),
                "GRAPH_TUNING_BASE_DIR": str(tmp_path / "review"),
            }
        )


def test_load_graph_tuning_config_accepts_master_manifest_file(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_graph_tuning_config(
        {
            "GRAPH_TUNING_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "GRAPH_TUNING_BASE_DIR": str(tmp_path / "review"),
        }
    )

    assert config.master_manifest_path == master_manifest_path
