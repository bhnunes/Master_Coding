from pathlib import Path

import pytest

from helpers.graph.tuning_config import load_graph_tuning_config


def test_load_graph_tuning_config_reads_expected_environment(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")
    config = load_graph_tuning_config(
        {
            "GRAPH_TUNING_SOURCE_HDF5_PATH": str(source_path),
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

    assert config.source_hdf5_path == source_path
    assert config.review_base_dir == tmp_path / "review"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "graph.log"
    assert config.test_set_size == 0.25
    assert config.n_splits_inner_cv == 4
    assert config.n_bayesian_calls == 60
    assert config.n_initial_points == 12
    assert config.random_state == 99
    assert config.bg_intensity_range == (120, 220)
    assert config.k_range == (110, 510)
    assert config.min_size_range == (20, 180)
    assert config.erosion_range == (1, 8)


def test_load_graph_tuning_config_requires_hdf5_source() -> None:
    with pytest.raises(ValueError, match="GRAPH_TUNING_SOURCE_HDF5_PATH"):
        load_graph_tuning_config({})


def test_load_graph_tuning_config_allows_hdf5_source_without_png_dirs(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_graph_tuning_config(
        {
            "GRAPH_TUNING_SOURCE_HDF5_PATH": str(source_path),
            "GRAPH_TUNING_BASE_DIR": str(tmp_path / "review"),
        }
    )

    assert config.source_hdf5_path == source_path
