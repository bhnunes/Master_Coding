from pathlib import Path

import pytest

from helpers.optimization_sampling.config import load_optimization_sampling_config

CONFIDENCE_LEVEL = 0.99
MARGIN_OF_ERROR = 0.1
PROPORTION = 0.4
PILOT_SAMPLE_SIZE = 50
MASTER_POOL_FRACTION = 0.2
OVERLAY_COLOR = (1, 2, 3)
OVERLAY_THICKNESS = 4
OVERLAY_ALPHA = 0.5
NUM_PROCESSES = 3
SEED = 987
DEFAULT_SEED = 42


def test_load_optimization_sampling_config_reads_expected_environment(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")
    config = load_optimization_sampling_config(
        {
            "OPTIMIZATION_SAMPLING_SOURCE_HDF5_PATH": str(source_path),
            "OPTIMIZATION_SAMPLING_OUTPUT_BASE": str(tmp_path / "output"),
            "OPTIMIZATION_SAMPLING_LOG_FOLDER": str(tmp_path / "logs"),
            "OPTIMIZATION_SAMPLING_LOG_FILE": "stage4.log",
            "OPTIMIZATION_SAMPLING_CONFIDENCE_LEVEL": "0.99",
            "OPTIMIZATION_SAMPLING_MARGIN_OF_ERROR": "0.1",
            "OPTIMIZATION_SAMPLING_PROPORTION": "0.4",
            "OPTIMIZATION_SAMPLING_PILOT_SAMPLE_SIZE": "50",
            "OPTIMIZATION_SAMPLING_MASTER_POOL_FRACTION": "0.2",
            "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_B": "1",
            "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_G": "2",
            "OPTIMIZATION_SAMPLING_OVERLAY_COLOR_R": "3",
            "OPTIMIZATION_SAMPLING_OVERLAY_THICKNESS": "4",
            "OPTIMIZATION_SAMPLING_OVERLAY_ALPHA": "0.5",
            "OPTIMIZATION_SAMPLING_SEED": "987",
            "OPTIMIZATION_SAMPLING_NUM_PROCESSES": "3",
        }
    )

    assert config.source_hdf5_path == source_path
    assert config.output_base == tmp_path / "output"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "stage4.log"
    assert config.confidence_level == CONFIDENCE_LEVEL
    assert config.margin_of_error == MARGIN_OF_ERROR
    assert config.proportion == PROPORTION
    assert config.pilot_sample_size == PILOT_SAMPLE_SIZE
    assert config.master_pool_fraction == MASTER_POOL_FRACTION
    assert config.overlay_color == OVERLAY_COLOR
    assert config.overlay_thickness == OVERLAY_THICKNESS
    assert config.overlay_alpha == OVERLAY_ALPHA
    assert config.seed == SEED
    assert config.num_processes == NUM_PROCESSES


def test_load_optimization_sampling_config_requires_source_and_output_paths() -> None:
    with pytest.raises(ValueError, match="OPTIMIZATION_SAMPLING_SOURCE_HDF5_PATH"):
        load_optimization_sampling_config({})


def test_load_optimization_sampling_config_uses_hdf5_source(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_optimization_sampling_config(
        {
            "OPTIMIZATION_SAMPLING_SOURCE_HDF5_PATH": str(source_path),
            "OPTIMIZATION_SAMPLING_OUTPUT_BASE": str(tmp_path / "output"),
        }
    )

    assert config.source_hdf5_path == source_path
    assert config.seed == DEFAULT_SEED
