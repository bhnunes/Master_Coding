from pathlib import Path

import pytest

from helpers.optimization_sampling_config import load_optimization_sampling_config


def test_load_optimization_sampling_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_optimization_sampling_config(
        {
            "OPTIMIZATION_SAMPLING_IMAGE_FOLDER": str(tmp_path / "images"),
            "OPTIMIZATION_SAMPLING_MASK_FOLDER": str(tmp_path / "masks"),
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
            "OPTIMIZATION_SAMPLING_NUM_PROCESSES": "3",
        }
    )

    assert config.image_folder == tmp_path / "images"
    assert config.mask_folder == tmp_path / "masks"
    assert config.output_base == tmp_path / "output"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "stage4.log"
    assert config.confidence_level == 0.99
    assert config.margin_of_error == 0.1
    assert config.proportion == 0.4
    assert config.pilot_sample_size == 50
    assert config.master_pool_fraction == 0.2
    assert config.overlay_color == (1, 2, 3)
    assert config.overlay_thickness == 4
    assert config.overlay_alpha == 0.5
    assert config.num_processes == 3


def test_load_optimization_sampling_config_requires_source_and_output_paths() -> None:
    with pytest.raises(ValueError, match="OPTIMIZATION_SAMPLING_IMAGE_FOLDER"):
        load_optimization_sampling_config({})
