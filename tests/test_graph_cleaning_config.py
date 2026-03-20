from pathlib import Path

import pytest

from helpers.graph_cleaning_config import load_graph_cleaning_config
from helpers.graph_contamination import GraphContaminationParameters


def test_load_graph_cleaning_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_graph_cleaning_config(
        {
            "GRAPH_CLEANING_SOURCE_IMAGE_DIR": str(tmp_path / "images"),
            "GRAPH_CLEANING_SOURCE_MASK_DIR": str(tmp_path / "masks"),
            "GRAPH_CLEANING_OUTPUT_BASE_DIR": str(tmp_path / "output"),
            "GRAPH_CLEANING_LOG_FOLDER": str(tmp_path / "logs"),
            "GRAPH_CLEANING_LOG_FILE": "cleaning.log",
            "GRAPH_CLEANING_NUM_WORKERS": "3",
            "GRAPH_CLEANING_BG_INTENSITY_THRESH": "198",
            "GRAPH_CLEANING_K": "386",
            "GRAPH_CLEANING_MIN_SIZE": "200",
            "GRAPH_CLEANING_EROSION_PX": "0",
            "GRAPH_CLEANING_TAU": "0.24",
        }
    )

    assert config.source_image_dir == tmp_path / "images"
    assert config.source_mask_dir == tmp_path / "masks"
    assert config.output_base_dir == tmp_path / "output"
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "cleaning.log"
    assert config.num_workers == 3
    assert config.tau == 0.24
    assert config.graph_params == GraphContaminationParameters(
        bg_intensity_thresh=198,
        k=386.0,
        min_size=200,
        erosion_px=0,
    )


def test_load_graph_cleaning_config_requires_source_image_dir() -> None:
    with pytest.raises(ValueError, match="GRAPH_CLEANING_SOURCE_IMAGE_DIR"):
        load_graph_cleaning_config({})
