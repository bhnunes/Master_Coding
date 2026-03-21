from pathlib import Path

import pytest

from helpers.packaging.config import load_packaging_config


def test_load_packaging_config_reads_defaults(tmp_path: Path) -> None:
    config = load_packaging_config({"PACKAGING_BASE_DIR": str(tmp_path)})

    assert config.base_dir == tmp_path
    assert config.output_dir == tmp_path
    assert config.splits == ("TRAIN", "VALIDATION", "TEST")
    assert config.img_size == 224
    assert config.patient_id_regex == r"PATIENT_(\d+)_"
    assert config.overwrite_outputs is False
    assert config.log_path == Path("logs/packaging.log")


def test_load_packaging_config_rejects_invalid_img_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PACKAGING_IMG_SIZE"):
        load_packaging_config(
            {
                "PACKAGING_BASE_DIR": str(tmp_path),
                "PACKAGING_IMG_SIZE": "0",
            }
        )
