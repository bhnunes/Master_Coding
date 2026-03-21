from pathlib import Path

import pytest

from helpers.smart_sampling.config import load_smart_sampler_config


def test_load_smart_sampler_config_reads_defaults(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    source_path.write_bytes(b"h5")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_H5": str(source_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
        }
    )

    assert config.source_h5_path == source_path
    assert config.output_dir == tmp_path / "out"
    assert config.output_filename == "TRAIN_FILTERED.h5"
    assert config.stage_input_locally is False
    assert config.write_sidecars is True
    assert config.seed == 42
    assert config.n_start == 512
    assert config.device in {"cpu", "cuda"}
    assert config.log_path == Path("logs/smart_sampler.log")


def test_load_smart_sampler_config_rejects_invalid_growth_factor(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    source_path.write_bytes(b"h5")

    with pytest.raises(ValueError, match="SMART_SAMPLER_GROWTH_FACTOR"):
        load_smart_sampler_config(
            {
                "SMART_SAMPLER_SOURCE_H5": str(source_path),
                "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
                "SMART_SAMPLER_GROWTH_FACTOR": "1.0",
            }
        )
