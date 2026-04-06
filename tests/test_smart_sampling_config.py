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
    assert config.stage_outputs_locally is False
    assert config.clean_local_work_dir is True
    assert config.adaptive_keep_enabled is True
    assert config.keep_min == 64
    assert config.keep_step == 64
    assert config.keep_improvement_threshold == pytest.approx(0.02)
    assert config.keep_patience == 2
    assert config.write_sidecars is True
    assert config.seed == 42
    assert config.n_start == 512
    assert config.device in {"cpu", "cuda"}
    assert config.use_gist is False
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


def test_load_smart_sampler_config_reads_gist_flag_and_legacy_alias(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    source_path.write_bytes(b"h5")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_H5": str(source_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_USE_GIST": "true",
        }
    )
    alias_config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_H5": str(source_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out2"),
            "USE_GIST_SCRIPT": "true",
        }
    )

    assert config.use_gist is True
    assert alias_config.use_gist is True
