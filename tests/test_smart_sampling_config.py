from pathlib import Path

import pytest

from helpers.smart_sampling.config import load_smart_sampler_config


def test_load_smart_sampler_config_reads_defaults(tmp_path: Path) -> None:
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()
    manifest_path = shard_dir / "manifest.parquet"
    manifest_path.write_bytes(b"parquet")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
        }
    )

    assert config.source_h5_path is None
    assert config.source_shard_dir == shard_dir
    assert config.source_manifest_path == shard_dir / "manifest.parquet"
    assert config.output_dir == tmp_path / "out"
    assert config.output_filename == "TRAIN_FILTERED_shards"
    assert config.stage_input_locally is False
    assert config.stage_outputs_locally is False
    assert config.clean_local_work_dir is True
    assert config.patient_shard_cache_dir is None
    assert config.patient_shard_cache_bytes == 0
    assert config.model_name == "owkin/phikon-v2"
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
    assert config.protect_positive_labels is True
    assert config.protect_mask_positive is True
    assert config.positive_mask_fraction_threshold == pytest.approx(0.0)
    assert config.log_path == Path("logs/smart_sampler.log")


def test_load_smart_sampler_config_rejects_invalid_growth_factor(tmp_path: Path) -> None:
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()

    with pytest.raises(ValueError, match="SMART_SAMPLER_GROWTH_FACTOR"):
        load_smart_sampler_config(
            {
                "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
                "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
                "SMART_SAMPLER_GROWTH_FACTOR": "1.0",
            }
        )


def test_load_smart_sampler_config_reads_gist_flag_and_legacy_alias(tmp_path: Path) -> None:
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_USE_GIST": "true",
        }
    )
    alias_config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out2"),
            "USE_GIST_SCRIPT": "true",
        }
    )

    assert config.use_gist is True
    assert alias_config.use_gist is True


def test_load_smart_sampler_config_reads_protection_settings(tmp_path: Path) -> None:
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_MODEL_NAME": "custom/phikon",
            "SMART_SAMPLER_PROTECT_POSITIVE_LABELS": "false",
            "SMART_SAMPLER_PROTECT_MASK_POSITIVE": "true",
            "SMART_SAMPLER_POSITIVE_MASK_FRACTION_THRESHOLD": "0.25",
        }
    )

    assert config.model_name == "custom/phikon"
    assert config.protect_positive_labels is False
    assert config.protect_mask_positive is True
    assert config.positive_mask_fraction_threshold == pytest.approx(0.25)


def test_load_smart_sampler_config_reads_local_cache_settings(tmp_path: Path) -> None:
    shard_dir = tmp_path / "TRAIN_shards"
    shard_dir.mkdir()

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_SOURCE_SHARD_DIR": str(shard_dir),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_LOCAL_SHARD_CACHE_DIR": str(tmp_path / "cache"),
            "SMART_SAMPLER_LOCAL_SHARD_CACHE_BYTES": "4096",
        }
    )

    assert config.patient_shard_cache_dir == tmp_path / "cache"
    assert config.patient_shard_cache_bytes == 4096
