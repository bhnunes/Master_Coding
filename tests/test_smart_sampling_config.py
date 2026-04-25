from pathlib import Path

import pytest

from helpers.smart_sampling.config import load_smart_sampler_config

ZERO_CACHE_BYTES = 0
KEEP_STEP = 64
KEEP_PATIENCE = 2
DEFAULT_SEED = 42
DEFAULT_N_START = 512
MASK_FRACTION_ZERO = 0.0
MASK_FRACTION_THRESHOLD = 0.25
LOCAL_CACHE_BYTES = 4096


@pytest.fixture(autouse=True)
def _stub_cuda_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("helpers.smart_sampling.config.require_cuda_device", lambda: object())


def test_load_smart_sampler_config_reads_defaults(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
        }
    )

    assert config.master_manifest_path == master_manifest_path
    assert config.output_dir == tmp_path / "STAGE6_SMART_SAMPLER"
    assert config.output_filename == "TRAIN_FILTERED_shards"
    assert config.stage_input_locally is False
    assert config.stage_outputs_locally is False
    assert config.clean_local_work_dir is True
    assert config.patient_shard_cache_dir is None
    assert config.patient_shard_cache_bytes == ZERO_CACHE_BYTES
    assert config.model_name == "owkin/phikon-v2"
    assert config.adaptive_keep_enabled is True
    assert config.keep_min == KEEP_STEP
    assert config.keep_step == KEEP_STEP
    assert config.keep_improvement_threshold == pytest.approx(0.02)
    assert config.keep_patience == KEEP_PATIENCE
    assert config.write_sidecars is True
    assert config.seed == DEFAULT_SEED
    assert config.n_start == DEFAULT_N_START
    assert config.device == "cuda"
    assert config.use_gist is False
    assert config.protect_positive_labels is True
    assert config.protect_mask_positive is True
    assert config.positive_mask_fraction_threshold == pytest.approx(MASK_FRACTION_ZERO)
    assert config.log_path == Path("logs/smart_sampler.log")


def test_load_smart_sampler_config_rejects_invalid_growth_factor(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    with pytest.raises(ValueError, match="SMART_SAMPLER_GROWTH_FACTOR"):
        load_smart_sampler_config(
            {
                "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
                "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
                "SMART_SAMPLER_GROWTH_FACTOR": "1.0",
            }
        )


def test_load_smart_sampler_config_reads_gist_flag_and_legacy_alias(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_USE_GIST": "true",
        }
    )
    alias_config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out2"),
            "USE_GIST_SCRIPT": "true",
        }
    )

    assert config.use_gist is True
    assert alias_config.use_gist is True


def test_load_smart_sampler_config_accepts_output_dir_under_manifest_root(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "custom_stage6"),
        }
    )

    assert config.output_dir == tmp_path / "custom_stage6"


def test_load_smart_sampler_config_rejects_output_dir_outside_manifest_root(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    master_manifest_path = manifest_root / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    with pytest.raises(ValueError, match="SMART_SAMPLER_OUTPUT_DIR must be under"):
        load_smart_sampler_config(
            {
                "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
                "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "CHILE_RESULTS"),
            }
        )


def test_load_smart_sampler_config_reads_protection_settings(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
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
    assert config.positive_mask_fraction_threshold == pytest.approx(MASK_FRACTION_THRESHOLD)


def test_load_smart_sampler_config_reads_local_cache_settings(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"sqlite")

    config = load_smart_sampler_config(
        {
            "SMART_SAMPLER_MASTER_MANIFEST_PATH": str(master_manifest_path),
            "SMART_SAMPLER_OUTPUT_DIR": str(tmp_path / "out"),
            "SMART_SAMPLER_LOCAL_SHARD_CACHE_DIR": str(tmp_path / "cache"),
            "SMART_SAMPLER_LOCAL_SHARD_CACHE_BYTES": "4096",
        }
    )

    assert config.patient_shard_cache_dir == tmp_path / "cache"
    assert config.patient_shard_cache_bytes == LOCAL_CACHE_BYTES
