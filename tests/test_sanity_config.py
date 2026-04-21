from pathlib import Path

import pytest

from helpers.sanity.config import load_sanity_config

DEFAULT_SAMPLE_PAIRS = 1000


def test_load_sanity_config_reads_defaults(tmp_path: Path) -> None:
    config = load_sanity_config({"SANITY_BASE_DIR": str(tmp_path)})

    assert config.base_dir == tmp_path
    assert config.sample_pairs == DEFAULT_SAMPLE_PAIRS
    assert config.full_mask_scan is False
    assert config.full_shape_scan is False
    assert config.checksum_mode == "SAMPLE"
    assert config.enforce_filename_uniqueness is True
    assert config.enforce_regex_patient_id_match is True
    assert config.enforce_split_stats_parity is True
    assert config.fail_on_empty_cancer_mask is True
    assert config.fail_on_positive_not_cancer_mask is True
    assert config.log_path == Path("logs/sanity_checks.log")


def test_load_sanity_config_rejects_invalid_checksum_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SANITY_CHECKSUM_MODE"):
        load_sanity_config(
            {
                "SANITY_BASE_DIR": str(tmp_path),
                "SANITY_CHECKSUM_MODE": "sometimes",
            }
        )
