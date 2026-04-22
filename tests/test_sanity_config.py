from pathlib import Path

import pytest

from helpers.sanity import config as sanity_config_module

DEFAULT_SAMPLE_PAIRS = 1000
CUSTOM_SAMPLE_PAIRS = 7
DEFAULT_PARSE_INT = 9
PARSED_INT = 17
SMOKE_SAMPLE_PAIRS = 5


def test_load_sanity_config_reads_defaults(tmp_path: Path) -> None:
    config = sanity_config_module.load_sanity_config({"SANITY_BASE_DIR": str(tmp_path)})

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


def test_load_sanity_config_reads_custom_values_and_trims_log_file(tmp_path: Path) -> None:
    config = sanity_config_module.load_sanity_config(
        {
            "SANITY_BASE_DIR": str(tmp_path),
            "SANITY_SAMPLE_PAIRS": "7",
            "SANITY_FULL_MASK_SCAN": "yes",
            "SANITY_FULL_SHAPE_SCAN": "on",
            "SANITY_CHECKSUM_MODE": " full ",
            "SANITY_ENFORCE_FILENAME_UNIQUENESS": "off",
            "SANITY_ENFORCE_REGEX_PATIENT_ID_MATCH": "0",
            "SANITY_ENFORCE_SPLIT_STATS_PARITY": "false",
            "SANITY_FAIL_ON_EMPTY_CANCER_MASK": "no",
            "SANITY_FAIL_ON_POSITIVE_NOT_CANCER_MASK": "0",
            "SANITY_LOG_FOLDER": str(tmp_path / "logs"),
            "SANITY_LOG_FILE": " custom.log ",
        }
    )

    assert config.base_dir == tmp_path
    assert config.sample_pairs == CUSTOM_SAMPLE_PAIRS
    assert config.full_mask_scan is True
    assert config.full_shape_scan is True
    assert config.checksum_mode == "FULL"
    assert config.enforce_filename_uniqueness is False
    assert config.enforce_regex_patient_id_match is False
    assert config.enforce_split_stats_parity is False
    assert config.fail_on_empty_cancer_mask is False
    assert config.fail_on_positive_not_cancer_mask is False
    assert config.log_folder == tmp_path / "logs"
    assert config.log_file_name == "custom.log"
    assert config.log_path == tmp_path / "logs" / "custom.log"


def test_load_sanity_config_rejects_invalid_checksum_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SANITY_CHECKSUM_MODE"):
        sanity_config_module.load_sanity_config(
            {
                "SANITY_BASE_DIR": str(tmp_path),
                "SANITY_CHECKSUM_MODE": "sometimes",
            }
        )


def test_sanity_config_helper_parsers_cover_defaults_and_validation() -> None:
    assert sanity_config_module._parse_bool(None, "FLAG", True) is True
    assert sanity_config_module._parse_bool("", "FLAG", False) is False
    assert sanity_config_module._parse_bool("1", "FLAG", False) is True
    assert sanity_config_module._parse_bool("true", "FLAG", False) is True
    assert sanity_config_module._parse_bool(" YES ", "FLAG", False) is True
    assert sanity_config_module._parse_bool(" Off ", "FLAG", True) is False

    with pytest.raises(
        ValueError,
        match=r"The 'FLAG' environment variable must be a boolean value\.",
    ):
        sanity_config_module._parse_bool("maybe", "FLAG", False)

    assert sanity_config_module._parse_int(None, "COUNT", DEFAULT_PARSE_INT) == DEFAULT_PARSE_INT
    assert sanity_config_module._parse_int("", "COUNT", DEFAULT_PARSE_INT) == DEFAULT_PARSE_INT
    assert sanity_config_module._parse_int("17", "COUNT", DEFAULT_PARSE_INT) == PARSED_INT

    assert sanity_config_module._parse_choice(None, "MODE", "sample") == "SAMPLE"
    assert sanity_config_module._parse_choice(" full ", "MODE", "sample") == "FULL"

    with pytest.raises(
        ValueError,
        match=r"The 'MODE' environment variable must be one of: OFF, SAMPLE, FULL\.",
    ):
        sanity_config_module._parse_choice("invalid", "MODE", "sample")


def test_required_path_forwards_required_and_system_name(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, str, str | None, bool]] = []

    def fake_resolve_env_path(
        value: str | None,
        variable_name: str,
        *,
        system_name: str | None = None,
        required: bool = False,
    ) -> Path:
        calls.append((value, variable_name, system_name, required))
        return Path("/resolved")

    monkeypatch.setattr(sanity_config_module, "resolve_env_path", fake_resolve_env_path)

    resolved = sanity_config_module._required_path(
        {"SANITY_BASE_DIR": "/data/base"},
        "SANITY_BASE_DIR",
        system_name="Windows",
    )

    assert resolved == Path("/resolved")
    assert calls == [("/data/base", "SANITY_BASE_DIR", "Windows", True)]


def test_load_sanity_config_namespace_smoke_uses_os_environ(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sanity_config_module,
        "os",
        type("FakeOS", (), {"environ": {"SANITY_BASE_DIR": str(tmp_path)}})(),
    )

    config = sanity_config_module.load_sanity_config()

    assert config.base_dir == tmp_path
    assert config.log_path == Path("logs/sanity_checks.log")


def test_load_sanity_config_forwards_system_name_to_required_path_and_log_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_path_calls: list[tuple[object, str, str | None]] = []
    log_folder_calls: list[tuple[object, str | None, tuple[str, ...]]] = []

    def fake_required_path(
        environment: object,
        variable_name: str,
        *,
        system_name: str | None = None,
    ) -> Path:
        required_path_calls.append((environment, variable_name, system_name))
        return Path("/data/base")

    def fake_resolve_log_folder(
        environment: object,
        *,
        system_name: str | None = None,
        fallback_names: tuple[str, ...] = (),
    ) -> Path:
        log_folder_calls.append((environment, system_name, fallback_names))
        return Path("/logs")

    monkeypatch.setattr(sanity_config_module, "_required_path", fake_required_path)
    monkeypatch.setattr(sanity_config_module, "resolve_log_folder", fake_resolve_log_folder)

    env = {"SANITY_BASE_DIR": "/ignored"}
    config = sanity_config_module.load_sanity_config(env, system_name="Windows")

    assert config.base_dir == Path("/data/base")
    assert config.log_folder == Path("/logs")
    assert required_path_calls == [(env, "SANITY_BASE_DIR", "Windows")]
    assert log_folder_calls == [(env, "Windows", ("SANITY_LOG_FOLDER",))]


def test_sanity_config_namespace_smoke_covers_defaults_and_validation(tmp_path: Path) -> None:
    default_config = sanity_config_module.load_sanity_config({"SANITY_BASE_DIR": str(tmp_path)})
    custom_config = sanity_config_module.load_sanity_config(
        {
            "SANITY_BASE_DIR": str(tmp_path),
            "SANITY_SAMPLE_PAIRS": "5",
            "SANITY_FULL_MASK_SCAN": "1",
            "SANITY_FULL_SHAPE_SCAN": "true",
            "SANITY_CHECKSUM_MODE": "off",
            "SANITY_ENFORCE_FILENAME_UNIQUENESS": "0",
            "SANITY_ENFORCE_REGEX_PATIENT_ID_MATCH": "0",
            "SANITY_ENFORCE_SPLIT_STATS_PARITY": "0",
            "SANITY_FAIL_ON_EMPTY_CANCER_MASK": "0",
            "SANITY_FAIL_ON_POSITIVE_NOT_CANCER_MASK": "0",
        }
    )

    assert default_config.sample_pairs == DEFAULT_SAMPLE_PAIRS
    assert default_config.full_mask_scan is False
    assert default_config.full_shape_scan is False
    assert default_config.checksum_mode == "SAMPLE"
    assert default_config.enforce_filename_uniqueness is True
    assert default_config.enforce_regex_patient_id_match is True
    assert default_config.enforce_split_stats_parity is True
    assert default_config.fail_on_empty_cancer_mask is True
    assert default_config.fail_on_positive_not_cancer_mask is True

    assert custom_config.sample_pairs == SMOKE_SAMPLE_PAIRS
    assert custom_config.full_mask_scan is True
    assert custom_config.full_shape_scan is True
    assert custom_config.checksum_mode == "OFF"
    assert custom_config.enforce_filename_uniqueness is False
    assert custom_config.enforce_regex_patient_id_match is False
    assert custom_config.enforce_split_stats_parity is False
    assert custom_config.fail_on_empty_cancer_mask is False
    assert custom_config.fail_on_positive_not_cancer_mask is False
