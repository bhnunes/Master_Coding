import inspect
from pathlib import Path

import pytest

from helpers.packaging import config as packaging_config
from helpers.packaging.config import PackagingConfig, load_packaging_config

DEFAULT_COPY_BATCH_SIZE = 256
CUSTOM_COPY_BATCH_SIZE = 1024
SMOKE_DEFAULT_COUNT = 7
PARSE_INT_DEFAULT_ONE = 5
PARSE_INT_DEFAULT_TWO = 6
PARSE_INT_VALID_VALUE = 7


def test_packaging_config_namespace_smoke_path(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    signature = inspect.signature(packaging_config.load_packaging_config)
    config = packaging_config.load_packaging_config(
        {"PACKAGING_SOURCE_HDF5_PATH": str(source_path)}
    )

    assert signature.parameters["env"].default is None
    assert packaging_config._parse_bool(None, "FLAG", True) is True
    assert (
        packaging_config._parse_positive_int(None, "COUNT", SMOKE_DEFAULT_COUNT)
        == SMOKE_DEFAULT_COUNT
    )
    assert packaging_config._parse_hdf5_compression("none", "COMP", "gzip") is None
    assert config.source_hdf5_path == source_path
    assert config.output_dir == tmp_path


def test_parse_bool_handles_empty_and_exact_truthy_falsy_values() -> None:
    assert packaging_config._parse_bool(None, "FLAG", True) is True
    assert packaging_config._parse_bool("", "FLAG", False) is False
    assert packaging_config._parse_bool("1", "FLAG", False) is True
    assert packaging_config._parse_bool(" true ", "FLAG", False) is True
    assert packaging_config._parse_bool("yes", "FLAG", False) is True
    assert packaging_config._parse_bool("on", "FLAG", False) is True
    assert packaging_config._parse_bool("0", "FLAG", True) is False
    assert packaging_config._parse_bool("false", "FLAG", True) is False
    assert packaging_config._parse_bool("no", "FLAG", True) is False
    assert packaging_config._parse_bool("OFF", "FLAG", True) is False


def test_parse_bool_rejects_invalid_value_with_variable_name() -> None:
    with pytest.raises(
        ValueError,
        match=r"^The 'PACKAGING_OVERWRITE_OUTPUTS' environment variable must be a boolean value\.$",
    ):
        packaging_config._parse_bool("maybe", "PACKAGING_OVERWRITE_OUTPUTS", False)


def test_required_path_forwards_required_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve_env_path(
        value: str | None,
        variable_name: str,
        *,
        required: bool,
    ) -> Path:
        captured["value"] = value
        captured["variable_name"] = variable_name
        captured["required"] = required
        return Path("/tmp/source.h5")

    monkeypatch.setattr(packaging_config, "resolve_env_path", fake_resolve_env_path)

    resolved = packaging_config._required_path("/tmp/source.h5", "PACKAGING_SOURCE_HDF5_PATH")

    assert resolved == Path("/tmp/source.h5")
    assert captured == {
        "value": "/tmp/source.h5",
        "variable_name": "PACKAGING_SOURCE_HDF5_PATH",
        "required": True,
    }


def test_parse_positive_int_handles_default_valid_and_invalid_values() -> None:
    assert (
        packaging_config._parse_positive_int(None, "COUNT", PARSE_INT_DEFAULT_ONE)
        == PARSE_INT_DEFAULT_ONE
    )
    assert (
        packaging_config._parse_positive_int("", "COUNT", PARSE_INT_DEFAULT_TWO)
        == PARSE_INT_DEFAULT_TWO
    )
    assert (
        packaging_config._parse_positive_int("7", "COUNT", 1)
        == PARSE_INT_VALID_VALUE
    )
    assert packaging_config._parse_positive_int("1", "COUNT", 9) == 1

    with pytest.raises(
        ValueError,
        match=r"^The 'PACKAGING_COPY_BATCH_SIZE' environment variable must be an integer\.$",
    ):
        packaging_config._parse_positive_int("abc", "PACKAGING_COPY_BATCH_SIZE", 1)

    with pytest.raises(
        ValueError,
        match=(
            r"^The 'PACKAGING_COPY_BATCH_SIZE' environment variable must be greater than zero\."
        ),
    ):
        packaging_config._parse_positive_int("0", "PACKAGING_COPY_BATCH_SIZE", 1)


def test_parse_hdf5_compression_handles_default_supported_and_invalid_values() -> None:
    assert packaging_config._parse_hdf5_compression(None, "COMP", "gzip") == "gzip"
    assert packaging_config._parse_hdf5_compression("", "COMP", "lzf") == "lzf"
    assert packaging_config._parse_hdf5_compression(" GZIP ", "COMP", None) == "gzip"
    assert packaging_config._parse_hdf5_compression("lzf", "COMP", None) == "lzf"
    assert packaging_config._parse_hdf5_compression("none", "COMP", "gzip") is None

    with pytest.raises(
        ValueError,
        match=(
            r"^The 'PACKAGING_HDF5_COMPRESSION' environment variable must be one of: "
            r"gzip, lzf, none\.$"
        ),
    ):
        packaging_config._parse_hdf5_compression("snappy", "PACKAGING_HDF5_COMPRESSION", None)


def test_infer_default_accepted_manifest_path_only_uses_adjacent_h5_manifest(
    tmp_path: Path,
) -> None:
    h5_source = tmp_path / "SOURCE_DATASET.h5"
    other_source = tmp_path / "SOURCE_DATASET.txt"
    manifest_path = tmp_path / "accepted_manifest.csv"
    h5_source.write_bytes(b"placeholder")
    other_source.write_text("placeholder", encoding="utf-8")
    manifest_path.write_text("filename\n", encoding="utf-8")

    assert (
        packaging_config._infer_default_accepted_manifest_path(source_hdf5_path=h5_source)
        == manifest_path
    )
    assert (
        packaging_config._infer_default_accepted_manifest_path(source_hdf5_path=other_source)
        is None
    )


def test_load_packaging_config_forwards_expected_values(monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = Path("/tmp/source.h5")
    manifest_path = Path("/tmp/accepted_manifest.csv")
    output_dir = Path("/tmp/output")
    log_folder = Path("/tmp/logs")
    calls: dict[str, object] = {}

    def fake_required_path(variable_value: str | None, variable_name: str) -> Path:
        calls["required_path"] = (variable_value, variable_name)
        return source_path

    def fake_resolve_env_path(value: str | None, variable_name: str) -> Path | None:
        resolve_calls = calls.setdefault("resolve_env_path", [])
        assert isinstance(resolve_calls, list)
        resolve_calls.append((value, variable_name))
        if variable_name == "PACKAGING_ACCEPTED_MANIFEST_PATH":
            return manifest_path
        if variable_name == "PACKAGING_OUTPUT_DIR":
            return output_dir
        return None

    def fake_parse_bool(value: str | None, variable_name: str, default: bool) -> bool:
        calls["parse_bool"] = (value, variable_name, default)
        return True

    def fake_parse_hdf5_compression(
        value: str | None,
        variable_name: str,
        default: str | None,
    ) -> str | None:
        calls["parse_hdf5_compression"] = (value, variable_name, default)
        return "gzip"

    def fake_parse_positive_int(value: str | None, variable_name: str, default: int) -> int:
        calls["parse_positive_int"] = (value, variable_name, default)
        return 1024

    def fake_resolve_log_folder(values: object, *, fallback_names: tuple[str, ...]) -> Path:
        calls["resolve_log_folder"] = (values, fallback_names)
        return log_folder

    monkeypatch.setattr(packaging_config, "_required_path", fake_required_path)
    monkeypatch.setattr(packaging_config, "resolve_env_path", fake_resolve_env_path)
    monkeypatch.setattr(packaging_config, "_parse_bool", fake_parse_bool)
    monkeypatch.setattr(packaging_config, "_parse_hdf5_compression", fake_parse_hdf5_compression)
    monkeypatch.setattr(packaging_config, "_parse_positive_int", fake_parse_positive_int)
    monkeypatch.setattr(packaging_config, "resolve_log_folder", fake_resolve_log_folder)

    env = {
        "PACKAGING_SOURCE_HDF5_PATH": "/tmp/source.h5",
        "PACKAGING_ACCEPTED_MANIFEST_PATH": "/tmp/accepted_manifest.csv",
        "PACKAGING_OUTPUT_DIR": "/tmp/output",
        "PACKAGING_OUTPUT_FILENAME": " packaged.h5 ",
        "PACKAGING_OVERWRITE_OUTPUTS": "yes",
        "PACKAGING_HDF5_COMPRESSION": "gzip",
        "PACKAGING_COPY_BATCH_SIZE": "1024",
        "PACKAGING_LOG_FILE": " custom.log ",
    }

    config = packaging_config.load_packaging_config(env)

    assert config == PackagingConfig(
        source_hdf5_path=source_path,
        output_dir=output_dir,
        output_filename="packaged.h5",
        overwrite_outputs=True,
        hdf5_compression="gzip",
        copy_batch_size=1024,
        accepted_manifest_path=manifest_path,
        log_folder=log_folder,
        log_file_name="custom.log",
    )
    assert calls["required_path"] == ("/tmp/source.h5", "PACKAGING_SOURCE_HDF5_PATH")
    assert calls["resolve_env_path"] == [
        ("/tmp/accepted_manifest.csv", "PACKAGING_ACCEPTED_MANIFEST_PATH"),
        ("/tmp/output", "PACKAGING_OUTPUT_DIR"),
    ]
    assert calls["parse_bool"] == ("yes", "PACKAGING_OVERWRITE_OUTPUTS", False)
    assert calls["parse_hdf5_compression"] == ("gzip", "PACKAGING_HDF5_COMPRESSION", None)
    assert calls["parse_positive_int"] == ("1024", "PACKAGING_COPY_BATCH_SIZE", 256)
    assert calls["resolve_log_folder"] == (env, ("PACKAGING_LOG_FOLDER",))


def test_load_packaging_config_reads_defaults(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config({"PACKAGING_SOURCE_HDF5_PATH": str(source_path)})

    assert config.output_dir == tmp_path
    assert config.output_filename == "SOURCE_DATASET.h5"
    assert config.overwrite_outputs is False
    assert config.hdf5_compression is None
    assert config.copy_batch_size == DEFAULT_COPY_BATCH_SIZE
    assert config.output_path == tmp_path / "SOURCE_DATASET.h5"
    assert config.log_path == Path("logs/packaging.log")
    assert config.log_file_name == "packaging.log"


def test_load_packaging_config_namespace_defaults_preserve_exact_default_names(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = packaging_config.load_packaging_config(
        {"PACKAGING_SOURCE_HDF5_PATH": str(source_path)}
    )

    assert config.output_filename == "SOURCE_DATASET.h5"
    assert config.output_path == tmp_path / "SOURCE_DATASET.h5"
    assert config.log_file_name == "packaging.log"
    assert config.log_path == Path("logs/packaging.log")


def test_load_packaging_config_requires_hdf5_source_path() -> None:
    with pytest.raises(ValueError, match="PACKAGING_SOURCE_HDF5_PATH"):
        load_packaging_config({})


def test_load_packaging_config_allows_hdf5_source_without_png_base_dir(tmp_path: Path) -> None:
    source_path = tmp_path / "stage2_source.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config({"PACKAGING_SOURCE_HDF5_PATH": str(source_path)})

    assert config.source_hdf5_path == source_path
    assert config.output_dir == tmp_path


def test_load_packaging_config_allows_hdf5_shard_directory(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()

    config = load_packaging_config({"PACKAGING_SOURCE_HDF5_PATH": str(shard_dir)})

    assert config.source_hdf5_path == shard_dir
    assert config.output_dir == tmp_path


def test_load_packaging_config_allows_accepted_manifest_path(tmp_path: Path) -> None:
    source_path = tmp_path / "stage2_source.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    source_path.write_bytes(b"placeholder")
    manifest_path.write_text("filename\n", encoding="utf-8")

    config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_ACCEPTED_MANIFEST_PATH": str(manifest_path),
        }
    )

    assert config.accepted_manifest_path == manifest_path


def test_load_packaging_config_auto_detects_adjacent_accepted_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    source_path.write_bytes(b"placeholder")
    manifest_path.write_text("filename\n", encoding="utf-8")

    config = load_packaging_config({"PACKAGING_SOURCE_HDF5_PATH": str(source_path)})

    assert config.accepted_manifest_path == manifest_path


def test_load_packaging_config_parses_hdf5_compression(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    gzip_config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_HDF5_COMPRESSION": "gzip",
        }
    )
    none_config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_HDF5_COMPRESSION": "none",
        }
    )

    assert gzip_config.hdf5_compression == "gzip"
    assert none_config.hdf5_compression is None


def test_load_packaging_config_rejects_invalid_hdf5_compression(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="PACKAGING_HDF5_COMPRESSION"):
        load_packaging_config(
            {
                "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
                "PACKAGING_HDF5_COMPRESSION": "snappy",
            }
        )


def test_load_packaging_config_parses_copy_batch_size(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_COPY_BATCH_SIZE": "1024",
        }
    )

    assert config.copy_batch_size == CUSTOM_COPY_BATCH_SIZE


def test_load_packaging_config_uses_default_copy_batch_size_for_empty_value(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_COPY_BATCH_SIZE": "",
        }
    )

    assert config.copy_batch_size == DEFAULT_COPY_BATCH_SIZE


def test_load_packaging_config_rejects_invalid_copy_batch_size(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="PACKAGING_COPY_BATCH_SIZE"):
        load_packaging_config(
            {
                "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
                "PACKAGING_COPY_BATCH_SIZE": "0",
            }
        )


def test_load_packaging_config_parses_overwrite_outputs_boolean(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_OVERWRITE_OUTPUTS": "true",
        }
    )

    assert config.overwrite_outputs is True


def test_load_packaging_config_rejects_invalid_overwrite_outputs_boolean(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="PACKAGING_OVERWRITE_OUTPUTS"):
        load_packaging_config(
            {
                "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
                "PACKAGING_OVERWRITE_OUTPUTS": "maybe",
            }
        )
