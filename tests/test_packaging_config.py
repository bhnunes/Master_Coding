from pathlib import Path

import pytest

from helpers.packaging.config import load_packaging_config


def test_load_packaging_config_reads_defaults(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    source_path.write_bytes(b"placeholder")

    config = load_packaging_config({"PACKAGING_SOURCE_HDF5_PATH": str(source_path)})

    assert config.output_dir == tmp_path
    assert config.output_filename == "SOURCE_DATASET.h5"
    assert config.overwrite_outputs is False
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
