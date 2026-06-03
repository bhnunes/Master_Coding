from __future__ import annotations

from pathlib import Path

import pytest

from helpers.extraction.manifest_paths import (
    build_hdf5_dataset_ref,
    manifest_dir_from_path,
    parse_hdf5_dataset_ref,
    resolve_manifest_path_ref,
    to_manifest_path_ref,
    to_source_path_ref,
)


def test_to_source_path_ref_round_trips_through_resolution(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    runtime_path = source_root / "IMAGES" / "patient_001" / "slide_a.svs"
    manifest_path = tmp_path / "artifacts" / "master_manifest.sqlite"

    stored = to_source_path_ref(runtime_path, source_root=source_root)

    assert stored == "SOURCE::IMAGES/patient_001/slide_a.svs"
    assert resolve_manifest_path_ref(
        stored,
        source_root=source_root,
        manifest_path=manifest_path,
    ) == runtime_path


def test_source_path_ref_resolves_under_different_source_roots(tmp_path: Path) -> None:
    original_source_root = tmp_path / "local_source"
    relocated_source_root = tmp_path / "colab_source"
    manifest_path = tmp_path / "artifacts" / "master_manifest.sqlite"
    stored = to_source_path_ref(
        original_source_root / "IMAGES" / "patient_001" / "slide_a.svs",
        source_root=original_source_root,
    )

    assert stored == "SOURCE::IMAGES/patient_001/slide_a.svs"
    assert resolve_manifest_path_ref(
        stored,
        source_root=relocated_source_root,
        manifest_path=manifest_path,
    ) == (relocated_source_root / "IMAGES" / "patient_001" / "slide_a.svs")


def test_to_manifest_path_ref_round_trips_through_resolution(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "master_manifest.sqlite"
    runtime_path = manifest_path.parent / "stage2_shards" / "patient_001.h5"

    stored = to_manifest_path_ref(runtime_path, manifest_path=manifest_path)

    assert stored == "MANIFEST::stage2_shards/patient_001.h5"
    assert resolve_manifest_path_ref(
        stored,
        source_root=tmp_path / "source",
        manifest_path=manifest_path,
    ) == runtime_path


def test_manifest_dir_from_path_returns_parent_directory(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run" / "master_manifest.sqlite"

    assert manifest_dir_from_path(manifest_path) == tmp_path / "run"


def test_to_source_path_ref_rejects_paths_outside_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    runtime_path = tmp_path / "other" / "slide_a.svs"

    with pytest.raises(ValueError, match="must be under root"):
        to_source_path_ref(runtime_path, source_root=source_root)


def test_to_manifest_path_ref_rejects_paths_outside_manifest_dir(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "master_manifest.sqlite"
    runtime_path = tmp_path / "other" / "normalization_stats.json"

    with pytest.raises(ValueError, match="must be under root"):
        to_manifest_path_ref(runtime_path, manifest_path=manifest_path)


@pytest.mark.parametrize(
    "stored_value",
    [
        "SOURCE::/absolute/path.svs",
        "SOURCE::../escape/slide.svs",
        r"SOURCE::IMAGES\\slide.svs",
        "MANIFEST::C:/temp/file.json",
        "UNKNOWN::value",
    ],
)
def test_resolve_manifest_path_ref_rejects_invalid_values(
    tmp_path: Path,
    stored_value: str,
) -> None:
    with pytest.raises(ValueError):
        resolve_manifest_path_ref(
            stored_value,
            source_root=tmp_path / "source",
            manifest_path=tmp_path / "artifacts" / "master_manifest.sqlite",
        )


def test_build_hdf5_dataset_ref_round_trips_with_parser() -> None:
    stored = build_hdf5_dataset_ref("images", 42)

    assert stored == "HDF5::images[42]"
    assert parse_hdf5_dataset_ref(stored) == ("images", 42)


@pytest.mark.parametrize(
    ("dataset_name", "row_index"),
    [("", 0), ("images[0]", 0), ("masks]", 0), ("images", -1)],
)
def test_build_hdf5_dataset_ref_rejects_invalid_inputs(
    dataset_name: str,
    row_index: int,
) -> None:
    with pytest.raises(ValueError):
        build_hdf5_dataset_ref(dataset_name, row_index)


@pytest.mark.parametrize(
    "stored_value",
    ["HDF5::images", "HDF5::images[-1]", "HDF5::images[a]", "images[1]"],
)
def test_parse_hdf5_dataset_ref_rejects_invalid_values(stored_value: str) -> None:
    with pytest.raises(ValueError):
        parse_hdf5_dataset_ref(stored_value)
