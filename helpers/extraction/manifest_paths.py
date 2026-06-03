from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

SOURCE_PATH_PREFIX = "SOURCE::"
MANIFEST_PATH_PREFIX = "MANIFEST::"
HDF5_DATASET_PREFIX = "HDF5::"

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]")
_HDF5_DATASET_REF_RE = re.compile(r"^HDF5::([^\[\]]+)\[(\d+)\]$")


def manifest_dir_from_path(manifest_path: Path) -> Path:
    """Return the runtime artifact root for one master manifest."""

    return manifest_path.parent


def to_source_path_ref(path: Path, *, source_root: Path) -> str:
    """Encode one source-rooted runtime path as a portable manifest reference."""

    return _encode_rooted_path(path, root=source_root, prefix=SOURCE_PATH_PREFIX)


def to_manifest_path_ref(path: Path, *, manifest_path: Path) -> str:
    """Encode one artifact-rooted runtime path as a portable manifest reference."""

    return _encode_rooted_path(
        path,
        root=manifest_dir_from_path(manifest_path),
        prefix=MANIFEST_PATH_PREFIX,
    )


def resolve_manifest_path_ref(
    value: str,
    *,
    source_root: Path,
    manifest_path: Path,
) -> Path:
    """Resolve one stored manifest path reference into a runtime path."""

    if value.startswith(SOURCE_PATH_PREFIX):
        relative_path = _parse_relative_payload(value, prefix=SOURCE_PATH_PREFIX)
        return source_root / relative_path
    if value.startswith(MANIFEST_PATH_PREFIX):
        relative_path = _parse_relative_payload(value, prefix=MANIFEST_PATH_PREFIX)
        return manifest_dir_from_path(manifest_path) / relative_path
    raise ValueError(
        "Unsupported manifest path reference. Legacy absolute-path manifest values are not "
        f"supported: {value!r}."
    )


def build_hdf5_dataset_ref(dataset_name: str, row_index: int) -> str:
    """Build one portable logical HDF5 dataset reference."""

    normalized_dataset_name = dataset_name.strip()
    has_invalid_brackets = any(character in normalized_dataset_name for character in "[]")
    if not normalized_dataset_name or has_invalid_brackets:
        raise ValueError(f"Invalid HDF5 dataset name: {dataset_name!r}.")
    if row_index < 0:
        raise ValueError(f"HDF5 row index must be non-negative, got {row_index}.")
    return f"{HDF5_DATASET_PREFIX}{normalized_dataset_name}[{row_index}]"


def parse_hdf5_dataset_ref(value: str) -> tuple[str, int]:
    """Parse one portable logical HDF5 dataset reference."""

    match = _HDF5_DATASET_REF_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid HDF5 dataset reference: {value!r}.")
    dataset_name, row_index = match.groups()
    return dataset_name, int(row_index)


def to_runtime_hdf5_ref(
    source_hdf5_path: Path,
    dataset_ref: str,
    *,
    expected_dataset: str,
) -> str:
    """Expand one portable logical HDF5 dataset reference into a runtime ref."""

    dataset_name, row_index = parse_hdf5_dataset_ref(dataset_ref)
    if dataset_name != expected_dataset:
        raise ValueError(
            f"Expected HDF5 dataset {expected_dataset!r}, got {dataset_name!r} in {dataset_ref!r}."
        )
    return f"{source_hdf5_path}::{dataset_name}[{row_index}]"


def _encode_rooted_path(path: Path, *, root: Path, prefix: str) -> str:
    normalized_path = path.expanduser()
    normalized_root = root.expanduser()
    try:
        relative_path = normalized_path.relative_to(normalized_root)
    except ValueError as error:
        raise ValueError(
            f"Path {normalized_path!s} must be under root {normalized_root!s}."
        ) from error
    return f"{prefix}{_to_portable_relative_path(relative_path)}"


def _parse_relative_payload(value: str, *, prefix: str) -> Path:
    payload = value[len(prefix) :]
    if not payload:
        raise ValueError(f"Manifest path reference payload is empty: {value!r}.")
    if "\\" in payload:
        raise ValueError(f"Manifest path reference must use forward slashes: {value!r}.")
    if payload.startswith("/"):
        raise ValueError(f"Manifest path reference must be relative: {value!r}.")
    if _WINDOWS_ABSOLUTE_PATH_RE.match(payload):
        raise ValueError(f"Manifest path reference must not embed an absolute path: {value!r}.")

    relative_path = PurePosixPath(payload)
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"Manifest path reference contains invalid relative traversal: {value!r}.")
    return Path(*relative_path.parts)


def _to_portable_relative_path(path: Path) -> str:
    relative_path = PurePosixPath(path.as_posix())
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"Relative path is not portable: {path!s}.")
    return relative_path.as_posix()
