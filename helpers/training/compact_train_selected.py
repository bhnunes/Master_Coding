from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np

from helpers.training.master_manifest_queries import CanonicalRowRecord

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = "index.sqlite"
SUMMARY_FILE_NAME = "summary.json"
MARKER_FILE_NAME = ".train_selected_compact_artifact"
SHARDS_DIR_NAME = "shards"
INDEX_SCHEMA_VERSION = 1
COPY_BATCH_ROWS = 512
VALID_HDF5_COMPRESSIONS = frozenset({"none", "lzf", "gzip"})

CompressionName = Literal["none", "lzf", "gzip"]


@dataclass(frozen=True)
class CompactTrainSelectedBuildResult:
    compact_dir: Path
    index_path: Path
    summary_path: Path
    row_count: int
    shard_count: int
    total_size_bytes: int
    compression: str

    def to_provenance(self) -> dict[str, Any]:
        return {
            "kind": "compact_train_selected",
            "compact_dir": str(self.compact_dir),
            "index_path": str(self.index_path),
            "summary_path": str(self.summary_path),
            "row_count": self.row_count,
            "shard_count": self.shard_count,
            "total_size_bytes": self.total_size_bytes,
            "compression": self.compression,
        }


@dataclass(frozen=True)
class CompactTrainSelectedLoadResult:
    records: tuple[CanonicalRowRecord, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _SelectedDecision:
    source_hdf5_path: Path
    source_row_index: int
    filename: str
    patient_id: str
    label: int


def normalize_hdf5_compression(value: str) -> CompressionName:
    normalized = value.strip().lower()
    if normalized not in VALID_HDF5_COMPRESSIONS:
        raise ValueError(
            "SMART_SAMPLER_COMPACT_HDF5_COMPRESSION must be one of "
            f"{sorted(VALID_HDF5_COMPRESSIONS)}, got {value!r}."
        )
    return cast(CompressionName, normalized)


def build_compact_train_selected_from_decisions(
    *,
    master_manifest_path: Path,
    decisions: Sequence[Mapping[str, object]],
    compact_dir: Path,
    local_work_dir: Path | None = None,
    compression: str = "none",
) -> CompactTrainSelectedBuildResult:
    """Build compact HDF5 shards for Stage 6 selected TRAIN rows."""

    normalized_compression = normalize_hdf5_compression(compression)
    selected_decisions = _collect_selected_decisions(decisions)
    compact_dir = compact_dir.expanduser()
    local_build_dir = local_work_dir.expanduser() if local_work_dir is not None else None
    temp_dir = _prepare_temp_dir(compact_dir=compact_dir, parent=local_build_dir)
    try:
        result = _build_into_temp_dir(
            master_manifest_path=master_manifest_path,
            selected_decisions=selected_decisions,
            temp_dir=temp_dir,
            final_compact_dir=compact_dir,
            stage_source_shards_locally=local_build_dir is not None,
            compression=normalized_compression,
        )
        _publish_compact_dir(temp_dir=temp_dir, compact_dir=compact_dir)
        return CompactTrainSelectedBuildResult(
            compact_dir=compact_dir,
            index_path=compact_dir / INDEX_FILE_NAME,
            summary_path=compact_dir / SUMMARY_FILE_NAME,
            row_count=result.row_count,
            shard_count=result.shard_count,
            total_size_bytes=_directory_size_bytes(compact_dir),
            compression=result.compression,
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def remap_records_to_compact_train_selected(
    records: Sequence[CanonicalRowRecord],
    *,
    compact_dir: Path,
) -> CompactTrainSelectedLoadResult:
    """Return records remapped from canonical TRAIN rows to compact local shards."""

    compact_dir = compact_dir.expanduser()
    _validate_compact_artifact_present(compact_dir)
    index_path = compact_dir / INDEX_FILE_NAME
    mappings = _load_index_mappings(index_path, compact_dir=compact_dir)
    remapped_records: list[CanonicalRowRecord] = []
    missing_keys: list[str] = []
    used_compact_paths: set[Path] = set()

    for record in records:
        key = (_path_key(record.source_hdf5_path), record.source_row_index)
        mapping = mappings.get(key)
        if mapping is None:
            missing_keys.append(f"{record.source_hdf5_path}::{record.source_row_index}")
            continue
        compact_path, compact_row_index = mapping
        used_compact_paths.add(compact_path)
        remapped_records.append(
            replace(
                record,
                source_hdf5_path=compact_path,
                source_row_index=compact_row_index,
            )
        )

    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        raise FileNotFoundError(
            "Compact TRAIN_SELECTED artifact is missing mappings for requested smart-sampled "
            "TRAIN rows. Rerun Stage 6 with SMART_SAMPLER_BUILD_COMPACT_TRAIN_SELECTED=True. "
            f"First missing rows: {preview}"
        )

    missing_files = sorted(str(path) for path in used_compact_paths if not path.is_file())
    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            "Compact TRAIN_SELECTED artifact index points to missing HDF5 shards. Rerun Stage 6 "
            "with SMART_SAMPLER_BUILD_COMPACT_TRAIN_SELECTED=True. Missing files: "
            f"{preview}"
        )

    summary = _read_summary(compact_dir / SUMMARY_FILE_NAME)
    provenance = {
        "kind": "compact_train_selected",
        "compact_dir": str(compact_dir),
        "index_path": str(index_path),
        "summary_path": str(compact_dir / SUMMARY_FILE_NAME),
        "requested_row_count": len(records),
        "mapped_row_count": len(remapped_records),
        "row_count": summary.get("row_count"),
        "shard_count": summary.get("shard_count"),
        "total_size_bytes": summary.get("total_size_bytes"),
        "compression": summary.get("compression"),
        "created_at": summary.get("created_at"),
    }
    return CompactTrainSelectedLoadResult(
        records=tuple(remapped_records),
        provenance=provenance,
    )


def _collect_selected_decisions(
    decisions: Sequence[Mapping[str, object]],
) -> list[_SelectedDecision]:
    selected: list[_SelectedDecision] = []
    for decision in decisions:
        if not bool(decision["is_stage7_selected"]):
            continue
        selected.append(
            _SelectedDecision(
                source_hdf5_path=Path(str(decision["source_hdf5_path"])),
                source_row_index=int(str(decision["source_row_index"])),
                filename=str(decision["filename"]),
                patient_id=str(decision["patient_id"]),
                label=int(str(decision["label"])),
            )
        )
    return selected


def _build_into_temp_dir(
    *,
    master_manifest_path: Path,
    selected_decisions: Sequence[_SelectedDecision],
    temp_dir: Path,
    final_compact_dir: Path,
    stage_source_shards_locally: bool,
    compression: CompressionName,
) -> CompactTrainSelectedBuildResult:
    temp_dir.mkdir(parents=True, exist_ok=False)
    shards_dir = temp_dir / SHARDS_DIR_NAME
    shards_dir.mkdir()
    grouped = _group_decisions_by_source(selected_decisions)
    mapping_rows: list[tuple[str, int, str, int, str, str, int]] = []
    shard_summaries: list[dict[str, Any]] = []

    for shard_number, (source_path, shard_decisions) in enumerate(grouped.items(), start=1):
        compact_relative_path = Path(SHARDS_DIR_NAME) / _compact_shard_name(
            source_path,
            shard_number,
        )
        compact_path = temp_dir / compact_relative_path
        read_source_path = (
            _stage_source_shard_for_compaction(
                source_path=source_path,
                temp_dir=temp_dir,
                shard_number=shard_number,
            )
            if stage_source_shards_locally
            else source_path
        )
        try:
            written_rows = _write_compact_shard(
                source_path=read_source_path,
                canonical_source_path=source_path,
                selected_decisions=shard_decisions,
                compact_path=compact_path,
                compression=compression,
            )
        finally:
            if read_source_path != source_path:
                read_source_path.unlink(missing_ok=True)
        compact_row_by_source_index = {
            decision.source_row_index: compact_index
            for compact_index, decision in enumerate(shard_decisions)
        }
        for decision in shard_decisions:
            mapping_rows.append(
                (
                    _path_key(decision.source_hdf5_path),
                    decision.source_row_index,
                    compact_relative_path.as_posix(),
                    compact_row_by_source_index[decision.source_row_index],
                    decision.filename,
                    decision.patient_id,
                    decision.label,
                )
            )
        shard_summaries.append(
            {
                "source_hdf5_path": str(source_path),
                "compact_hdf5_path": compact_relative_path.as_posix(),
                "row_count": written_rows,
                "size_bytes": compact_path.stat().st_size,
            }
        )

    source_staging_dir = temp_dir / ".source_staging"
    if source_staging_dir.exists():
        shutil.rmtree(source_staging_dir)

    index_path = temp_dir / INDEX_FILE_NAME
    _write_index(index_path, mapping_rows=mapping_rows)
    row_count = len(mapping_rows)
    summary_payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "master_manifest_path": str(master_manifest_path),
        "compact_dir": str(final_compact_dir),
        "row_count": row_count,
        "shard_count": len(grouped),
        "total_size_bytes": 0,
        "compression": compression,
        "shards": shard_summaries,
    }
    summary_path = temp_dir / SUMMARY_FILE_NAME
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    _validate_temp_artifact(temp_dir, expected_row_count=row_count)
    (temp_dir / MARKER_FILE_NAME).write_text(
        json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "artifact": "train_selected_compact",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_payload["total_size_bytes"] = _directory_size_bytes(temp_dir)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    return CompactTrainSelectedBuildResult(
        compact_dir=temp_dir,
        index_path=index_path,
        summary_path=summary_path,
        row_count=row_count,
        shard_count=len(grouped),
        total_size_bytes=_directory_size_bytes(temp_dir),
        compression=compression,
    )


def _group_decisions_by_source(
    selected_decisions: Sequence[_SelectedDecision],
) -> dict[Path, list[_SelectedDecision]]:
    grouped: dict[Path, list[_SelectedDecision]] = {}
    for decision in selected_decisions:
        grouped.setdefault(decision.source_hdf5_path, []).append(decision)
    return {
        source_path: sorted(rows, key=lambda row: row.source_row_index)
        for source_path, rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def _write_compact_shard(
    *,
    source_path: Path,
    canonical_source_path: Path,
    selected_decisions: Sequence[_SelectedDecision],
    compact_path: Path,
    compression: CompressionName,
) -> int:
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5_compression = None if compression == "none" else compression
    with h5py.File(source_path, "r") as source, h5py.File(compact_path, "w") as target:
        source_images = source["images"]
        source_masks = source["masks"]
        source_labels = source["labels"]
        source_patient_ids = source["patient_ids"]
        source_filenames = _get_filenames_dataset(source)
        row_count = len(selected_decisions)
        target_images = target.create_dataset(
            "images",
            shape=(row_count, *source_images.shape[1:]),
            dtype=source_images.dtype,
            compression=hdf5_compression,
        )
        target_masks = target.create_dataset(
            "masks",
            shape=(row_count, *source_masks.shape[1:]),
            dtype=source_masks.dtype,
            compression=hdf5_compression,
        )
        target_labels = target.create_dataset(
            "labels",
            shape=(row_count,),
            dtype=source_labels.dtype,
            compression=hdf5_compression,
        )
        target_patient_ids = target.create_dataset(
            "patient_ids",
            shape=(row_count,),
            dtype=source_patient_ids.dtype,
            compression=hdf5_compression,
        )
        target_filenames = target.create_dataset(
            "filenames",
            shape=(row_count,),
            dtype=source_filenames.dtype,
            compression=hdf5_compression,
        )
        for attr_name, attr_value in source.attrs.items():
            target.attrs[attr_name] = attr_value
        target.attrs["compact_train_selected"] = True
        target.attrs["source_hdf5_path"] = str(canonical_source_path)

        output_offset = 0
        for start, stop, decisions_slice in _decision_batches(selected_decisions):
            source_slice = slice(start, stop)
            batch_size = stop - start
            output_slice = slice(output_offset, output_offset + batch_size)
            _validate_source_rows(
                source_labels=source_labels,
                source_patient_ids=source_patient_ids,
                source_filenames=source_filenames,
                decisions=decisions_slice,
            )
            target_images[output_slice] = source_images[source_slice]
            target_masks[output_slice] = source_masks[source_slice]
            target_labels[output_slice] = source_labels[source_slice]
            target_patient_ids[output_slice] = source_patient_ids[source_slice]
            target_filenames[output_slice] = source_filenames[source_slice]
            output_offset += batch_size
    return len(selected_decisions)


def _decision_batches(
    decisions: Sequence[_SelectedDecision],
) -> list[tuple[int, int, Sequence[_SelectedDecision]]]:
    batches: list[tuple[int, int, Sequence[_SelectedDecision]]] = []
    if not decisions:
        return batches
    run_start = decisions[0].source_row_index
    run_rows: list[_SelectedDecision] = [decisions[0]]
    previous_index = decisions[0].source_row_index
    for decision in decisions[1:]:
        if decision.source_row_index == previous_index + 1 and len(run_rows) < COPY_BATCH_ROWS:
            run_rows.append(decision)
            previous_index = decision.source_row_index
            continue
        batches.append((run_start, previous_index + 1, tuple(run_rows)))
        run_start = decision.source_row_index
        run_rows = [decision]
        previous_index = decision.source_row_index
    batches.append((run_start, previous_index + 1, tuple(run_rows)))
    return batches


def _validate_source_rows(
    *,
    source_labels: Any,
    source_patient_ids: Any,
    source_filenames: Any,
    decisions: Sequence[_SelectedDecision],
) -> None:
    indices = [decision.source_row_index for decision in decisions]
    labels = source_labels[indices]
    patient_ids = source_patient_ids[indices]
    filenames = source_filenames[indices]
    for offset, decision in enumerate(decisions):
        observed_label = int(labels[offset])
        observed_patient_id = _decode_hdf5_scalar(patient_ids[offset])
        observed_filename = _decode_hdf5_scalar(filenames[offset])
        if observed_label != decision.label:
            raise ValueError(
                "Compact TRAIN_SELECTED label parity failed for "
                f"{decision.source_hdf5_path} row {decision.source_row_index}: "
                f"manifest={decision.label}, hdf5={observed_label}"
            )
        if observed_patient_id != decision.patient_id:
            raise ValueError(
                "Compact TRAIN_SELECTED patient parity failed for "
                f"{decision.source_hdf5_path} row {decision.source_row_index}: "
                f"manifest={decision.patient_id}, hdf5={observed_patient_id}"
            )
        if observed_filename != decision.filename:
            raise ValueError(
                "Compact TRAIN_SELECTED filename parity failed for "
                f"{decision.source_hdf5_path} row {decision.source_row_index}: "
                f"manifest={decision.filename}, hdf5={observed_filename}"
            )


def _write_index(
    index_path: Path,
    *,
    mapping_rows: Sequence[tuple[str, int, str, int, str, str, int]],
) -> None:
    with closing(sqlite3.connect(index_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE row_mapping (
                original_source_hdf5_path TEXT NOT NULL,
                original_source_row_index INTEGER NOT NULL,
                compact_hdf5_path TEXT NOT NULL,
                compact_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                label INTEGER NOT NULL,
                PRIMARY KEY (original_source_hdf5_path, original_source_row_index)
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", str(INDEX_SCHEMA_VERSION)),
                ("artifact", "train_selected_compact"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO row_mapping (
                original_source_hdf5_path,
                original_source_row_index,
                compact_hdf5_path,
                compact_row_index,
                filename,
                patient_id,
                label
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            mapping_rows,
        )
        connection.commit()


def _load_index_mappings(
    index_path: Path,
    *,
    compact_dir: Path,
) -> dict[tuple[str, int], tuple[Path, int]]:
    with closing(sqlite3.connect(index_path)) as connection:
        connection.row_factory = sqlite3.Row
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM metadata").fetchall()
        }
        if metadata.get("artifact") != "train_selected_compact":
            raise ValueError(f"Invalid compact TRAIN_SELECTED index artifact: {index_path}")
        if int(metadata.get("schema_version", "0")) != INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported compact TRAIN_SELECTED index schema: {index_path}")
        rows = connection.execute(
            """
            SELECT original_source_hdf5_path, original_source_row_index,
                   compact_hdf5_path, compact_row_index
            FROM row_mapping
            """
        ).fetchall()
    mappings: dict[tuple[str, int], tuple[Path, int]] = {}
    for row in rows:
        mappings[(str(row["original_source_hdf5_path"]), int(row["original_source_row_index"]))] = (
            compact_dir / Path(str(row["compact_hdf5_path"])),
            int(row["compact_row_index"]),
        )
    return mappings


def _validate_temp_artifact(temp_dir: Path, *, expected_row_count: int) -> None:
    index_path = temp_dir / INDEX_FILE_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Compact TRAIN_SELECTED index was not created: {index_path}")
    with closing(sqlite3.connect(index_path)) as connection:
        observed_row_count = int(
            connection.execute("SELECT COUNT(*) FROM row_mapping").fetchone()[0]
        )
    if observed_row_count != expected_row_count:
        raise RuntimeError(
            "Compact TRAIN_SELECTED index row count mismatch: "
            f"expected={expected_row_count}, observed={observed_row_count}"
        )


def _validate_compact_artifact_present(compact_dir: Path) -> None:
    missing_paths = [
        path
        for path in (
            compact_dir,
            compact_dir / MARKER_FILE_NAME,
            compact_dir / INDEX_FILE_NAME,
            compact_dir / SUMMARY_FILE_NAME,
        )
        if not path.exists()
    ]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Compact TRAIN_SELECTED artifact is missing or incomplete. Rerun Stage 6 with "
            "SMART_SAMPLER_BUILD_COMPACT_TRAIN_SELECTED=True. Missing: "
            f"{preview}"
        )


def _prepare_temp_dir(*, compact_dir: Path, parent: Path | None = None) -> Path:
    parent = compact_dir.parent if parent is None else parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f".{compact_dir.name}.{uuid.uuid4().hex}.tmp"


def _publish_compact_dir(*, temp_dir: Path, compact_dir: Path) -> None:
    if temp_dir.parent.resolve(strict=False) == compact_dir.parent.resolve(strict=False):
        _replace_compact_dir(temp_dir=temp_dir, compact_dir=compact_dir)
        return

    publish_temp_dir = _prepare_temp_dir(compact_dir=compact_dir)
    try:
        shutil.copytree(temp_dir, publish_temp_dir)
        _validate_compact_artifact_present(publish_temp_dir)
        _replace_compact_dir(temp_dir=publish_temp_dir, compact_dir=compact_dir)
    except Exception:
        if publish_temp_dir.exists():
            shutil.rmtree(publish_temp_dir)
        raise


def _replace_compact_dir(*, temp_dir: Path, compact_dir: Path) -> None:
    if compact_dir.exists():
        marker_path = compact_dir / MARKER_FILE_NAME
        if not marker_path.is_file():
            raise FileExistsError(
                "Refusing to replace TRAIN_SELECTED_COMPACT_DIR because it does not contain "
                f"the expected compact-artifact marker: {compact_dir}"
            )
        shutil.rmtree(compact_dir)
    temp_dir.replace(compact_dir)


def _stage_source_shard_for_compaction(
    *,
    source_path: Path,
    temp_dir: Path,
    shard_number: int,
) -> Path:
    staging_dir = temp_dir / ".source_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / f"{shard_number:05d}_{source_path.name}"
    temp_path = staging_dir / f".{staged_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source_path, temp_path)
        temp_path.replace(staged_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        staged_path.unlink(missing_ok=True)
        raise
    return staged_path


def _read_summary(summary_path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))


def _get_filenames_dataset(handle: h5py.File) -> Any:
    for dataset_name in ("filenames", "filename"):
        if dataset_name in handle:
            return handle[dataset_name]
    available = ", ".join(handle.keys())
    raise KeyError(
        "HDF5 file is missing the filename dataset. Expected one of "
        f"('filenames', 'filename'). Available: {available}"
    )


def _decode_hdf5_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _compact_shard_name(source_path: Path, shard_number: int) -> str:
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in source_path.stem
    )
    return f"{shard_number:05d}_{safe_stem}.h5"


def _path_key(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _directory_size_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
