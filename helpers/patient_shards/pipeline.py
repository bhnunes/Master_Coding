from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from helpers.patient_shards.config import PatientShardsConfig
from helpers.patient_shards.io import (
    build_sample_manifest_rows,
    build_shard_manifest_rows,
    prepare_output_shard_dir,
    verify_patient_shard_integrity,
    write_patient_shard,
    write_sample_manifest_parquet,
    write_split_manifest_parquet,
    write_split_summary_json,
)
from helpers.stage_contracts import (
    STAGE5_MANIFEST_FILE_NAME,
    stage5_singleton_split_paths,
    stage6_5_patient_shard_dir_paths,
)


@dataclass(frozen=True)
class PatientShardsRunSummary:
    output_base_dir: Path
    split_shard_counts: dict[str, int]
    total_shards: int


def _load_stage5_manifest(base_dir: Path) -> pd.DataFrame:
    manifest_path = base_dir / STAGE5_MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{STAGE5_MANIFEST_FILE_NAME} not found in base_dir: {manifest_path}"
        )
    return pd.read_csv(manifest_path)


def _patient_row_indices(source_path: Path) -> dict[int, list[int]]:
    with h5py.File(source_path, "r") as handle:
        patient_ids = handle["patient_ids"][:]
    indices_by_patient: dict[int, list[int]] = {}
    for row_index, patient_id in enumerate(patient_ids.tolist()):
        indices_by_patient.setdefault(int(patient_id), []).append(row_index)
    return indices_by_patient


def _patient_expected_metadata(source_path: Path) -> dict[int, dict[str, list[Any]]]:
    with h5py.File(source_path, "r") as handle:
        patient_ids = handle["patient_ids"][:].tolist()
        labels = handle["labels"][:].tolist()
        filenames = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["filenames"][:].tolist()
        ]

    metadata_by_patient: dict[int, dict[str, list[Any]]] = {}
    for row_index, patient_id_value in enumerate(patient_ids):
        patient_id = int(patient_id_value)
        patient_metadata = metadata_by_patient.setdefault(
            patient_id,
            {"labels": [], "filenames": [], "source_row_indices": []},
        )
        patient_metadata["labels"].append(int(labels[row_index]))
        patient_metadata["filenames"].append(filenames[row_index])
        patient_metadata["source_row_indices"].append(row_index)
    return metadata_by_patient


def _validate_stage5_split_presence(
    split_name: str,
    split_path: Path,
    manifest_df: pd.DataFrame,
) -> None:
    split_manifest_df = cast(pd.DataFrame, manifest_df.loc[manifest_df["split"] == split_name])
    if split_manifest_df.empty:
        return
    if not split_path.is_file():
        raise FileNotFoundError(
            "Stage 5 manifest references non-empty split "
            f"'{split_name}' but file is missing: {split_path}"
        )


def validate_split_shard_integrity(split_name: str, source_path: Path, shard_dir: Path) -> None:
    """Validate that emitted Stage 6.5 shards preserve the Stage 5 split contract."""

    shard_paths = sorted(shard_dir.glob("*.h5"))
    with h5py.File(source_path, "r") as source_handle:
        source_patient_ids = np.asarray(source_handle["patient_ids"][:], dtype=np.int32)
        source_labels = np.asarray(source_handle["labels"][:], dtype=np.uint8)
        source_filenames = np.asarray(source_handle["filenames"][:], dtype=object)

    source_patient_set = {int(value) for value in source_patient_ids.tolist()}
    source_row_count = int(len(source_patient_ids))
    source_class_counts = {
        0: int((source_labels == 0).sum()),
        1: int((source_labels == 1).sum()),
    }
    source_rows_per_patient = {
        patient_id: int((source_patient_ids == patient_id).sum())
        for patient_id in source_patient_set
    }

    shard_patient_set: set[int] = set()
    shard_row_count = 0
    shard_class_counts = {0: 0, 1: 0}
    shard_rows_per_patient: dict[int, int] = {}
    observed_records: list[tuple[int, int, str]] = []

    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as shard_handle:
            patient_ids = np.asarray(shard_handle["patient_ids"][:], dtype=np.int32)
            labels = np.asarray(shard_handle["labels"][:], dtype=np.uint8)
            filenames = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in shard_handle["filenames"][:].tolist()
            ]
            recorded_patient_id = int(shard_handle.attrs["patient_id"])
            recorded_split_name = str(shard_handle.attrs["split_name"])
            recorded_source_path = str(shard_handle.attrs["source_split_hdf5_path"])
            recorded_source_row_indices = cast(
                list[int],
                json.loads(str(shard_handle.attrs["source_split_row_indices_json"])),
            )

        unique_patient_ids = {int(value) for value in patient_ids.tolist()}
        if unique_patient_ids != {recorded_patient_id}:
            raise ValueError(
                f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                f"contains patient ids {sorted(unique_patient_ids)} instead of one patient."
            )
        if recorded_split_name != split_name:
            raise ValueError(
                f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                f"records split_name='{recorded_split_name}'."
            )
        if recorded_source_path != str(source_path):
            raise ValueError(
                f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                "records the wrong source split path."
            )
        if len(recorded_source_row_indices) != len(patient_ids):
            raise ValueError(
                f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                "source row lineage length does not match shard rows."
            )

        for row_in_shard, source_row_index in enumerate(recorded_source_row_indices):
            source_patient_id = int(source_patient_ids[source_row_index])
            source_label = int(source_labels[source_row_index])
            source_filename_raw = source_filenames[source_row_index]
            source_filename = (
                source_filename_raw.decode("utf-8")
                if isinstance(source_filename_raw, bytes)
                else str(source_filename_raw)
            )
            if source_patient_id != recorded_patient_id:
                raise ValueError(
                    f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                    f"references source row {source_row_index} from patient {source_patient_id}."
                )
            if (
                int(labels[row_in_shard]) != source_label
                or filenames[row_in_shard] != source_filename
            ):
                raise ValueError(
                    f"Stage 6.5 validation FAILED for {split_name}: shard '{shard_path.name}' "
                    "does not preserve label/filename alignment from the source split."
                )
            observed_records.append((recorded_patient_id, source_label, source_filename))

        shard_patient_set.add(recorded_patient_id)
        shard_row_count += len(patient_ids)
        shard_class_counts[0] += int((labels == 0).sum())
        shard_class_counts[1] += int((labels == 1).sum())
        shard_rows_per_patient[recorded_patient_id] = int(len(patient_ids))

    expected_records = [
        (
            int(patient_id),
            int(label),
            filename.decode("utf-8") if isinstance(filename, bytes) else str(filename),
        )
        for patient_id, label, filename in zip(
            source_patient_ids.tolist(),
            source_labels.tolist(),
            source_filenames.tolist(),
            strict=True,
        )
    ]
    if shard_row_count != source_row_count:
        raise ValueError(
            f"Stage 6.5 validation FAILED for {split_name}: row count {shard_row_count} "
            f"does not match source {source_row_count}."
        )
    if shard_patient_set != source_patient_set:
        raise ValueError(
            "Stage 6.5 validation FAILED for "
            f"{split_name}: patient set {sorted(shard_patient_set)} "
            f"does not match source {sorted(source_patient_set)}."
        )
    if shard_rows_per_patient != source_rows_per_patient:
        raise ValueError(
            f"Stage 6.5 validation FAILED for {split_name}: per-patient row counts do not match."
        )
    if shard_class_counts != source_class_counts:
        raise ValueError(
            f"Stage 6.5 validation FAILED for {split_name}: class counts {shard_class_counts} "
            f"do not match source {source_class_counts}."
        )
    if sorted(observed_records) != sorted(expected_records):
        raise ValueError(
            "Stage 6.5 validation FAILED for "
            f"{split_name}: emitted shard records do not match source rows."
        )


def run_patient_shards_pipeline(config: PatientShardsConfig) -> PatientShardsRunSummary:
    manifest_df = _load_stage5_manifest(config.stage5_base_dir)
    split_paths = stage5_singleton_split_paths(config.stage5_base_dir)
    output_dirs = stage6_5_patient_shard_dir_paths(config.output_base_dir)
    split_shard_counts: dict[str, int] = {}

    for output_dir in output_dirs.values():
        prepare_output_shard_dir(output_dir, overwrite=config.overwrite_output)

    for split_name, split_path in split_paths.items():
        _validate_stage5_split_presence(split_name, split_path, manifest_df)
        if not split_path.is_file():
            write_split_summary_json(output_dirs[split_name] / "summary.json", [])
            split_shard_counts[split_name] = 0
            continue

        patient_rows = _patient_row_indices(split_path)
        patient_metadata = _patient_expected_metadata(split_path)
        logging.info(
            "Stage 6.5 split start: %s | patients=%s | source=%s",
            split_name,
            len(patient_rows),
            split_path,
        )
        for patient_id, row_indices in sorted(patient_rows.items()):
            output_path = output_dirs[split_name] / f"{patient_id}.h5"
            write_patient_shard(
                source_path=split_path,
                output_path=output_path,
                split_name=split_name,
                patient_id=patient_id,
                source_row_indices=row_indices,
                hdf5_compression=config.hdf5_compression,
                copy_batch_size=config.copy_batch_size,
                overwrite=config.overwrite_output,
            )
            verify_patient_shard_integrity(
                output_path,
                patient_id=patient_id,
                expected_rows=len(row_indices),
                expected_labels=cast(list[int], patient_metadata[patient_id]["labels"]),
                expected_filenames=cast(list[str], patient_metadata[patient_id]["filenames"]),
                expected_source_row_indices=cast(
                    list[int], patient_metadata[patient_id]["source_row_indices"]
                ),
                expected_source_split_path=split_path,
            )
        summary_rows = build_shard_manifest_rows(split_name, output_dirs[split_name])
        sample_rows = build_sample_manifest_rows(split_name, output_dirs[split_name])
        write_split_manifest_parquet(output_dirs[split_name] / "manifest.parquet", summary_rows)
        write_sample_manifest_parquet(
            output_dirs[split_name] / "sample_manifest.parquet",
            sample_rows,
        )
        write_split_summary_json(output_dirs[split_name] / "summary.json", summary_rows)
        validate_split_shard_integrity(split_name, split_path, output_dirs[split_name])
        split_shard_counts[split_name] = len(summary_rows)
        logging.info(
            "Stage 6.5 split done: %s | shards=%s | output_dir=%s",
            split_name,
            split_shard_counts[split_name],
            output_dirs[split_name],
        )

    total_shards = sum(split_shard_counts.values())
    logging.info(
        "Stage 6.5 complete: total_shards=%s | output_base_dir=%s",
        total_shards,
        config.output_base_dir,
    )
    return PatientShardsRunSummary(
        output_base_dir=config.output_base_dir,
        split_shard_counts=split_shard_counts,
        total_shards=total_shards,
    )
