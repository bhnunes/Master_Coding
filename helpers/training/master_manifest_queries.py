from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from helpers.extraction.manifest_paths import resolve_manifest_path_ref


@dataclass(frozen=True)
class CanonicalRowRecord:
    source_hdf5_path: Path
    source_row_index: int
    patient_id: str
    label: int
    filename: str
    split: str
    normalization_method: str | None
    normalization_artifact_id: int | None
    sampling_decision: str | None
    is_stage7_selected: bool
    stage4_split_bundle_id: int | None = None


def load_lr_finder_training_records(
    master_manifest_path: Path,
    *,
    smart_sampling: bool,
) -> list[CanonicalRowRecord]:
    """Load Stage 8 training rows from SQLite once at startup."""

    return load_training_records(master_manifest_path, smart_sampling=smart_sampling)


def load_training_records(
    master_manifest_path: Path,
    *,
    smart_sampling: bool,
) -> list[CanonicalRowRecord]:
    """Load Stage 9 training rows from SQLite once at startup."""

    return _load_split_records(
        master_manifest_path,
        split="TRAIN",
        require_stage7_selected=smart_sampling,
    )


def load_validation_records(master_manifest_path: Path) -> list[CanonicalRowRecord]:
    """Load validation rows for Stages 8-10 from SQLite once at startup."""

    return _load_split_records(master_manifest_path, split="VALIDATION")


def load_test_records(master_manifest_path: Path) -> list[CanonicalRowRecord]:
    """Load Stage 11 test rows from SQLite once at startup."""

    return _load_split_records(master_manifest_path, split="TEST")


def _load_split_records(
    master_manifest_path: Path,
    *,
    split: str,
    require_stage7_selected: bool = False,
) -> list[CanonicalRowRecord]:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.row_factory = sqlite3.Row
        patch_stage_state_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(patch_stage_state)").fetchall()
        }
        stage4_split_bundle_select = (
            "s.stage4_split_bundle_id"
            if "stage4_split_bundle_id" in patch_stage_state_columns
            else "NULL AS stage4_split_bundle_id"
        )
        query = f"""
        SELECT
            p.source_hdf5_path,
            p.source_row_index,
            p.patient_id,
            p.label,
            p.filename,
            s.split,
            {stage4_split_bundle_select},
            s.normalization_method,
            s.normalization_artifact_id,
            s.sampling_decision,
            s.is_stage7_selected
        FROM patches p
        INNER JOIN patch_stage_state s ON s.patch_id = p.patch_id
        WHERE s.is_stage4_accepted = 1
          AND s.split = ?
    """
        parameters: list[object] = [split]
        if require_stage7_selected:
            query += " AND s.is_stage7_selected = 1"
        query += " ORDER BY p.patient_id ASC, p.source_hdf5_path ASC, p.source_row_index ASC"
        rows = connection.execute(query, parameters).fetchall()

    return [
        CanonicalRowRecord(
            source_hdf5_path=resolve_manifest_path_ref(
                str(row["source_hdf5_path"]),
                source_root=master_manifest_path.parent,
                manifest_path=master_manifest_path,
            ),
            source_row_index=int(row["source_row_index"]),
            patient_id=str(row["patient_id"]),
            label=int(row["label"]),
            filename=str(row["filename"]),
            split=str(row["split"]),
            normalization_method=(
                str(row["normalization_method"])
                if row["normalization_method"] is not None
                else None
            ),
            normalization_artifact_id=(
                int(row["normalization_artifact_id"])
                if row["normalization_artifact_id"] is not None
                else None
            ),
            sampling_decision=(
                str(row["sampling_decision"]) if row["sampling_decision"] is not None else None
            ),
            is_stage7_selected=bool(row["is_stage7_selected"]),
            stage4_split_bundle_id=(
                int(row["stage4_split_bundle_id"])
                if row["stage4_split_bundle_id"] is not None
                else None
            ),
        )
        for row in rows
    ]
