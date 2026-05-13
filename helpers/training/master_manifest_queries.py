from __future__ import annotations

import sqlite3
from collections.abc import Sequence
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


@dataclass(frozen=True)
class SplitRecordAvailability:
    split: str
    accepted_count: int
    stage7_selected_count: int


def load_lr_finder_training_records(
    master_manifest_path: Path,
    *,
    smart_sampling: bool,
) -> list[CanonicalRowRecord]:
    """Load LR-finder training rows from SQLite once at startup."""

    return load_training_records(master_manifest_path, smart_sampling=smart_sampling)


def load_training_records(
    master_manifest_path: Path,
    *,
    smart_sampling: bool,
) -> list[CanonicalRowRecord]:
    """Load canonical training rows from SQLite once at startup."""

    return _load_split_records(
        master_manifest_path,
        split="TRAIN",
        require_stage7_selected=smart_sampling,
    )


def load_validation_records(master_manifest_path: Path) -> list[CanonicalRowRecord]:
    """Load canonical validation rows from SQLite once at startup."""

    return _load_split_records(master_manifest_path, split="VALIDATION")


def load_test_records(master_manifest_path: Path) -> list[CanonicalRowRecord]:
    """Load canonical test rows from SQLite once at startup."""

    return _load_split_records(master_manifest_path, split="TEST")


def summarize_split_record_availability(
    master_manifest_path: Path,
) -> tuple[SplitRecordAvailability, ...]:
    """Count Stage 4-accepted rows and Stage 6-selected rows by split."""

    with sqlite3.connect(master_manifest_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                s.split AS split,
                COUNT(*) AS accepted_count,
                COALESCE(
                    SUM(CASE WHEN s.is_stage7_selected = 1 THEN 1 ELSE 0 END),
                    0
                ) AS stage7_selected_count
            FROM patches p
            INNER JOIN patch_stage_state s ON s.patch_id = p.patch_id
            WHERE s.is_stage4_accepted = 1
            GROUP BY s.split
            ORDER BY s.split
            """
        ).fetchall()

    return tuple(
        SplitRecordAvailability(
            split=str(row["split"]) if row["split"] is not None else "<NULL>",
            accepted_count=int(row["accepted_count"]),
            stage7_selected_count=int(row["stage7_selected_count"]),
        )
        for row in rows
    )


def format_split_record_availability(
    availability: Sequence[SplitRecordAvailability],
) -> str:
    """Format split availability counts for configuration errors."""

    if not availability:
        return "no Stage 4-accepted rows"
    return "; ".join(
        (
            f"{item.split}: accepted={item.accepted_count}, "
            f"stage7_selected={item.stage7_selected_count}"
        )
        for item in availability
    )


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
