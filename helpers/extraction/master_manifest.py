from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, SupportsInt, cast

import h5py

from helpers.provenance import hash_file_sha256

STAGE2_STAGE_NAME = "STAGE2"
STAGE4_CLEANING_STAGE_NAME = "STAGE4_3"
STAGE5_STAGE_NAME = "STAGE5"
STAGE7_STAGE_NAME = "STAGE7_2"


@dataclass(frozen=True)
class Stage2SlideRows:
    """Canonical Stage 2 slide payload written into the master manifest."""

    source_hdf5_path: Path
    records: Sequence[Mapping[str, object]]
    source_slide_path: Path
    annotation_path: Path | None
    artifacts_geojson_path: Path | None
    stage2_case_record_id: int
    stage2_processing_signature: str | None
    stage2_status: str


def _logical_hdf5_ref(output_path: Path, dataset_name: str, row_index: int) -> str:
    return f"{output_path}::{dataset_name}[{row_index}]"


def _coerce_int(value: object, *, field_name: str) -> int:
    try:
        return int(cast(SupportsInt | str | bytes | bytearray, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage 2 master manifest requires an integer {field_name}.") from error


def _coerce_float(value: object, *, field_name: str) -> float:
    try:
        return float(cast(SupportsFloat | str | bytes | bytearray, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage 2 master manifest requires a float {field_name}.") from error


def _coerce_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field_name="contamination_rate")


def _coerce_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


class MasterManifest:
    """SQLite-backed master manifest owned by Stage 2."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create the master-manifest schema when needed."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patches (
                    patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_hdf5_path TEXT NOT NULL,
                    source_row_index INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    patient_id INTEGER NOT NULL,
                    label INTEGER NOT NULL,
                    slide_id TEXT,
                    source_signature TEXT,
                    source_image_path TEXT NOT NULL,
                    source_mask_path TEXT NOT NULL,
                    source_slide_path TEXT NOT NULL,
                    annotation_path TEXT,
                    artifacts_geojson_path TEXT,
                    stage2_case_record_id INTEGER NOT NULL,
                    stage2_processing_signature TEXT,
                    stage2_status TEXT NOT NULL,
                    cov_fold REAL NOT NULL DEFAULT 0.0,
                    cov_penmarking REAL NOT NULL DEFAULT 0.0,
                    cov_oof REAL NOT NULL DEFAULT 0.0,
                    cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                    cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source_hdf5_path, source_row_index)
                );

                CREATE TABLE IF NOT EXISTS patch_stage_state (
                    patch_id INTEGER PRIMARY KEY,
                    cleaning_decision TEXT,
                    contamination_rate REAL,
                    split TEXT,
                    normalization_method TEXT,
                    normalization_artifact_id INTEGER,
                    sampling_decision TEXT,
                    is_stage4_accepted INTEGER,
                    is_stage7_selected INTEGER,
                    last_updated_stage_name TEXT,
                    last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patch_id) REFERENCES patches(patch_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage_name TEXT NOT NULL,
                    config_path TEXT,
                    config_sha256 TEXT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    code_version TEXT,
                    input_summary_json_path TEXT,
                    input_summary_sha256 TEXT
                );

                CREATE TABLE IF NOT EXISTS normalization_artifacts (
                    normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    state_path TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    template_path TEXT,
                    template_sha256 TEXT,
                    fit_scope TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_patches_patient_id ON patches(patient_id);
                CREATE INDEX IF NOT EXISTS idx_patches_filename ON patches(filename);
                CREATE INDEX IF NOT EXISTS idx_patch_stage_state_cleaning_decision
                    ON patch_stage_state(cleaning_decision);
                CREATE INDEX IF NOT EXISTS idx_patch_stage_state_split
                    ON patch_stage_state(split);
                CREATE INDEX IF NOT EXISTS idx_patch_stage_state_sampling_decision
                    ON patch_stage_state(sampling_decision);
                """
            )
            connection.commit()

    def replace_stage2_slide_rows(self, payload: Stage2SlideRows) -> None:
        """Replace one slide's Stage 2 rows with the canonical HDF5-backed identities."""

        self.initialize()
        source_signature = (
            self._read_source_signature(payload.source_hdf5_path) if payload.records else None
        )

        with self._connect() as connection:
            existing_patch_ids = [
                int(row["patch_id"])
                for row in connection.execute(
                    "SELECT patch_id FROM patches WHERE source_hdf5_path = ? "
                    "ORDER BY source_row_index ASC",
                    (str(payload.source_hdf5_path),),
                ).fetchall()
            ]
            if existing_patch_ids:
                connection.executemany(
                    "DELETE FROM patch_stage_state WHERE patch_id = ?",
                    [(patch_id,) for patch_id in existing_patch_ids],
                )
            connection.execute(
                "DELETE FROM patches WHERE source_hdf5_path = ?",
                (str(payload.source_hdf5_path),),
            )

            for source_row_index, record in enumerate(payload.records):
                artifact_coverages = self._artifact_coverages(record)
                cursor = connection.execute(
                    """
                    INSERT INTO patches (
                        source_hdf5_path,
                        source_row_index,
                        filename,
                        patient_id,
                        label,
                        slide_id,
                        source_signature,
                        source_image_path,
                        source_mask_path,
                        source_slide_path,
                        annotation_path,
                        artifacts_geojson_path,
                        stage2_case_record_id,
                        stage2_processing_signature,
                        stage2_status,
                        cov_fold,
                        cov_penmarking,
                        cov_oof,
                        cov_darkspot_foreign,
                        cov_edge_airbubble
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(payload.source_hdf5_path),
                        source_row_index,
                        str(record["filename"]),
                        _coerce_int(record.get("patient_id"), field_name="patient_id"),
                        _coerce_int(record.get("label"), field_name="label"),
                        str(record.get("slide_id") or payload.source_hdf5_path.stem),
                        source_signature,
                        _logical_hdf5_ref(payload.source_hdf5_path, "images", source_row_index),
                        _logical_hdf5_ref(payload.source_hdf5_path, "masks", source_row_index),
                        str(payload.source_slide_path),
                        (
                            str(payload.annotation_path)
                            if payload.annotation_path is not None
                            else None
                        ),
                        (
                            str(payload.artifacts_geojson_path)
                            if payload.artifacts_geojson_path is not None
                            else None
                        ),
                        payload.stage2_case_record_id,
                        payload.stage2_processing_signature,
                        payload.stage2_status,
                        artifact_coverages["cov_fold"],
                        artifact_coverages["cov_penmarking"],
                        artifact_coverages["cov_oof"],
                        artifact_coverages["cov_darkspot_foreign"],
                        artifact_coverages["cov_edge_airbubble"],
                    ),
                )
                if cursor.lastrowid is None:
                    raise ValueError("Stage 2 master manifest insert did not return a patch_id.")
                patch_id = int(cursor.lastrowid)
                connection.execute(
                    """
                    INSERT INTO patch_stage_state (
                        patch_id,
                        last_updated_stage_name
                    ) VALUES (?, ?)
                    """,
                    (patch_id, STAGE2_STAGE_NAME),
                )
            connection.commit()

    def update_stage4_cleaning_decisions(
        self,
        *,
        decisions: Sequence[Mapping[str, object]],
    ) -> None:
        """Persist Stage 4.3 accepted/rejected row state onto existing patch rows."""

        self.initialize()
        with self._connect() as connection:
            for decision in decisions:
                source_hdf5_path = str(decision["source_hdf5_path"])
                source_row_index = _coerce_int(
                    decision.get("source_row_index"),
                    field_name="source_row_index",
                )
                patch_row = connection.execute(
                    "SELECT patch_id, filename, patient_id, slide_id, source_signature "
                    "FROM patches WHERE source_hdf5_path = ? AND source_row_index = ?",
                    (source_hdf5_path, source_row_index),
                ).fetchone()
                if patch_row is None:
                    raise ValueError(
                        "Stage 4.3 decision targets a missing canonical Stage 2 row: "
                        f"({source_hdf5_path}, {source_row_index})"
                    )
                self._validate_stage4_decision_provenance(patch_row=patch_row, decision=decision)
                decision_text = str(decision["decision"])
                connection.execute(
                    """
                    UPDATE patch_stage_state
                    SET cleaning_decision = ?,
                        contamination_rate = ?,
                        is_stage4_accepted = ?,
                        last_updated_stage_name = ?,
                        last_updated_at = CURRENT_TIMESTAMP
                    WHERE patch_id = ?
                    """,
                    (
                        decision_text,
                        _coerce_float_or_none(decision.get("contamination_rate")),
                        1 if decision_text.lower() == "accepted" else 0,
                        STAGE4_CLEANING_STAGE_NAME,
                        int(patch_row["patch_id"]),
                    ),
                )
            connection.commit()

    def create_run(
        self,
        *,
        stage_name: str,
        config_path: Path | None = None,
        input_summary_json_path: Path | None = None,
    ) -> int:
        """Insert one stage execution row and return its run_id."""

        self.initialize()
        config_path_str = str(config_path) if config_path is not None else None
        input_summary_path_str = (
            str(input_summary_json_path) if input_summary_json_path is not None else None
        )
        config_sha256 = (
            hash_file_sha256(config_path)
            if config_path is not None and config_path.is_file()
            else None
        )
        input_summary_sha256 = (
            hash_file_sha256(input_summary_json_path)
            if input_summary_json_path is not None and input_summary_json_path.is_file()
            else None
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    stage_name,
                    config_path,
                    config_sha256,
                    completed_at,
                    input_summary_json_path,
                    input_summary_sha256
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """,
                (
                    stage_name,
                    config_path_str,
                    config_sha256,
                    input_summary_path_str,
                    input_summary_sha256,
                ),
            )
            connection.commit()
        if cursor.lastrowid is None:
            raise ValueError(f"{stage_name} run insert did not return a run_id.")
        return int(cursor.lastrowid)

    def create_normalization_artifact(
        self,
        *,
        run_id: int,
        method: str,
        state_path: Path,
        template_path: Path | None,
        fit_scope: str,
    ) -> int:
        """Insert one Stage 5 normalization artifact row and return its id."""

        self.initialize()
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Stage 5 normalization state artifact is missing: {state_path}"
            )
        state_sha256 = hash_file_sha256(state_path)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO normalization_artifacts (
                    run_id,
                    method,
                    state_path,
                    state_sha256,
                    template_path,
                    template_sha256,
                    fit_scope
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    method,
                    str(state_path),
                    state_sha256,
                    str(template_path) if template_path is not None else None,
                    None,
                    fit_scope,
                ),
            )
            connection.commit()
        if cursor.lastrowid is None:
            raise ValueError("Stage 5 normalization artifact insert did not return an id.")
        return int(cursor.lastrowid)

    def update_stage5_split_assignments(
        self,
        *,
        assignments: Sequence[Mapping[str, object]],
        normalization_method: str,
        normalization_artifact_id: int | None,
    ) -> None:
        """Persist Stage 5 split assignments onto existing canonical patch rows."""

        self.initialize()
        with self._connect() as connection:
            for assignment in assignments:
                source_hdf5_path = str(assignment["source_hdf5_path"])
                source_row_index = _coerce_int(
                    assignment.get("source_row_index"),
                    field_name="source_row_index",
                )
                patch_row = connection.execute(
                    "SELECT patch_id, filename, patient_id, label FROM patches "
                    "WHERE source_hdf5_path = ? AND source_row_index = ?",
                    (source_hdf5_path, source_row_index),
                ).fetchone()
                if patch_row is None:
                    raise ValueError(
                        "Stage 5 split assignment targets a missing canonical Stage 2 row: "
                        f"({source_hdf5_path}, {source_row_index})"
                    )
                self._validate_stage5_assignment_provenance(
                    patch_row=patch_row,
                    assignment=assignment,
                )
                connection.execute(
                    """
                    UPDATE patch_stage_state
                    SET split = ?,
                        normalization_method = ?,
                        normalization_artifact_id = ?,
                        last_updated_stage_name = ?,
                        last_updated_at = CURRENT_TIMESTAMP
                    WHERE patch_id = ?
                    """,
                    (
                        str(assignment["split"]),
                        normalization_method,
                        normalization_artifact_id,
                        STAGE5_STAGE_NAME,
                        int(patch_row["patch_id"]),
                    ),
                )
            connection.commit()

    def update_stage7_sampling_decisions(
        self,
        *,
        decisions: Sequence[Mapping[str, object]],
    ) -> None:
        """Persist Stage 7.2 sampling decisions onto existing canonical patch rows."""

        self.initialize()
        with self._connect() as connection:
            for decision in decisions:
                source_hdf5_path = str(decision["source_hdf5_path"])
                source_row_index = _coerce_int(
                    decision.get("source_row_index"),
                    field_name="source_row_index",
                )
                patch_row = connection.execute(
                    "SELECT patch_id, filename, patient_id, label FROM patches "
                    "WHERE source_hdf5_path = ? AND source_row_index = ?",
                    (source_hdf5_path, source_row_index),
                ).fetchone()
                if patch_row is None:
                    raise ValueError(
                        "Stage 7.2 sampling decision targets a missing canonical Stage 2 row: "
                        f"({source_hdf5_path}, {source_row_index})"
                    )
                self._validate_stage5_assignment_provenance(
                    patch_row=patch_row,
                    assignment=decision,
                )
                connection.execute(
                    """
                    UPDATE patch_stage_state
                    SET sampling_decision = ?,
                        is_stage7_selected = ?,
                        last_updated_stage_name = ?,
                        last_updated_at = CURRENT_TIMESTAMP
                    WHERE patch_id = ?
                    """,
                    (
                        str(decision["sampling_decision"]),
                        1 if bool(decision["is_stage7_selected"]) else 0,
                        STAGE7_STAGE_NAME,
                        int(patch_row["patch_id"]),
                    ),
                )
            connection.commit()

    def _validate_stage4_decision_provenance(
        self,
        *,
        patch_row: sqlite3.Row,
        decision: Mapping[str, object],
    ) -> None:
        expected_filename = _coerce_str_or_none(decision.get("filename"))
        if expected_filename is not None and expected_filename != str(patch_row["filename"]):
            raise ValueError(
                "Stage 4.3 decision filename does not match the canonical Stage 2 row: "
                f"expected {patch_row['filename']!r}, got {expected_filename!r}."
            )

        expected_patient_id = decision.get("patient_id")
        if expected_patient_id is not None and _coerce_int(
            expected_patient_id,
            field_name="patient_id",
        ) != int(patch_row["patient_id"]):
            raise ValueError(
                "Stage 4.3 decision patient_id does not match the canonical Stage 2 row: "
                f"expected {patch_row['patient_id']!r}, got {expected_patient_id!r}."
            )

        expected_slide_id = _coerce_str_or_none(decision.get("slide_id"))
        actual_slide_id = _coerce_str_or_none(patch_row["slide_id"])
        if expected_slide_id is not None and expected_slide_id != actual_slide_id:
            raise ValueError(
                "Stage 4.3 decision slide_id does not match the canonical Stage 2 row: "
                f"expected {actual_slide_id!r}, got {expected_slide_id!r}."
            )

        expected_source_signature = _coerce_str_or_none(decision.get("source_hdf5_sha256"))
        actual_source_signature = _coerce_str_or_none(patch_row["source_signature"])
        if (
            expected_source_signature is not None
            and expected_source_signature != actual_source_signature
        ):
            raise ValueError(
                "Stage 4.3 decision source_signature does not match the canonical Stage 2 row: "
                f"expected {actual_source_signature!r}, got {expected_source_signature!r}."
            )

    def _validate_stage5_assignment_provenance(
        self,
        *,
        patch_row: sqlite3.Row,
        assignment: Mapping[str, object],
    ) -> None:
        expected_filename = _coerce_str_or_none(assignment.get("filename"))
        if expected_filename is not None and expected_filename != str(patch_row["filename"]):
            raise ValueError(
                "Stage 5 split assignment filename does not match the canonical Stage 2 row: "
                f"expected {patch_row['filename']!r}, got {expected_filename!r}."
            )

        expected_patient_id = assignment.get("patient_id")
        if expected_patient_id is not None and _coerce_int(
            expected_patient_id,
            field_name="patient_id",
        ) != int(patch_row["patient_id"]):
            raise ValueError(
                "Stage 5 split assignment patient_id does not match the canonical Stage 2 row: "
                f"expected {patch_row['patient_id']!r}, got {expected_patient_id!r}."
            )

        expected_label = assignment.get("label")
        if expected_label is not None and _coerce_int(
            expected_label,
            field_name="label",
        ) != int(patch_row["label"]):
            raise ValueError(
                "Stage 5 split assignment label does not match the canonical Stage 2 row: "
                f"expected {patch_row['label']!r}, got {expected_label!r}."
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _read_source_signature(self, source_hdf5_path: Path) -> str:
        if not source_hdf5_path.is_file():
            raise FileNotFoundError(
                f"Stage 2 master manifest requires an existing HDF5 shard: {source_hdf5_path}"
            )
        with h5py.File(source_hdf5_path, "r") as handle:
            source_signature = handle.attrs.get("source_signature")
        if source_signature is None:
            raise ValueError(
                f"Stage 2 master manifest requires source_signature on shard: {source_hdf5_path}"
            )
        return str(source_signature)

    def _artifact_coverages(self, record: Mapping[str, object]) -> dict[str, float]:
        return {
            "cov_fold": _coerce_float(record.get("cov_fold", 0.0), field_name="cov_fold"),
            "cov_penmarking": _coerce_float(
                record.get("cov_penmarking", 0.0),
                field_name="cov_penmarking",
            ),
            "cov_oof": _coerce_float(record.get("cov_oof", 0.0), field_name="cov_oof"),
            "cov_darkspot_foreign": _coerce_float(
                record.get("cov_darkspot_foreign", 0.0),
                field_name="cov_darkspot_foreign",
            ),
            "cov_edge_airbubble": _coerce_float(
                record.get("cov_edge_airbubble", 0.0),
                field_name="cov_edge_airbubble",
            ),
        }
