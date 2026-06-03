from __future__ import annotations

import re

import pandas as pd

from helpers.sanity.models import CheckResult

PATIENT_FILENAME_PATTERN = re.compile(r"PATIENT_(\d+)_")
LOGICAL_HDF5_REF_PATTERN = re.compile(r"^HDF5::(?P<dataset>images|masks)\[(?P<row>\d+)\]$")
RUNTIME_HDF5_REF_PATTERN = re.compile(r"^(?P<path>.+)::(?P<dataset>images|masks)\[(?P<row>\d+)\]$")


def _match_hdf5_ref(reference: str) -> re.Match[str] | None:
    return LOGICAL_HDF5_REF_PATTERN.match(reference) or RUNTIME_HDF5_REF_PATTERN.match(reference)


def check_filename_uniqueness(manifest_df: pd.DataFrame) -> CheckResult:
    duplicate_counts = manifest_df["filename"].value_counts()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if duplicates.empty:
        return CheckResult("PASS", "All filenames are globally unique across the dataset.")
    examples = list(duplicates.to_dict().keys())[:10]
    return CheckResult(
        "FAIL",
        (
            f"Found duplicate filenames across the dataset: {examples}. "
            "Filename uniqueness is required downstream."
        ),
        stats={"duplicate_filenames": int(len(duplicates))},
    )


def check_filename_patient_id_consistency(manifest_df: pd.DataFrame) -> CheckResult:
    mismatches: list[str] = []
    unparsable: list[str] = []
    for row in manifest_df.itertuples(index=False):
        filename = str(row.filename)
        match = PATIENT_FILENAME_PATTERN.search(filename)
        if match is None:
            unparsable.append(filename)
            continue
        if int(match.group(1)) != int(row.patient_id):
            mismatches.append(filename)
    if unparsable:
        return CheckResult(
            "FAIL",
            f"Filename(s) do not match the expected PATIENT_(id)_ pattern: {unparsable[:10]}",
        )
    if mismatches:
        return CheckResult(
            "FAIL",
            (
                "Filename-derived patient_id does not match manifest patient_id "
                f"for: {mismatches[:10]}"
            ),
        )
    return CheckResult("PASS", "Filename regex and manifest patient_id values are consistent.")


def check_source_reference_contract(manifest_df: pd.DataFrame) -> CheckResult:
    source_columns = {"source_image_path", "source_mask_path"}
    if not source_columns.issubset(manifest_df.columns):
        return CheckResult(
            "N/A",
            "Manifest has no source provenance columns; skipping source reference checks.",
        )

    invalid: list[str] = []
    for row in manifest_df.itertuples(index=False):
        image_ref = str(getattr(row, "source_image_path", "") or "")
        mask_ref = str(getattr(row, "source_mask_path", "") or "")
        if not image_ref and not mask_ref:
            continue
        if not image_ref or not mask_ref:
            invalid.append(str(getattr(row, "filename", "<missing filename>")))
            continue

        image_match = _match_hdf5_ref(image_ref)
        mask_match = _match_hdf5_ref(mask_ref)
        if image_match and mask_match:
            if (
                image_match.group("row") != mask_match.group("row")
                or image_match.group("dataset") != "images"
                or mask_match.group("dataset") != "masks"
                or image_match.groupdict().get("path") != mask_match.groupdict().get("path")
            ):
                invalid.append(str(getattr(row, "filename", "<missing filename>")))
            continue

        if image_match or mask_match:
            invalid.append(str(getattr(row, "filename", "<missing filename>")))
            continue

        if image_ref or mask_ref:
            invalid.append(str(getattr(row, "filename", "<missing filename>")))

    if invalid:
        return CheckResult(
            "FAIL",
            "Invalid source provenance contract: logical HDF5 refs must be paired image/mask "
            f"references to the same source row. Examples: {invalid[:10]}",
        )
    return CheckResult(
        "PASS",
        "Source references are valid, including logical HDF5 refs when present.",
    )


def check_canonical_source_row_contract(manifest_df: pd.DataFrame) -> CheckResult:
    """Validate that sanity inputs reference canonical source HDF5 rows."""

    missing_paths = [
        str(row.filename)
        for row in manifest_df.itertuples(index=False)
        if not str(getattr(row, "source_hdf5_path", "") or "")
    ][:10]
    negative_rows = [
        str(row.filename)
        for row in manifest_df.itertuples(index=False)
        if int(getattr(row, "source_row_index", -1)) < 0
    ][:10]
    if missing_paths or negative_rows:
        return CheckResult(
            "FAIL",
            "Invalid canonical source-row references. "
            f"missing_paths={missing_paths}, negative_rows={negative_rows}",
        )
    return CheckResult(
        "PASS",
        "Manifest uses canonical source_hdf5_path/source_row_index references.",
    )
