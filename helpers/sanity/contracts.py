from __future__ import annotations

import re

import pandas as pd

from helpers.sanity.models import CheckResult
from helpers.stage_contracts import STAGE5_SINGLETON_SPLIT_FILES

PATIENT_FILENAME_PATTERN = re.compile(r"PATIENT_(\d+)_")
LOGICAL_HDF5_REF_PATTERN = re.compile(r"^HDF5::(?P<dataset>images|masks)\[(?P<row>\d+)\]$")
RUNTIME_HDF5_REF_PATTERN = re.compile(
    r"^(?P<path>.+)::(?P<dataset>images|masks)\[(?P<row>\d+)\]$"
)


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


def check_stage5_singleton_layout_contract(manifest_df: pd.DataFrame) -> CheckResult:
    """Validate that Stage 6 inputs still reference only canonical Stage 5 singleton outputs."""

    invalid_examples: list[str] = []
    expected_paths = STAGE5_SINGLETON_SPLIT_FILES
    for row in manifest_df.itertuples(index=False):
        split_name = str(getattr(row, "split", ""))
        relative_hdf5_path = str(getattr(row, "relative_hdf5_path", ""))
        expected_path = expected_paths.get(split_name)
        if expected_path is None or relative_hdf5_path != expected_path:
            invalid_examples.append(f"{split_name}:{relative_hdf5_path}")

    if invalid_examples:
        expected_layout = ", ".join(
            f"{split_name} -> {path}" for split_name, path in expected_paths.items()
        )
        return CheckResult(
            "FAIL",
            "Stage 6 only supports Stage 5 singleton split outputs. "
            f"Expected layout: {expected_layout}. Examples: {invalid_examples[:10]}",
        )
    return CheckResult(
        "PASS",
        "Manifest paths match the canonical Stage 5 singleton split layout.",
    )
