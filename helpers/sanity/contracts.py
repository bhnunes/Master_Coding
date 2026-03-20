from __future__ import annotations

import re

import pandas as pd

from helpers.sanity.models import CheckResult

PATIENT_FILENAME_PATTERN = re.compile(r"PATIENT_(\d+)_")


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
