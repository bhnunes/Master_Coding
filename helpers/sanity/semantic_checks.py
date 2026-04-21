from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from helpers.sanity.disk_checks import IndexedInspection, inspect_hdf5_row
from helpers.sanity.models import CheckResult


def _load_mask_positive_flag(
    record: dict[str, object],
    *,
    base_dir: Path,
    row_inspections: dict[int, IndexedInspection] | None,
) -> bool | None:
    if row_inspections is not None:
        inspection = row_inspections[cast(int, record["index"])].inspection
    else:
        inspection = inspect_hdf5_row(
            str(base_dir / str(record["relative_hdf5_path"])),
            cast(int, record["hdf5_row_index"]),
        )
    return inspection["mask_has_positive_pixels"]


def _build_mask_semantics_result(
    *,
    positive_not_cancer: list[str],
    empty_cancer: list[str],
    fail_on_empty_cancer_mask: bool,
    fail_on_positive_not_cancer_mask: bool,
) -> CheckResult:
    failures: list[str] = []
    if fail_on_positive_not_cancer_mask and positive_not_cancer:
        failures.append(f"NOT_CANCER masks contain positive pixels: {positive_not_cancer[:10]}")
    if fail_on_empty_cancer_mask and empty_cancer:
        failures.append(f"CANCER masks are empty: {empty_cancer[:10]}")
    if failures:
        return CheckResult("FAIL", "; ".join(failures))

    warnings: list[str] = []
    if positive_not_cancer:
        warnings.append(f"NOT_CANCER masks contain positive pixels: {positive_not_cancer[:10]}")
    if empty_cancer:
        warnings.append(f"CANCER masks are empty: {empty_cancer[:10]}")
    if warnings:
        return CheckResult("WARN", "; ".join(warnings))

    return CheckResult("PASS", "Mask contents are semantically consistent with split labels.")


def check_mask_label_semantics(
    manifest_split: pd.DataFrame,
    base_dir: Path,
    split: str,
    *,
    fail_on_empty_cancer_mask: bool = True,
    fail_on_positive_not_cancer_mask: bool = True,
    row_inspections: dict[int, IndexedInspection] | None = None,
) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")
    scan_df = manifest_split.reset_index()
    positive_not_cancer: list[str] = []
    empty_cancer: list[str] = []
    for record in tqdm(
        scan_df.to_dict("records"),
        total=len(scan_df),
        desc=f"{split}: mask semantics",
        leave=False,
    ):
        has_positive_pixels = _load_mask_positive_flag(
            record,
            base_dir=base_dir,
            row_inspections=row_inspections,
        )
        if has_positive_pixels is None:
            return CheckResult("FAIL", f"Unreadable mask encountered for {record['filename']}.")
        if int(record["label"]) == 0 and has_positive_pixels:
            positive_not_cancer.append(str(record["filename"]))
        if int(record["label"]) == 1 and not has_positive_pixels:
            empty_cancer.append(str(record["filename"]))
    return _build_mask_semantics_result(
        positive_not_cancer=positive_not_cancer,
        empty_cancer=empty_cancer,
        fail_on_empty_cancer_mask=fail_on_empty_cancer_mask,
        fail_on_positive_not_cancer_mask=fail_on_positive_not_cancer_mask,
    )


def check_class_balance_visibility(manifest_split: pd.DataFrame) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; class balance not meaningful.")
    n0 = int((manifest_split["label"] == 0).sum())
    n1 = int((manifest_split["label"] == 1).sum())
    total = n0 + n1
    pct1 = (100.0 * n1 / total) if total else 0.0
    return CheckResult(
        "PASS",
        f"Patch-level class counts: NOT_CANCER={n0}, CANCER={n1} ({pct1:.2f}% cancer patches).",
        stats={"n0": n0, "n1": n1, "pct1": pct1},
    )


def check_patient_level_balance(manifest_split: pd.DataFrame) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; patient-level balance not meaningful.")
    patient_labels = manifest_split.groupby("patient_id")["label"].max()
    pos_count = int(patient_labels.sum())
    total_count = int(len(patient_labels))
    neg_count = total_count - pos_count
    return CheckResult(
        "PASS",
        f"Patient-level label counts: neg={neg_count}, pos={pos_count}, total={total_count}.",
        stats={"neg": neg_count, "pos": pos_count, "total": total_count},
    )


def check_patches_per_patient_stats(manifest_split: pd.DataFrame) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; patches-per-patient stats not meaningful.")
    counts = manifest_split.groupby("patient_id").size().astype(int)
    quantiles = counts.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    stats = {
        "mean": float(counts.mean()),
        "std": float(counts.std(ddof=0)),
        "min": int(quantiles[0.0]),
        "q1": float(quantiles[0.25]),
        "median": float(quantiles[0.5]),
        "q3": float(quantiles[0.75]),
        "max": int(quantiles[1.0]),
    }
    skewed = stats["max"] > 20 * max(1.0, stats["median"])
    details = (
        f"Patches per patient: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
        f"min={stats['min']}, q1={stats['q1']:.2f}, median={stats['median']:.2f}, "
        f"q3={stats['q3']:.2f}, max={stats['max']}."
    )
    if skewed:
        return CheckResult(
            "WARN", details + " Extreme skew detected (max > 20x median).", stats=stats
        )
    return CheckResult("PASS", details, stats=stats)


def check_split_class_presence(manifest_split: pd.DataFrame) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; class presence not meaningful.")
    labels = set(manifest_split["label"].astype(int).unique().tolist())
    if labels == {0, 1}:
        return CheckResult("PASS", "Both classes are present in this split.")
    return CheckResult("WARN", f"Only one class is present in this split: {sorted(labels)}")
