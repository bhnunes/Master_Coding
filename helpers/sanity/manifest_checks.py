from __future__ import annotations

import math
from typing import Any

import pandas as pd

from helpers.sanity.models import SPLITS, CheckResult

REQUIRED_MANIFEST_COLUMNS = {
    "run_id",
    "split",
    "label",
    "patient_id",
    "filename",
    "relative_path_image",
    "relative_path_mask",
    "normalization_method",
    "is_normalized",
}
SPLIT_STATS_COLUMNS = (
    "n_patients",
    "n_pos_patients",
    "n_neg_patients",
    "n_images",
    "patches_per_patient_mean",
    "patches_per_patient_std",
    "patches_per_patient_min",
    "patches_per_patient_q1",
    "patches_per_patient_median",
    "patches_per_patient_q3",
    "patches_per_patient_max",
    "n_images_neg",
    "n_images_pos",
)


def check_manifest_schema(manifest_df: pd.DataFrame) -> CheckResult:
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest_df.columns))
    if missing:
        return CheckResult(
            "FAIL",
            (
                f"Manifest missing required columns: {missing}. "
                "Re-generate dataset with updated Stage 5."
            ),
        )
    invalid_splits = sorted(set(manifest_df["split"].dropna()) - set(SPLITS))
    if invalid_splits:
        return CheckResult("FAIL", f"Manifest contains invalid split names: {invalid_splits}")
    invalid_labels = sorted(set(manifest_df["label"].dropna()) - {0, 1})
    if invalid_labels:
        return CheckResult(
            "FAIL", f"Manifest contains invalid labels (expected 0/1): {invalid_labels}"
        )
    run_ids = manifest_df["run_id"].dropna().unique().tolist()
    if len(run_ids) != 1:
        return CheckResult("WARN", f"Expected a single run_id; found {len(run_ids)}: {run_ids[:5]}")
    return CheckResult("PASS", "Manifest schema and basic values look valid.")


def check_duplicate_rows(manifest_df: pd.DataFrame) -> CheckResult:
    duplicates = int(
        manifest_df.duplicated(subset=["split", "label", "patient_id", "filename"]).sum()
    )
    if duplicates:
        return CheckResult(
            "FAIL",
            f"Manifest contains {duplicates} duplicated (split, label, patient_id, filename) rows.",
        )
    return CheckResult("PASS", "No duplicate manifest rows found.")


def check_patient_leakage(manifest_df: pd.DataFrame) -> CheckResult:
    counts = manifest_df.groupby("patient_id")["split"].nunique()
    leaking = counts[counts > 1].index.tolist()
    if leaking:
        return CheckResult(
            "FAIL",
            f"Patient leakage detected for {len(leaking)} patients: {leaking[:50]}",
        )
    return CheckResult("PASS", "No patient leakage detected across TRAIN/VALIDATION/TEST.")


def check_split_patient_lists_against_run_config(
    manifest_df: pd.DataFrame,
    run_cfg: dict[str, Any] | None,
) -> CheckResult:
    if not run_cfg:
        return CheckResult("N/A", "run_config.json not found; skipping patient-list cross-check.")
    patients = run_cfg.get("patients")
    if not isinstance(patients, dict):
        return CheckResult(
            "WARN", "run_config.json missing key 'patients'; cannot cross-check patient lists."
        )
    train_list = patients.get("train")
    val_list = patients.get("validation", patients.get("val"))
    test_list = patients.get("test")
    if train_list is None or val_list is None or test_list is None:
        return CheckResult(
            "WARN", "run_config.json patient lists are incomplete; cannot cross-check all splits."
        )
    cfg_map = {
        "TRAIN": {int(value) for value in train_list},
        "VALIDATION": {int(value) for value in val_list},
        "TEST": {int(value) for value in test_list},
    }
    diffs: list[str] = []
    for split_name in SPLITS:
        manifest_patients = set(
            manifest_df.loc[manifest_df["split"] == split_name, "patient_id"]
            .astype(int)
            .unique()
            .tolist()
        )
        if manifest_patients != cfg_map[split_name]:
            cfg_only = sorted(cfg_map[split_name] - manifest_patients)[:10]
            manifest_only = sorted(manifest_patients - cfg_map[split_name])[:10]
            diffs.append(
                f"{split_name} mismatch: cfg_only={cfg_only} | manifest_only={manifest_only}"
            )
    if diffs:
        return CheckResult(
            "FAIL",
            "Patient lists in manifest do not match run_config.json. " + " || ".join(diffs),
        )
    return CheckResult("PASS", "Patient lists match run_config.json for all splits.")


def check_split_constraints_from_run_config(
    manifest_df: pd.DataFrame,
    run_cfg: dict[str, Any] | None,
) -> CheckResult:
    if not run_cfg:
        return CheckResult("N/A", "run_config.json not found; skipping constraint cross-check.")
    constraints = run_cfg.get("constraints")
    if not isinstance(constraints, dict):
        return CheckResult(
            "WARN", "run_config.json missing key 'constraints'; cannot verify split constraints."
        )
    missing = [
        key
        for key in ("min_test_patients", "min_train_patients", "min_val_patients")
        if key not in constraints
    ]
    if missing:
        return CheckResult(
            "WARN",
            f"run_config.json constraints missing keys {missing}; cannot verify split constraints.",
        )
    failures: list[str] = []
    counts = {split_name: _count_unique_patients(manifest_df, split_name) for split_name in SPLITS}
    if counts["TRAIN"] < int(constraints["min_train_patients"]):
        failures.append(f"TRAIN patients {counts['TRAIN']} < {constraints['min_train_patients']}")
    if counts["VALIDATION"] < int(constraints["min_val_patients"]):
        failures.append(
            f"VALIDATION patients {counts['VALIDATION']} < {constraints['min_val_patients']}"
        )
    if counts["TEST"] < int(constraints["min_test_patients"]):
        failures.append(f"TEST patients {counts['TEST']} < {constraints['min_test_patients']}")
    if failures:
        return CheckResult("FAIL", "Split constraints violated: " + "; ".join(failures))
    return CheckResult("PASS", "Split patient minima satisfy constraints in run_config.json.")


def _recompute_split_stats(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        split_df = manifest_df.loc[manifest_df["split"] == split_name].copy()
        if split_df.empty:
            continue
        patient_labels = split_df.groupby("patient_id")["label"].max()
        per_patient = split_df.groupby("patient_id").size()
        rows.append(
            {
                "run_id": split_df["run_id"].iloc[0],
                "split": split_name,
                "n_patients": int(split_df["patient_id"].nunique()),
                "n_pos_patients": int(patient_labels.sum()),
                "n_neg_patients": int(len(patient_labels) - patient_labels.sum()),
                "n_images": int(len(split_df)),
                "patches_per_patient_mean": float(per_patient.mean()),
                "patches_per_patient_std": float(per_patient.std(ddof=1))
                if len(per_patient) > 1
                else 0.0,
                "patches_per_patient_min": int(per_patient.min()),
                "patches_per_patient_q1": float(per_patient.quantile(0.25)),
                "patches_per_patient_median": float(per_patient.quantile(0.50)),
                "patches_per_patient_q3": float(per_patient.quantile(0.75)),
                "patches_per_patient_max": int(per_patient.max()),
                "n_images_neg": int((split_df["label"] == 0).sum()),
                "n_images_pos": int((split_df["label"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def _count_unique_patients(manifest_df: pd.DataFrame, split_name: str) -> int:
    split_df = manifest_df.loc[manifest_df["split"] == split_name].copy()
    return int(split_df["patient_id"].nunique()) if not split_df.empty else 0


def check_split_stats_against_manifest(
    manifest_df: pd.DataFrame,
    split_stats_df: pd.DataFrame | None,
) -> CheckResult:
    if split_stats_df is None:
        return CheckResult(
            "WARN", "split_stats.csv not found; cannot cross-check manifest-derived statistics."
        )
    recomputed_df = _recompute_split_stats(manifest_df)
    if recomputed_df.empty and split_stats_df.empty:
        return CheckResult("PASS", "split_stats.csv matches an empty manifest.")
    split_stats_indexed = split_stats_df.set_index("split")
    recomputed_indexed = recomputed_df.set_index("split")
    if set(split_stats_indexed.index) != set(recomputed_indexed.index):
        return CheckResult("FAIL", "split_stats.csv split membership does not match manifest.csv.")
    for split_name in recomputed_indexed.index:
        for column_name in SPLIT_STATS_COLUMNS:
            expected = recomputed_indexed.loc[split_name, column_name]
            observed = split_stats_indexed.loc[split_name, column_name]
            if isinstance(expected, float) or isinstance(observed, float):
                if not math.isclose(float(expected), float(observed), rel_tol=1e-6, abs_tol=1e-6):
                    return CheckResult(
                        "FAIL",
                        (
                            f"split_stats.csv mismatch for {split_name}.{column_name}: "
                            f"manifest={expected}, split_stats={observed}"
                        ),
                    )
            elif int(expected) != int(observed):
                return CheckResult(
                    "FAIL",
                    (
                        f"split_stats.csv mismatch for {split_name}.{column_name}: "
                        f"manifest={expected}, split_stats={observed}"
                    ),
                )
    return CheckResult("PASS", "split_stats.csv matches manifest-derived statistics.")
