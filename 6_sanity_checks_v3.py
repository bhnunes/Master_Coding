#!/usr/bin/env python3
"""
Scientific Integrity & Data Quality Assurance (Manifest-Driven)

Purpose
-------
Run strict, reviewer-grade sanity checks on a prepared dataset directory produced by the
data-prep pipeline (that writes manifest.csv, split_stats.csv, run_config.json).

Outputs
-------
- Prints a detailed report with PASS/WARN/FAIL per check and per split.
- Final verdict:
    SPLITS PASSED  (exit code 0)
    SPLITS REJECTED (exit code 2)

Design goals
------------
- Prevent silent scientific flaws or bugs from propagating into training.
- Prefer using manifest.csv as the source of truth, and verify disk parity.
- Fail-fast on hard integrity violations (e.g., patient leakage, missing files, unreadable masks).

How to run
----------
python 6_sanity_checks_v3_manifest_driven.py --base_dir /path/to/output_run_dir

Optional:
  --full_mask_scan         Scan ALL masks for pixel values (slow but thorough)
  --full_shape_scan        Check shapes for ALL image/mask pairs (slow but thorough)
  --checksum_mode off|sample|full   Verify sha256 in manifest (default: sample)
  --sample_pairs 1000      Number of pairs to sample for expensive checks
"""

import argparse
import json
import os
import platform
import sys
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

SPLITS = ["TRAIN", "VALIDATION", "TEST"]
LABEL_DIR = {0: "NOT_CANCER", 1: "CANCER"}
MASK_DIR = {0: "NOT_CANCER_MASK", 1: "CANCER_MASK"}
EXPECTED_MASK_VALUES = {0, 255}

# -----------------------------
# Reporting utilities
# -----------------------------

@dataclass
class CheckResult:
    status: str   # PASS | WARN | FAIL | N/A
    details: str
    stats: Optional[dict] = None

def worst_status(statuses: List[str]) -> str:
    order = {"FAIL": 3, "WARN": 2, "PASS": 1, "N/A": 0}
    inv = {v: k for k, v in order.items()}
    return inv[max(order.get(s, 0) for s in statuses)] if statuses else "N/A"

def fmt_status(s: str) -> str:
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "N/A": "➖"}.get(s, "➖")
    return f"{icon} {s}"

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# -----------------------------
# Manifest loading & schema
# -----------------------------

def load_manifest(base_dir: str) -> pd.DataFrame:
    manifest_path = os.path.join(base_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found in base_dir: {manifest_path}")
    df = pd.read_csv(manifest_path)
    return df

def load_run_config(base_dir: str) -> Optional[dict]:
    p = os.path.join(base_dir, "run_config.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def check_manifest_schema(manifest_df: pd.DataFrame) -> CheckResult:
    # Required minimal columns for new pipeline (allow extra columns)
    required = {
        "run_id", "split", "label", "patient_id", "filename",
        "relative_path_image", "relative_path_mask",
        "normalization_method", "is_normalized",
    }
    missing = sorted(list(required - set(manifest_df.columns)))
    if missing:
        return CheckResult(
            "FAIL",
            f"Manifest missing required columns: {missing}. "
            "Re-generate dataset with updated data-prep script."
        )

    # Basic type/value checks
    bad_split = sorted(set(manifest_df["split"].dropna()) - set(SPLITS))
    if bad_split:
        return CheckResult("FAIL", f"Manifest contains invalid split names: {bad_split}")

    bad_label = sorted(set(manifest_df["label"].dropna()) - {0, 1})
    if bad_label:
        return CheckResult("FAIL", f"Manifest contains invalid labels (expected 0/1): {bad_label}")

    # run_id should be constant for a run
    run_ids = manifest_df["run_id"].dropna().unique().tolist()
    if len(run_ids) != 1:
        return CheckResult("WARN", f"Expected a single run_id; found {len(run_ids)}: {run_ids[:5]}...")

    return CheckResult("PASS", "Manifest schema and basic values look valid.")

# -----------------------------
# Dataset-wide checks
# -----------------------------

def check_patient_leakage(manifest_df: pd.DataFrame) -> CheckResult:
    # Patient must belong to exactly one split
    counts = manifest_df.groupby("patient_id")["split"].nunique()
    leaking = counts[counts > 1].index.tolist()
    if leaking:
        return CheckResult("FAIL", f"Patient leakage detected for {len(leaking)} patients: {leaking[:50]}{'...' if len(leaking)>50 else ''}")
    return CheckResult("PASS", "No patient leakage detected across TRAIN/VALIDATION/TEST.")

def check_duplicate_rows(manifest_df: pd.DataFrame) -> CheckResult:
    dup = manifest_df.duplicated(subset=["split", "label", "patient_id", "filename"]).sum()
    if dup > 0:
        return CheckResult("FAIL", f"Manifest contains {dup} duplicated (split,label,patient_id,filename) rows.")
    return CheckResult("PASS", "No duplicate rows found in manifest (by split,label,patient_id,filename).")

def check_split_patient_lists_against_run_config(manifest_df: pd.DataFrame, run_cfg: Optional[dict]) -> CheckResult:
    if not run_cfg:
        return CheckResult("N/A", "run_config.json not found; skipping patient-list cross-check.")
    for key in ["train_patients", "val_patients", "test_patients"]:
        if key not in run_cfg:
            return CheckResult("WARN", f"run_config.json missing key '{key}'; cannot cross-check patient lists.")
    # Compare sets
    cfg_train = set(run_cfg["train_patients"])
    cfg_val = set(run_cfg["val_patients"])
    cfg_test = set(run_cfg["test_patients"])

    m_train = set(manifest_df.loc[manifest_df["split"] == "TRAIN", "patient_id"].unique().tolist())
    m_val = set(manifest_df.loc[manifest_df["split"] == "VALIDATION", "patient_id"].unique().tolist())
    m_test = set(manifest_df.loc[manifest_df["split"] == "TEST", "patient_id"].unique().tolist())

    diffs = []
    if cfg_train != m_train:
        diffs.append(f"TRAIN mismatch: cfg_only={sorted(list(cfg_train-m_train))[:10]} | manifest_only={sorted(list(m_train-cfg_train))[:10]}")
    if cfg_val != m_val:
        diffs.append(f"VALIDATION mismatch: cfg_only={sorted(list(cfg_val-m_val))[:10]} | manifest_only={sorted(list(m_val-cfg_val))[:10]}")
    if cfg_test != m_test:
        diffs.append(f"TEST mismatch: cfg_only={sorted(list(cfg_test-m_test))[:10]} | manifest_only={sorted(list(m_test-cfg_test))[:10]}")

    if diffs:
        return CheckResult("FAIL", "Patient lists in manifest do not match run_config.json. " + " || ".join(diffs))
    return CheckResult("PASS", "Patient lists match run_config.json for all splits.")

def check_split_constraints_from_run_config(manifest_df: pd.DataFrame, run_cfg: Optional[dict]) -> CheckResult:
    if not run_cfg:
        return CheckResult("N/A", "run_config.json not found; skipping constraint cross-check.")
    # If these are not present, we won't fail; we'll warn.
    needed = ["min_test_patients", "min_train_patients", "min_val_patients"]
    missing = [k for k in needed if k not in run_cfg]
    if missing:
        return CheckResult("WARN", f"run_config.json missing constraint keys {missing}; cannot verify split constraints.")
    # Verify patient minima
    n_train = manifest_df[manifest_df["split"]=="TRAIN"]["patient_id"].nunique()
    n_val = manifest_df[manifest_df["split"]=="VALIDATION"]["patient_id"].nunique()
    n_test = manifest_df[manifest_df["split"]=="TEST"]["patient_id"].nunique()
    fails = []
    if n_train < int(run_cfg["min_train_patients"]): fails.append(f"TRAIN patients {n_train} < {run_cfg['min_train_patients']}")
    if n_val < int(run_cfg["min_val_patients"]): fails.append(f"VALIDATION patients {n_val} < {run_cfg['min_val_patients']}")
    if n_test < int(run_cfg["min_test_patients"]): fails.append(f"TEST patients {n_test} < {run_cfg['min_test_patients']}")
    if fails:
        return CheckResult("FAIL", "Split constraints violated: " + "; ".join(fails))
    return CheckResult("PASS", "Split patient minima satisfy constraints in run_config.json.")

# -----------------------------
# Split-level disk integrity
# -----------------------------

def _count_pngs(p: str) -> int:
    if not os.path.isdir(p):
        return 0
    return sum(1 for f in os.listdir(p) if f.lower().endswith(".png"))

def check_manifest_disk_parity(manifest_split: pd.DataFrame, base_dir: str, split: str) -> CheckResult:
    # Count pngs in dirs, compare to manifest rows
    errors = []
    for label in [0, 1]:
        img_dir = os.path.join(base_dir, split, LABEL_DIR[label])
        disk_n = _count_pngs(img_dir)
        man_n = int((manifest_split["label"] == label).sum())
        if disk_n != man_n:
            errors.append(f"{LABEL_DIR[label]} mismatch: manifest={man_n}, disk={disk_n}")
        mask_dir = os.path.join(base_dir, split, MASK_DIR[label])
        disk_m = _count_pngs(mask_dir)
        if disk_m != disk_n:
            errors.append(f"{MASK_DIR[label]} mismatch vs images: images={disk_n}, masks={disk_m}")
    if errors:
        return CheckResult("FAIL", "; ".join(errors))
    return CheckResult("PASS", "Manifest counts match disk counts for images and masks (per class).")

def check_paths_exist_and_relative(manifest_split: pd.DataFrame, base_dir: str, split: str, sample_n: int = 1000) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")

    sample = manifest_split.sample(n=min(sample_n, len(manifest_split)), random_state=42)
    missing = []
    abs_path = []
    for _, r in sample.iterrows():
        rel_img = str(r["relative_path_image"])
        rel_msk = str(r["relative_path_mask"])
        if os.path.isabs(rel_img) or os.path.isabs(rel_msk):
            abs_path.append(r["filename"])
        img_path = os.path.join(base_dir, rel_img)
        msk_path = os.path.join(base_dir, rel_msk)
        if not os.path.isfile(img_path) or not os.path.isfile(msk_path):
            missing.append(r["filename"])
    if abs_path:
        return CheckResult("FAIL", f"Found absolute paths in manifest (should be relative). Examples: {abs_path[:10]}")
    if missing:
        return CheckResult("FAIL", f"Missing files referenced by manifest (sampled {len(sample)}): {missing[:20]}{'...' if len(missing)>20 else ''}")
    return CheckResult("PASS", f"Sampled {len(sample)} rows: relative paths exist for both image and mask.")

def check_decode_and_shapes(manifest_split: pd.DataFrame, base_dir: str, split: str, sample_n: int = 1000, full_scan: bool = False) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")

    df = manifest_split
    if not full_scan:
        df = df.sample(n=min(sample_n, len(df)), random_state=123)

    mismatches = []
    unreadable = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{split}: decoding & shape", leave=False):
        img_path = os.path.join(base_dir, str(r["relative_path_image"]))
        msk_path = os.path.join(base_dir, str(r["relative_path_mask"]))

        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            unreadable.append(r["filename"])
            continue
        if img.shape[0] != msk.shape[0] or img.shape[1] != msk.shape[1]:
            mismatches.append(f"{r['filename']} (img={img.shape[:2]} mask={msk.shape[:2]})")

    if unreadable:
        return CheckResult("FAIL", f"Unreadable image/mask pairs: {len(unreadable)} examples: {unreadable[:10]}")
    if mismatches:
        return CheckResult("FAIL", f"Image/mask shape mismatches: {len(mismatches)} examples: {mismatches[:10]}")
    return CheckResult("PASS", f"{'Full-scan' if full_scan else 'Sampled'} {len(df)} pairs: decodable and shapes match.")

def check_mask_pixel_values(manifest_split: pd.DataFrame, base_dir: str, split: str, sample_n: int = 1000, full_scan: bool = False) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("PASS", "Split is empty; nothing to check.")

    df = manifest_split
    if not full_scan:
        df = df.sample(n=min(sample_n, len(df)), random_state=999)

    unexpected = set()
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{split}: mask values", leave=False):
        msk_path = os.path.join(base_dir, str(r["relative_path_mask"]))
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            return CheckResult("FAIL", f"Unreadable mask encountered (e.g., {r['filename']}).")
        vals = set(np.unique(msk).tolist())
        unexpected |= (vals - EXPECTED_MASK_VALUES)
        if unexpected and full_scan:
            # keep scanning to aggregate values; for sample scan, we can stop early
            pass
        elif unexpected and (not full_scan):
            break

    if unexpected:
        return CheckResult("FAIL", f"Unexpected mask pixel values found: {sorted(list(unexpected))}. Expected only {sorted(list(EXPECTED_MASK_VALUES))}.")
    return CheckResult("PASS", f"{'Full-scan' if full_scan else 'Sampled'} masks: pixel values are only {sorted(list(EXPECTED_MASK_VALUES))}.")

def check_class_balance_visibility(manifest_split: pd.DataFrame, split: str) -> CheckResult:
    # Not a fail condition, but highly informative for reviewers
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; class balance not meaningful.")
    n0 = int((manifest_split["label"] == 0).sum())
    n1 = int((manifest_split["label"] == 1).sum())
    total = n0 + n1
    pct1 = (100.0 * n1 / total) if total else 0.0
    return CheckResult("PASS", f"Patch-level class counts: NOT_CANCER={n0}, CANCER={n1} ({pct1:.2f}% cancer patches).", stats={"n0": n0, "n1": n1, "pct1": pct1})

def check_patient_level_balance(manifest_split: pd.DataFrame, split: str) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; patient-level balance not meaningful.")
    # patient label = max patch label per patient
    pt = manifest_split.groupby("patient_id")["label"].max()
    n_pos = int(pt.sum())
    n_total = int(len(pt))
    n_neg = n_total - n_pos
    return CheckResult("PASS", f"Patient-level label (max patch label) counts: neg={n_neg}, pos={n_pos}, total={n_total}.", stats={"neg": n_neg, "pos": n_pos, "total": n_total})

def check_patches_per_patient_stats(manifest_split: pd.DataFrame, split: str) -> CheckResult:
    if manifest_split.empty:
        return CheckResult("WARN", "Split empty; patches-per-patient stats not meaningful.")
    counts = manifest_split.groupby("patient_id").size().astype(int)
    q = counts.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    stats = {
        "mean": float(counts.mean()),
        "std": float(counts.std(ddof=0)),
        "min": int(q[0.0]),
        "q1": float(q[0.25]),
        "median": float(q[0.5]),
        "q3": float(q[0.75]),
        "max": int(q[1.0]),
    }
    # Warn if extreme skew (helpful reviewer narrative)
    skew_warn = stats["max"] > 20 * max(1.0, stats["median"])
    status = "WARN" if skew_warn else "PASS"
    details = (
        f"Patches per patient: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
        f"min={stats['min']}, q1={stats['q1']:.2f}, median={stats['median']:.2f}, "
        f"q3={stats['q3']:.2f}, max={stats['max']}."
        + (" Extreme skew detected (max > 20x median)." if skew_warn else "")
    )
    return CheckResult(status, details, stats=stats)

def check_checksums(manifest_split: pd.DataFrame, base_dir: str, split: str, mode: str = "sample", sample_n: int = 200) -> CheckResult:
    # Only if sha256 columns exist
    if "sha256_image" not in manifest_split.columns or "sha256_mask" not in manifest_split.columns:
        return CheckResult("N/A", "sha256 columns not present in manifest; skipping checksum verification.")
    if mode == "off":
        return CheckResult("N/A", "Checksum verification disabled (--checksum_mode off).")

    df = manifest_split
    if mode == "sample":
        df = df.sample(n=min(sample_n, len(df)), random_state=2026)

    bad = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{split}: checksums", leave=False):
        img_path = os.path.join(base_dir, str(r["relative_path_image"]))
        msk_path = os.path.join(base_dir, str(r["relative_path_mask"]))
        try:
            if str(r["sha256_image"]) != sha256_file(img_path):
                bad.append(f"{r['filename']} (image)")
            if str(r["sha256_mask"]) != sha256_file(msk_path):
                bad.append(f"{r['filename']} (mask)")
        except Exception as e:
            return CheckResult("FAIL", f"Checksum computation error on {r['filename']}: {e}")

    if bad:
        return CheckResult("FAIL", f"Checksum mismatch for {len(bad)} items. Examples: {bad[:10]}")
    return CheckResult("PASS", f"Checksum verification OK ({mode}): checked {len(df)} pairs.")

# -----------------------------
# Orchestrator
# -----------------------------

def run_checks(base_dir, sample_pairs, full_mask_scan, full_shape_scan, checksum_mode) -> Tuple[Dict[str, Dict[str, CheckResult]], Dict[str, CheckResult], str]:
    manifest_df = load_manifest(base_dir)
    run_cfg = load_run_config(base_dir)

    dataset_checks = {
        "Manifest Schema": check_manifest_schema(manifest_df),
        "Duplicate Rows": check_duplicate_rows(manifest_df),
        "Patient Leakage": check_patient_leakage(manifest_df),
        "RunConfig Patient Lists": check_split_patient_lists_against_run_config(manifest_df, run_cfg),
        "RunConfig Constraints": check_split_constraints_from_run_config(manifest_df, run_cfg),
    }

    split_results: Dict[str, Dict[str, CheckResult]] = {}
    for split in SPLITS:
        sdf = manifest_df[manifest_df["split"] == split].copy()

        split_results[split] = {
            "Disk Parity": check_manifest_disk_parity(sdf, base_dir, split),
            "Paths Exist": check_paths_exist_and_relative(sdf, base_dir, split, sample_n=sample_pairs),
            "Decode & Shapes": check_decode_and_shapes(sdf, base_dir, split, sample_n=sample_pairs, full_scan=full_shape_scan),
            "Mask Pixel Values": check_mask_pixel_values(sdf, base_dir, split, sample_n=sample_pairs, full_scan=full_mask_scan),
            "Patch Class Balance": check_class_balance_visibility(sdf, split),
            "Patient Class Balance": check_patient_level_balance(sdf, split),
            "Patches/Patient": check_patches_per_patient_stats(sdf, split),
            "Checksums": check_checksums(sdf, base_dir, split, mode=checksum_mode, sample_n=min(200, sample_pairs)),
        }

    # Overall verdict: any FAIL in dataset_checks or any split check => REJECTED
    all_statuses = [r.status for r in dataset_checks.values()]
    for split in SPLITS:
        all_statuses += [r.status for r in split_results[split].values()]

    verdict = "SPLITS PASSED" if worst_status(all_statuses) != "FAIL" else "SPLITS REJECTED"
    return split_results, dataset_checks, verdict

def print_report(base_dir: str, split_results, dataset_checks, verdict: str):
    print("=" * 88)
    print("SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT (Manifest-Driven)")
    print("=" * 88)
    print(f"Base directory: {os.path.normpath(base_dir)}")
    print(f"Platform: {platform.platform()}")
    print("-" * 88)

    # Dataset-wide checks
    print("\n[DATASET-WIDE CHECKS]")
    for name, res in dataset_checks.items():
        print(f"  - {name:<26} {fmt_status(res.status)}  {res.details}")

    # Split summary table
    print("\n[SPLIT SUMMARY TABLE]")
    rows = []
    for split in SPLITS:
        row = {"Split": split}
        for chk, res in split_results[split].items():
            row[chk] = res.status
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Split")
    print(df.to_string())

    # Split details
    print("\n[DETAILED SPLIT FINDINGS]")
    for split in SPLITS:
        print("\n" + "-" * 88)
        print(f"SPLIT: {split}")
        print("-" * 88)
        for name, res in split_results[split].items():
            print(f"  - {name:<26} {fmt_status(res.status)}  {res.details}")

    print("\n" + "=" * 88)
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 88)


def main(base_dir, sample_pairs, full_mask_scan, full_shape_scan, checksum_mode):
    base_dir = os.path.normpath(base_dir)

    try:
        split_results, dataset_checks, verdict = run_checks(base_dir, sample_pairs, full_mask_scan, full_shape_scan, checksum_mode)
        print_report(base_dir, split_results, dataset_checks, verdict)
    except Exception as e:
        print("=" * 88)
        print("SCIENTIFIC INTEGRITY & DATA QUALITY ASSURANCE REPORT")
        print("=" * 88)
        print(f"❌ FATAL ERROR: {e}")
        print("FINAL VERDICT: SPLITS REJECTED")
        sys.exit(2)

    if verdict == "SPLITS PASSED":
        sys.exit(0)
    sys.exit(2)

if __name__ == "__main__":
    base_dir = 'D:\Usuario\Desktop\Base_de_dados\CAMELYON16\PATCHES\NOT_NORMALIZED\NOT_NORMALIZED_seed_42' #"Path to the prepared dataset run directory (contains manifest.csv)."
    sample_pairs = 1000 #"Number of image/mask pairs to sample for expensive checks."
    full_mask_scan = True #"Scan ALL masks for pixel values (slow)."
    full_shape_scan = True #"Decode+shape check ALL pairs (slow)."
    checksum_mode = "sample" #"Checksum verification: off, sample, or full (slow)."
    main(base_dir, sample_pairs, full_mask_scan, full_shape_scan, checksum_mode)
