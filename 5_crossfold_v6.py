# -----------------------------------------------------------------------------
# UNIFIED DATA PREPARATION PIPELINE (REFactored + "Best Split" Selection)
# -----------------------------------------------------------------------------
# What changed vs your current script:
# 1) Split search can evaluate MANY feasible patient-level splits (up to max_tries)
#    instead of returning the first feasible one.
# 2) Adds an "information metric" objective:
#       - patch entropy for all patches (parallelizable)
#       - patient entropy = median(patch entropies per patient)
#       - split score = median(patient entropies over TRAIN patients)
#    Pick feasible split with highest score (or lowest if desired).
# 3) Adds an adaptive constraint sizing option so the same script works across datasets.
#
# What remains scientifically identical:
# - Patient-disjoint splitting (no leakage)
# - Patient stratification uses patient_label = max(patch_label)
# - Stain normalization fit ONLY on TRAIN, using high-entropy templates per TRAIN patient
# - Full traceability: manifest + split_stats + run_config
# -----------------------------------------------------------------------------

import os
import re
import sys
import json
import cv2
import math
import shutil
import hashlib
import logging
import platform
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from functools import partial
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Optional: OpenSlide / TIAtoolbox setup (same intent as your original script)
# -----------------------------------------------------------------------------
load_dotenv()
OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")

try:
    if hasattr(os, "add_dll_directory") and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            from tiatoolbox.tools import stainnorm
    else:
        from tiatoolbox.tools import stainnorm
except (ImportError, FileNotFoundError) as e:
    print("FATAL ERROR: Could not initialize OpenSlide / TIAtoolbox dependency.")
    print("1) Install OpenSlide binaries (Windows) if needed.")
    print("2) Ensure OPENSLIDE_PATH in .env points to the OpenSlide 'bin' folder.")
    print(f"   OPENSLIDE_PATH={OPENSLIDE_PATH}")
    print(f"   Details: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConstraints:
    # Desired minimum patients (may be relaxed if adaptive=True and dataset is too small)
    min_test_patients: int = 20
    min_val_patients: int = 5
    min_train_patients: int = 5

    # Ratios (used to size splits)
    test_ratio: float = 0.10
    val_ratio: float = 0.10

    # Guardrails
    require_train_image_dominance: bool = True  # TRAIN images > VAL and > TEST (your current Rule 2)
    require_both_classes_if_possible: bool = True  # If dataset has both classes, try to keep both in each split

    # Search
    max_tries: int = 1000

    # Adaptive behavior
    adaptive: bool = True


@dataclass(frozen=True)
class ObjectiveConfig:
    enable_objective: bool = True

    # Which split to score: typically TRAIN (because it is the “learning set”)
    score_split: str = "TRAIN"  # TRAIN | VALIDATION | TEST

    # Score direction
    maximize: bool = True  # True: pick highest score; False: pick lowest score

    # Metric
    metric: str = "entropy_median_per_patient_median"  # fixed in this implementation

    # Entropy computation
    num_workers: int = max(1, (os.cpu_count() or 1) - 1)
    chunksize: int = 128

    # NEW: entropy speedup
    entropy_thumbnail: int = 128  # set to 256 if you want higher fidelity


@dataclass(frozen=True)
class RunConfig:
    normalization_method: str = "NOT_NORMALIZED"  # NOT_NORMALIZED|REINHARD|RUIFROK|MACENKO|VAHADANE
    data_directory: str = r"D:\Usuario\Desktop\Base_de_dados\CAMELYON16\PATCHES"
    overwrite_output_dir: bool = True
    random_state: int = 42

    constraints: SplitConstraints = SplitConstraints()
    objective: ObjectiveConfig = ObjectiveConfig()

    # Reproducibility / performance
    calc_checksums: bool = True  # can be slow at 100k+ patches
    save_entropy_cache_csv: bool = True  # saves per-image entropy so you don’t recompute later


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "data_preparation.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logging.info(f"Logging configured: {log_file}")


# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

_PATIENT_RE = re.compile(r"PATIENT_(\d+)_")

def _extract_patient_id(filename: str) -> Optional[int]:
    m = _PATIENT_RE.search(filename)
    if not m:
        return None
    return int(m.group(1))


def calculate_image_entropy_from_path(image_path: str, thumb: int = 128) -> Tuple[str, float]:
    """
    Fast Shannon entropy of 8-bit grayscale image.

    Speedups:
      - computes entropy on a downsampled thumbnail (thumb x thumb)
      - uses np.bincount instead of cv2.calcHist (fast for small images)

    Returns: (image_path, entropy)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return image_path, 0.0

        if thumb is not None:
            # INTER_AREA is best for downsampling
            img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)

        # Histogram of uint8 values (0..255)
        hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
        s = hist.sum()
        if s <= 0:
            return image_path, 0.0

        p = hist / s
        p = p[p > 0]
        ent = float(-(p * np.log2(p)).sum())
        return image_path, ent

    except Exception:
        return image_path, 0.0


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Loads image + mask pairs from:
      CANCER / CANCER_MASK
      NOT_CANCER / NOT_CANCER_MASK

    Expects mask filename == image filename.
    Extracts patient_id from filename pattern: PATIENT_<id>_
    """
    logging.info(f"Loading data from: {data_dir}")
    rows = []

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.normpath(os.path.join(data_dir, label_name))
        mask_dir = os.path.normpath(os.path.join(data_dir, f"{label_name}_MASK"))
        label = 1 if label_name == "CANCER" else 0

        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            logging.warning(f"Missing dirs for {label_name}. Skipping.")
            continue

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith(".png")}

        logging.info(f"Scanning {label_name}: {len(image_files)} images")
        for fname in tqdm(image_files, desc=f"Indexing {label_name}", leave=False):
            if fname not in mask_files:
                logging.warning(f"Mask not found for {fname}. Skipping.")
                continue

            pid = _extract_patient_id(fname)
            if pid is None:
                logging.warning(f"Could not extract patient id from {fname}. Skipping.")
                continue

            img_path = os.path.join(image_dir, fname)
            msk_path = os.path.join(mask_dir, fname)
            if not (os.path.isfile(img_path) and os.path.isfile(msk_path)):
                logging.warning(f"Invalid paths for {fname}. Skipping.")
                continue


            rows.append(
                dict(
                    patient_id=pid,
                    image_path=img_path,
                    mask_path=msk_path,
                    label=label,
                    filename=fname,
                )
            )

    if not rows:
        raise ValueError("No valid, readable image/mask pairs were found.")

    df = pd.DataFrame(rows)
    logging.info(f"Loaded {len(df)} patch pairs from {df['patient_id'].nunique()} patients.")
    return df


# -----------------------------------------------------------------------------
# Patient-level tables and adaptive sizing
# -----------------------------------------------------------------------------

def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient:
      patient_label = max(patch_label)  (any cancer => 1)
      n_images      = number of patches for that patient (cancer + not-cancer)
    """
    return (
        df.groupby("patient_id")
          .agg(patient_label=("label", "max"), n_images=("label", "size"))
          .reset_index()
          .sort_values("patient_id")
          .reset_index(drop=True)
    )


def _adaptive_min_patients(
    N: int,
    constraints: SplitConstraints,
) -> Tuple[int, int, int]:
    """
    Returns (min_train, min_val, min_test) after adapting to dataset size if needed.
    Ensures at least 1 patient per split when possible.
    """
    min_train = constraints.min_train_patients
    min_val = constraints.min_val_patients
    min_test = constraints.min_test_patients

    if not constraints.adaptive:
        return min_train, min_val, min_test

    # If N is too small, relax minimums down to feasible values.
    # Priority: keep TEST and VAL at least 1 if possible; TRAIN at least 1 always.
    # We do NOT silently violate; we log what we relaxed.
    if N < (min_train + min_val + min_test):
        # Start from ratio-based sizes, then clip
        desired_test = max(1, int(round(constraints.test_ratio * N)))
        desired_val = max(1, int(round(constraints.val_ratio * N)))
        desired_train = max(1, N - desired_test - desired_val)

        # If still not feasible (due to rounding), repair deterministically
        if desired_train <= 0:
            desired_train = 1
        remaining = N - desired_train
        desired_test = min(desired_test, remaining)
        remaining = remaining - desired_test
        desired_val = min(desired_val, remaining)

        min_train = min(min_train, desired_train)
        min_test = min(min_test, desired_test)
        min_val = min(min_val, desired_val)

        logging.warning(
            "Adaptive sizing engaged due to limited N_patients. "
            f"Relaxed mins -> train>={min_train}, val>={min_val}, test>={min_test} (N={N})."
        )

    # Always keep at least 1 per split if N allows
    if N >= 3:
        min_train = max(1, min_train)
        min_val = max(1, min_val)
        min_test = max(1, min_test)

    return min_train, min_val, min_test


def decide_split_sizes(patient_df: pd.DataFrame, constraints: SplitConstraints) -> Tuple[int, int, int, Dict]:
    """
    Returns (n_train, n_val, n_test, sizing_meta).
    """
    N = len(patient_df)
    min_train, min_val, min_test = _adaptive_min_patients(N, constraints)

    n_test = max(min_test, int(round(constraints.test_ratio * N)))
    n_val = max(min_val, int(round(constraints.val_ratio * N)))
    n_train = N - n_test - n_val

    # Repair if train falls below min_train (shrink val first, then test if adaptive)
    if n_train < min_train:
        n_val = max(min_val, N - n_test - min_train)
        n_train = N - n_test - n_val

    if constraints.adaptive and n_train < min_train:
        # if still impossible, shrink test as well
        n_test = max(min_test, N - n_val - min_train)
        n_train = N - n_test - n_val

    if n_train < min_train or n_val < min_val or n_test < min_test:
        raise ValueError(
            f"Split impossible after sizing: train={n_train}, val={n_val}, test={n_test}. "
            f"mins: train>={min_train}, val>={min_val}, test>={min_test}. "
            "Need more patients or relax constraints."
        )

    meta = dict(
        N_patients=N,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        min_train=min_train,
        min_val=min_val,
        min_test=min_test,
        test_ratio=constraints.test_ratio,
        val_ratio=constraints.val_ratio,
        adaptive=constraints.adaptive,
    )
    return n_train, n_val, n_test, meta


# -----------------------------------------------------------------------------
# Entropy cache (patch -> entropy), then patient entropies
# -----------------------------------------------------------------------------

def compute_all_patch_entropies(
    df: pd.DataFrame,
    num_workers: int,
    chunksize: int,
    entropy_thumbnail: int = 128,  # NEW
) -> pd.DataFrame:
    image_paths = df["image_path"].tolist()

    logging.info(
        f"Computing patch entropies for {len(image_paths)} images "
        f"using num_workers={num_workers}, thumb={entropy_thumbnail}..."
    )

    worker = partial(calculate_image_entropy_from_path, thumb=entropy_thumbnail)

    if num_workers <= 1:
        out = [worker(p) for p in tqdm(image_paths, desc="Entropy", leave=False)]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            out = list(
                tqdm(
                    pool.imap(worker, image_paths, chunksize=chunksize),
                    total=len(image_paths),
                    desc="Entropy",
                    leave=False,
                )
            )

    ent_df = pd.DataFrame(out, columns=["image_path", "entropy"])
    return ent_df


def compute_patient_entropy_median(
    df: pd.DataFrame,
    entropy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins entropy into the patch dataframe, then:
      patient_entropy = median(entropy across ALL patches of that patient)
    Returns one row per patient: [patient_id, patient_entropy]
    """
    merged = df.merge(entropy_df, on="image_path", how="left")
    merged["entropy"] = merged["entropy"].fillna(0.0)

    patient_entropy = (
        merged.groupby("patient_id")["entropy"]
              .median()
              .reset_index(name="patient_entropy_median")
    )
    return patient_entropy


def score_split_by_patient_entropy_median(
    patient_entropy_df: pd.DataFrame,
    patient_ids: List[int],
) -> float:
    """
    split_score = median(patient_entropy_median over patients in the split)
    """
    sub = patient_entropy_df[patient_entropy_df["patient_id"].isin(patient_ids)]
    if sub.empty:
        return float("-inf")
    return float(sub["patient_entropy_median"].median())


# -----------------------------------------------------------------------------
# Split generation: evaluate multiple feasible splits, pick best
# -----------------------------------------------------------------------------

def create_train_val_test_split_best(
    df: pd.DataFrame,
    random_state: int,
    constraints: SplitConstraints,
    objective: ObjectiveConfig,
    patient_entropy_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Patient-level stratified split search:
      - Generate up to max_tries candidate splits (random seeds derived from random_state)
      - Enforce constraints
      - If objective.enable_objective=True:
          score candidates and pick best (max or min)
        Else:
          return first feasible (classic behavior)

    Stratification label: patient_label = max(patch_label)
    """
    patient_df = build_patient_table(df)
    N = len(patient_df)

    # Split sizing (adaptive option here)
    n_train, n_val, n_test, sizing_meta = decide_split_sizes(patient_df, constraints)

    patient_ids = patient_df["patient_id"].to_numpy()
    y = patient_df["patient_label"].astype(int).to_numpy()

    # Determine if "both classes" constraint is meaningful (dataset might be single-class)
    dataset_has_both_classes = (patient_df["patient_label"].nunique() >= 2)

    rng = np.random.default_rng(random_state)

    def image_count(pid_set: set) -> int:
        return int(patient_df[patient_df["patient_id"].isin(pid_set)]["n_images"].sum())

    # Tracking failures for auditability
    fail_val_strat = 0
    fail_overlap = 0
    fail_min_patients = 0
    fail_train_dominance = 0
    fail_class_coverage = 0

    best = None
    best_score = None

    # We will store some feasible candidates if objective enabled (optional)
    feasible_found = 0

    for attempt in range(constraints.max_tries):
        seed = int(rng.integers(0, 2**31 - 1))

        # 1) TEST split from all patients
        try:
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
            trainval_idx, test_idx = next(sss_test.split(patient_ids, y))
        except ValueError as e:
            # If stratification is impossible (e.g., too few minority patients), this is structural
            raise ValueError(
                f"Stratified split impossible at patient level for TEST (test_size={n_test}). "
                f"Details: {e}"
            )

        tv_ids = patient_ids[trainval_idx]
        tv_y = y[trainval_idx]

        # 2) VAL from remaining
        try:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed + 1)
            train_idx, val_idx = next(sss_val.split(tv_ids, tv_y))
        except ValueError:
            fail_val_strat += 1
            continue

        train_patients = set(tv_ids[train_idx])
        val_patients = set(tv_ids[val_idx])
        test_patients = set(patient_ids[test_idx])

        # No leakage
        if (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients):
            fail_overlap += 1
            continue

        # Minimum patient counts
        if len(train_patients) < sizing_meta["min_train"] or len(val_patients) < sizing_meta["min_val"] or len(test_patients) < sizing_meta["min_test"]:
            fail_min_patients += 1
            continue

        # Optional: class coverage guardrail (only if dataset has both classes)
        if constraints.require_both_classes_if_possible and dataset_has_both_classes:
            def split_has_both(pids: set) -> bool:
                labs = set(patient_df[patient_df["patient_id"].isin(pids)]["patient_label"].tolist())
                return (0 in labs) and (1 in labs)

            # If feasible, enforce for all splits
            if not (split_has_both(train_patients) and split_has_both(val_patients) and split_has_both(test_patients)):
                fail_class_coverage += 1
                continue

        # Train image dominance rule (your current Rule 2)
        train_imgs = image_count(train_patients)
        val_imgs = image_count(val_patients)
        test_imgs = image_count(test_patients)

        if constraints.require_train_image_dominance:
            if not (train_imgs > val_imgs and train_imgs > test_imgs):
                fail_train_dominance += 1
                continue

        feasible_found += 1

        # If objective is off, return first feasible (backward compatible behavior)
        if not objective.enable_objective:
            return _build_split_return(
                df=df,
                patient_df=patient_df,
                train_patients=train_patients,
                val_patients=val_patients,
                test_patients=test_patients,
                seed=seed,
                attempt=attempt + 1,
                constraints=constraints,
                sizing_meta=sizing_meta,
                score=None,
                score_split=objective.score_split,
            )

        # Objective enabled: score candidate
        if patient_entropy_df is None:
            raise ValueError("Objective enabled but patient_entropy_df is None.")

        score_split = objective.score_split.upper()
        if score_split == "TRAIN":
            score_ids = sorted(train_patients)
        elif score_split == "VALIDATION":
            score_ids = sorted(val_patients)
        elif score_split == "TEST":
            score_ids = sorted(test_patients)
        else:
            raise ValueError(f"Invalid objective.score_split: {objective.score_split}")

        score = score_split_by_patient_entropy_median(patient_entropy_df, score_ids)

        if best is None:
            best = (train_patients, val_patients, test_patients, seed, attempt + 1)
            best_score = score
        else:
            if objective.maximize and score > best_score:
                best = (train_patients, val_patients, test_patients, seed, attempt + 1)
                best_score = score
            if (not objective.maximize) and score < best_score:
                best = (train_patients, val_patients, test_patients, seed, attempt + 1)
                best_score = score

    # If we got here, either no feasible splits OR objective on but none feasible
    if best is not None:
        train_patients, val_patients, test_patients, seed, attempt_no = best
        logging.info(
            f"Best feasible split selected among {feasible_found} feasible candidates "
            f"(searched {constraints.max_tries} attempts). Best_score={best_score:.6f}"
        )
        return _build_split_return(
            df=df,
            patient_df=patient_df,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            seed=seed,
            attempt=attempt_no,
            constraints=constraints,
            sizing_meta=sizing_meta,
            score=float(best_score),
            score_split=objective.score_split,
        )

    msg = (
        "Split search failed under configured constraints.\n"
        f"Tried {constraints.max_tries} randomized stratified attempts (random_state={random_state}).\n"
        "Failure breakdown:\n"
        f"  - VAL stratification failed: {fail_val_strat}\n"
        f"  - Patient leakage overlap: {fail_overlap}\n"
        f"  - Minimum patient counts failed: {fail_min_patients}\n"
        f"  - Class coverage failed: {fail_class_coverage}\n"
        f"  - Train image dominance failed: {fail_train_dominance}\n"
        "Action: add patients (especially minority class), relax constraints, or disable class-coverage.\n"
    )
    raise ValueError(msg)


def _build_split_return(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    train_patients: set,
    val_patients: set,
    test_patients: set,
    seed: int,
    attempt: int,
    constraints: SplitConstraints,
    sizing_meta: Dict,
    score: Optional[float],
    score_split: str,
) -> Dict:
    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

    logging.info(
        f"Split OK (attempt {attempt}/{constraints.max_tries}, seed={seed}): "
        f"patients train/val/test={len(train_patients)}/{len(val_patients)}/{len(test_patients)} | "
        f"images train/val/test="
        f"{int(patient_df[patient_df['patient_id'].isin(train_patients)]['n_images'].sum())}/"
        f"{int(patient_df[patient_df['patient_id'].isin(val_patients)]['n_images'].sum())}/"
        f"{int(patient_df[patient_df['patient_id'].isin(test_patients)]['n_images'].sum())}"
        + (f" | objective({score_split})={score:.6f}" if score is not None else "")
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "split_seed": seed,
        "split_attempt": attempt,
        "objective_score": score,
        "objective_score_split": score_split,
        "constraints": {
            **asdict(constraints),
            **sizing_meta,
            "random_state": None,  # filled by caller in write_manifest
        },
    }


# -----------------------------------------------------------------------------
# Normalization (unchanged logic)
# -----------------------------------------------------------------------------

def make_aggregate_target(image_paths: List[str]) -> np.ndarray:
    images_rgb = []
    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        images_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not images_rgb:
        raise ValueError("No valid images found to build aggregate target.")
    stack = np.stack(images_rgb, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def fit_normalizer_on_train_set(
    train_df: pd.DataFrame,
    method_name: str,
    entropy_df: Optional[pd.DataFrame] = None,
):
    """
    Fits a stain normalizer on TRAIN only, using one high-entropy image per TRAIN patient,
    and returns (normalizer, template_paths).

    If entropy_df is provided (columns: [image_path, entropy]), template selection is done
    WITHOUT re-reading images from disk (fast). Otherwise, falls back to per-image reads.
    """
    logging.info(f"Fitting '{method_name}' normalizer using TRAIN only...")

    if train_df is None or train_df.empty:
        raise ValueError("TRAIN dataframe is empty; cannot fit normalizer.")

    # --- FAST PATH: reuse cached entropy_df (no cv2.imread during selection) ---
    if entropy_df is not None and not entropy_df.empty:
        logging.info("Selecting templates using cached entropy_df (no disk rereads).")

        # Join entropy into TRAIN patch table
        merged = train_df.merge(entropy_df, on="image_path", how="left")
        merged["entropy"] = merged["entropy"].fillna(0.0)

        # For each patient, take the image_path with maximum entropy
        idx = merged.groupby("patient_id")["entropy"].idxmax()
        template_paths = merged.loc[idx, "image_path"].dropna().tolist()

    else:
        # --- SLOW FALLBACK: compute entropy by reading images again ---
        logging.warning(
            "entropy_df not provided; selecting templates by rereading images (slow)."
        )
        patient_files = train_df.groupby("patient_id")["image_path"].apply(list).to_dict()
        if not patient_files:
            raise ValueError("No patient images found in TRAIN to fit the normalizer.")

        template_paths = []
        for pid, files in tqdm(patient_files.items(), desc="Selecting templates", leave=False):
            best_p = None
            best_e = -1.0
            for p in files:
                _, e = calculate_image_entropy_from_path(p)
                if e > best_e:
                    best_e = e
                    best_p = p
            if best_p is not None:
                template_paths.append(best_p)

    if not template_paths:
        raise ValueError("No templates found to fit normalizer.")

    logging.info(f"Creating aggregate target from {len(template_paths)} templates...")
    target_rgb = make_aggregate_target(template_paths)

    normalizer = stainnorm.get_normalizer(method_name)
    normalizer.fit(target_rgb)
    logging.info(f"Normalizer '{method_name}' fitted.")
    return normalizer, template_paths


def save_normalizer_stats(normalizer, method_name: str, output_dir: str, template_paths: List[str]) -> None:
    stats = {"method": method_name}

    try:
        if isinstance(normalizer, stainnorm.StainNormalizer):
            if hasattr(normalizer, "stain_matrix_target"):
                stats["stain_matrix_target"] = normalizer.stain_matrix_target.tolist()
            if hasattr(normalizer, "maxC_target"):
                stats["maxC_target"] = normalizer.maxC_target.tolist()
            if method_name == "MACENKO" and hasattr(normalizer.extractor, "stains"):
                stats["stain_vectors_source_estimate"] = normalizer.extractor.stains.tolist()

        elif isinstance(normalizer, stainnorm.ReinhardNormalizer):
            stats["target_means"] = list(normalizer.target_means)
            stats["target_stds"] = list(normalizer.target_stds)
        else:
            stats["info"] = "Unknown or unsupported normalizer type."
            logging.warning(f"Unrecognized normalizer type for '{method_name}'.")

    except Exception as e:
        stats["error"] = str(e)
        logging.error(f"Failed extracting normalizer stats: {e}", exc_info=True)

    out_json = os.path.join(output_dir, "normalization_stats.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    logging.info(f"Normalization stats saved: {out_json}")

    template_dir = os.path.join(output_dir, "normalization_templates")
    os.makedirs(template_dir, exist_ok=True)
    for i, src in enumerate(template_paths):
        dst = os.path.join(template_dir, f"template_{i:03d}_{os.path.basename(src)}")
        shutil.copy(src, dst)
    logging.info(f"Saved {len(template_paths)} template images: {template_dir}")


# -----------------------------------------------------------------------------
# File I/O: normalize images (optional) + copy masks
# -----------------------------------------------------------------------------

def process_and_write_image(src_path: str, dest_path: str, normalizer) -> Tuple[bool, Optional[str]]:
    """
    Non-destructive: reads image, optionally normalizes, writes to dest_path.
    Used for normalized pipelines or when you explicitly want a copy.
    """
    try:
        image_bgr = cv2.imread(src_path)
        if image_bgr is None:
            raise IOError(f"Could not read image: {src_path}")

        if normalizer is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            normalized_rgb = normalizer.transform(image_rgb)
            out_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = image_bgr

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        ok = cv2.imwrite(dest_path, out_bgr)
        if not ok:
            raise IOError(f"Failed writing to: {dest_path}")

        return True, None
    except Exception as e:
        return False, f"{os.path.basename(src_path)}: {e}"


_MOVE_FALLBACKS = 0

def move_file(src_path: str, dest_path: str) -> Tuple[bool, Optional[str]]:
    global _MOVE_FALLBACKS
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            os.replace(src_path, dest_path)
            return True, None
        except OSError:
            _MOVE_FALLBACKS += 1
            shutil.move(src_path, dest_path)
            return True, None
    except Exception as e:
        return False, f"{os.path.basename(src_path)}: {e}"


def process_and_write_split_files(
    split_df: pd.DataFrame,
    output_dir: str,
    split_name: str,
    normalizer,
    normalization_method: str,
) -> None:
    """
    Streaming IO writer:
      - avoids building huge task lists
      - avoids creating 1 Future per file
      - processes 1 row (image+mask) per call

    If NOT_NORMALIZED: MOVE images+masks (destructive)
    Else: write (optionally normalized) images and copy masks (non-destructive)
    """
    if split_df is None or split_df.empty:
        logging.info(f"Skipping empty split: {split_name}")
        return

    is_not_normalized = (normalization_method == "NOT_NORMALIZED")
    if is_not_normalized:
        logging.warning(
            f"[DESTRUCTIVE MODE] NOT_NORMALIZED => moving files into split folders for {split_name}. "
            "Original dataset folders will be modified."
        )

    logging.info(f"Writing {len(split_df)} items for {split_name}...")

    max_workers = min(32, (os.cpu_count() or 1) + 4)

    # Important: itertuples is much faster than iterrows
    rows_iter = split_df.itertuples(index=False)

    def _process_one(row) -> Tuple[bool, Optional[str]]:
        """
        Worker: process a single patch row (image+mask).
        Returns (ok, error_msg).
        """
        try:
            label_dir = "CANCER" if int(row.label) == 1 else "NOT_CANCER"

            img_dst = os.path.join(output_dir, split_name, label_dir, row.filename)
            msk_dst = os.path.join(output_dir, split_name, f"{label_dir}_MASK", row.filename)

            # Ensure dirs exist (cheap if already exists)
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(msk_dst), exist_ok=True)

            if is_not_normalized:
                ok, msg = move_file(row.image_path, img_dst)
                if not ok:
                    return False, f"IMG move failed | {msg}"
                ok, msg = move_file(row.mask_path, msk_dst)
                if not ok:
                    return False, f"MSK move failed | {msg}"
                return True, None

            # Normalized path: write image, copy mask
            ok, msg = process_and_write_image(row.image_path, img_dst, normalizer)
            if not ok:
                return False, f"IMG write failed | {msg}"

            shutil.copy2(row.mask_path, msk_dst)
            return True, None

        except Exception as e:
            # Keep error small but useful
            return False, f"{row.filename}: {e}"

    errors: List[str] = []
    import concurrent.futures

    # executor.map streams work items without creating a Future per file explicitly
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # chunksize affects batching to workers; higher reduces overhead (esp. Windows)
        results_iter = ex.map(_process_one, rows_iter, chunksize=256)

        for ok, msg in tqdm(results_iter, total=len(split_df), desc=f"Write {split_name}", leave=False):
            if not ok and msg:
                errors.append(msg)

    if errors:
        for e in errors[:30]:
            logging.error(f"Write error: {e}")
        raise RuntimeError(f"Failed to process {len(errors)} items for split {split_name}.")



def verify_split_integrity(output_dir: str, split_name: str) -> None:
    split_dir = os.path.join(output_dir, split_name)
    if not os.path.isdir(split_dir):
        logging.warning(f"Split dir missing (skip integrity): {split_dir}")
        return

    for label_name in ["CANCER", "NOT_CANCER"]:
        img_dir = os.path.join(split_dir, label_name)
        msk_dir = os.path.join(split_dir, f"{label_name}_MASK")
        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            continue

        img_files = {f for f in os.listdir(img_dir) if f.lower().endswith(".png")}
        msk_files = {f for f in os.listdir(msk_dir) if f.lower().endswith(".png")}
        if img_files != msk_files:
            raise ValueError(f"Integrity FAILED for {split_name}/{label_name}: image/mask filenames differ.")

    logging.info(f"Integrity PASSED: {split_name}")


# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------

def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def fill_manifest_checksums_inplace(manifest_df: pd.DataFrame, num_workers: int = 8) -> pd.DataFrame:
    """
    Compute sha256 for destination files listed in manifest_df['abs_image_path'] / ['abs_mask_path'].
    Parallelized. Still expensive for 700k+ files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def hash_pair(img_path: str, msk_path: str) -> Tuple[str, str]:
        sha_img = _sha256_file(img_path) if os.path.isfile(img_path) else None
        sha_msk = _sha256_file(msk_path) if os.path.isfile(msk_path) else None
        return sha_img, sha_msk

    img_paths = manifest_df["abs_image_path"].tolist()
    msk_paths = manifest_df["abs_mask_path"].tolist()

    sha_img_out = [None] * len(manifest_df)
    sha_msk_out = [None] * len(manifest_df)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(hash_pair, img_paths[i], msk_paths[i]): i for i in range(len(manifest_df))}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="SHA256", leave=False):
            i = futures[fut]
            try:
                sha_img, sha_msk = fut.result()
                sha_img_out[i] = sha_img
                sha_msk_out[i] = sha_msk
            except Exception:
                # keep None; caller can decide to fail hard if desired
                pass

    manifest_df["sha256_image"] = sha_img_out
    manifest_df["sha256_mask"] = sha_msk_out
    return manifest_df

def _get_git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def _collect_library_versions() -> Dict:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "opencv": getattr(cv2, "__version__", None),
    }
    try:
        import sklearn
        versions["sklearn"] = getattr(sklearn, "__version__", None)
    except Exception:
        versions["sklearn"] = None

    try:
        import tiatoolbox
        versions["tiatoolbox"] = getattr(tiatoolbox, "__version__", None)
    except Exception:
        versions["tiatoolbox"] = None

    return versions


def build_manifest_from_split_dfs(
    output_dir: str,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    split_data: Dict,
) -> pd.DataFrame:
    """
    Builds manifest rows from split_data DataFrames directly.
    No filesystem scanning. No os.listdir. Deterministic and fast.
    """

    manifest_rows = []

    def add_rows(split_name: str, sdf: pd.DataFrame) -> None:
        if sdf is None or sdf.empty:
            return

        for _, row in sdf.iterrows():
            label_dir = "CANCER" if int(row["label"]) == 1 else "NOT_CANCER"

            # Destination (what the script writes/moves to)
            rel_img = os.path.join(split_name, label_dir, row["filename"]).replace("\\", "/")
            rel_msk = os.path.join(split_name, f"{label_dir}_MASK", row["filename"]).replace("\\", "/")

            abs_img = os.path.join(output_dir, split_name, label_dir, row["filename"])
            abs_msk = os.path.join(output_dir, split_name, f"{label_dir}_MASK", row["filename"])

            manifest_rows.append(
                dict(
                    run_id=run_id,
                    split=split_name,
                    label=int(row["label"]),
                    patient_id=int(row["patient_id"]),
                    filename=row["filename"],
                    normalization_method=normalization_method,
                    is_normalized=bool(is_normalized),
                    relative_path_image=rel_img,
                    relative_path_mask=rel_msk,
                    # keep source paths for traceability/debug (optional but useful)
                    source_image_path=row["image_path"],
                    source_mask_path=row["mask_path"],
                    # filled later only if requested
                    sha256_image=None,
                    sha256_mask=None,
                    # useful for quick existence checks
                    abs_image_path=abs_img,
                    abs_mask_path=abs_msk,
                )
            )

    add_rows("TRAIN", split_data["train_df"])
    add_rows("VALIDATION", split_data["val_df"])
    add_rows("TEST", split_data["test_df"])

    manifest_df = pd.DataFrame(manifest_rows)

    # Make ordering stable (useful for diffs)
    manifest_df = manifest_df.sort_values(
        by=["split", "label", "patient_id", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)

    return manifest_df

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        # numpy scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def write_manifest_and_log_stats(
    output_dir: str,
    run_id: str,
    normalization_method: str,
    is_normalized: bool,
    data_directory: str,
    split_data: Dict,
    manifest_df: pd.DataFrame,
    calc_checksums: bool = True,
    extra: Optional[Dict] = None,
) -> None:
    logging.info("Writing manifest.csv, split_stats.csv, run_config.json ...")

    # --- manifest.csv (NO DIRECTORY SCAN) ---
    if calc_checksums:
        # Warning: this is still expensive for 700k+ files.
        logging.warning("calc_checksums=True: computing SHA256 for all files (can take hours).")
        manifest_df = fill_manifest_checksums_inplace(
            manifest_df,
            num_workers=min(16, (os.cpu_count() or 1)),
        )

    # Drop absolute paths from final manifest if you prefer a cleaner artifact
    manifest_out = manifest_df.drop(columns=["abs_image_path", "abs_mask_path"], errors="ignore")

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_out.to_csv(manifest_path, index=False)
    logging.info(f"Manifest written: {manifest_path} (rows={len(manifest_out)})")

    # --- split_stats.csv (unchanged, computed from split_data dfs) ---
    stats_rows = []
    for split_name, sdf in [("TRAIN", split_data["train_df"]), ("VALIDATION", split_data["val_df"]), ("TEST", split_data["test_df"])]:
        if sdf is None or sdf.empty:
            continue

        patient_counts = (
            sdf.groupby(["patient_id", "label"])
              .size()
              .reset_index(name="n_images_patient")
        )

        n_patients = int(patient_counts["patient_id"].nunique())
        n_images = int(len(sdf))
        n_pos_patients = int(patient_counts[patient_counts["label"] == 1]["patient_id"].nunique())
        n_neg_patients = int(patient_counts[patient_counts["label"] == 0]["patient_id"].nunique())

        per_patient = patient_counts.groupby("patient_id")["n_images_patient"].sum()

        stats = dict(
            run_id=run_id,
            split=split_name,
            n_patients=n_patients,
            n_pos_patients=n_pos_patients,
            n_neg_patients=n_neg_patients,
            n_images=n_images,
            patches_per_patient_mean=float(per_patient.mean()),
            patches_per_patient_std=float(per_patient.std(ddof=1)) if len(per_patient) > 1 else 0.0,
            patches_per_patient_min=int(per_patient.min()),
            patches_per_patient_q1=float(per_patient.quantile(0.25)),
            patches_per_patient_median=float(per_patient.quantile(0.50)),
            patches_per_patient_q3=float(per_patient.quantile(0.75)),
            patches_per_patient_max=int(per_patient.max()),
        )

        per_class_images = sdf.groupby("label").size().to_dict()
        stats["n_images_neg"] = int(per_class_images.get(0, 0))
        stats["n_images_pos"] = int(per_class_images.get(1, 0))
        stats_rows.append(stats)

    split_stats_df = pd.DataFrame(stats_rows)
    split_stats_path = os.path.join(output_dir, "split_stats.csv")
    split_stats_df.to_csv(split_stats_path, index=False)
    logging.info(f"Split stats written: {split_stats_path}")

    # --- run_config.json (unchanged; but keep your SafeJSONEncoder fix!) ---
    run_cfg = dict(
        run_id=run_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        data_directory=os.path.normpath(data_directory),
        output_dir=os.path.normpath(output_dir),
        normalization_method=normalization_method,
        is_normalized=bool(is_normalized),
        constraints=split_data.get("constraints", {}),
        split_seed=split_data.get("split_seed", None),
        split_attempt=split_data.get("split_attempt", None),
        objective_score=split_data.get("objective_score", None),
        objective_score_split=split_data.get("objective_score_split", None),
        patients=dict(
            train=[int(x) for x in split_data.get("train_patients", [])],
            validation=[int(x) for x in split_data.get("val_patients", [])],
            test=[int(x) for x in split_data.get("test_patients", [])],
        ),
        library_versions=_collect_library_versions(),
        git_commit=_get_git_commit_hash(),
        extra=extra or {},
    )

    cfg_path = os.path.join(output_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, cls=SafeJSONEncoder)  # <- keep this fix
    logging.info(f"Run config written: {cfg_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg: RunConfig) -> None:
    valid_methods = ["NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"]
    if cfg.normalization_method not in valid_methods:
        raise ValueError(f"Invalid normalization_method={cfg.normalization_method}. Must be one of {valid_methods}")

    data_dir = os.path.normpath(cfg.data_directory)
    output_base_dir = os.path.join(data_dir, cfg.normalization_method)
    output_run_dir = os.path.join(output_base_dir, f"{cfg.normalization_method}_seed_{cfg.random_state}")

    if os.path.exists(output_run_dir):
        if cfg.overwrite_output_dir:
            shutil.rmtree(output_run_dir)
        else:
            raise RuntimeError(f"Output directory exists: {output_run_dir} (set overwrite_output_dir=True)")

    os.makedirs(output_run_dir, exist_ok=True)
    setup_logging(output_run_dir)

    logging.info("=== Data Preparation (Refactored) ===")
    logging.info(f"Normalization method: {cfg.normalization_method}")
    logging.info(f"Random state: {cfg.random_state}")
    logging.info(f"Constraints: {asdict(cfg.constraints)}")
    logging.info(f"Objective: {asdict(cfg.objective)}")
    logging.info(f"Output: {output_run_dir}")

    # 1) Load data
    df = load_data(data_dir)

    # 2) Compute entropy caches if objective enabled
    entropy_df = None
    patient_entropy_df = None
    entropy_cache_path = os.path.join(output_run_dir, "entropy_cache.csv")

    if cfg.objective.enable_objective:
        entropy_df = compute_all_patch_entropies(
            df=df,
            num_workers=cfg.objective.num_workers,
            chunksize=cfg.objective.chunksize,
            entropy_thumbnail=cfg.objective.entropy_thumbnail,
        )
        patient_entropy_df = compute_patient_entropy_median(df, entropy_df)

        if cfg.save_entropy_cache_csv:
            entropy_df.to_csv(entropy_cache_path, index=False)
            patient_entropy_df.to_csv(os.path.join(output_run_dir, "patient_entropy_median.csv"), index=False)
            logging.info(f"Saved entropy cache: {entropy_cache_path}")

    # 3) Split search: pick best feasible split (or first feasible if objective disabled)
    split_data = create_train_val_test_split_best(
        df=df,
        random_state=cfg.random_state,
        constraints=cfg.constraints,
        objective=cfg.objective,
        patient_entropy_df=patient_entropy_df,
    )

    # Store random_state in constraints for provenance
    split_data["constraints"]["random_state"] = cfg.random_state

    # 3.5) Build manifest from split DataFrames (FAST: no directory scanning)
    manifest_df = build_manifest_from_split_dfs(
        output_dir=output_run_dir,
        run_id=f"{cfg.normalization_method}_seed_{cfg.random_state}",
        normalization_method=cfg.normalization_method,
        is_normalized=(cfg.normalization_method != "NOT_NORMALIZED"),
        split_data=split_data,
    )

    # 4) Create output dirs
    for split_name in ["TRAIN", "VALIDATION", "TEST"]:
        for label in ["CANCER", "NOT_CANCER"]:
            os.makedirs(os.path.join(output_run_dir, split_name, label), exist_ok=True)
            os.makedirs(os.path.join(output_run_dir, split_name, f"{label}_MASK"), exist_ok=True)

    # 5) Fit normalizer on TRAIN only
    normalizer = None
    template_paths = None
    if cfg.normalization_method != "NOT_NORMALIZED":
        normalizer, template_paths = fit_normalizer_on_train_set(
                                            split_data["train_df"],
                                            cfg.normalization_method,
                                            entropy_df=entropy_df,   # <- NEW
                                        )
        save_normalizer_stats(normalizer, cfg.normalization_method, output_run_dir, template_paths)

    # 6) Write files (normalize images if enabled; masks copied raw)
    split_dfs = {"TRAIN": split_data["train_df"], "VALIDATION": split_data["val_df"], "TEST": split_data["test_df"]}
    for split_name, sdf in split_dfs.items():
        process_and_write_split_files(
                                    sdf,
                                    output_run_dir,
                                    split_name,
                                    normalizer,
                                    normalization_method=cfg.normalization_method,
                                )
        verify_split_integrity(output_run_dir, split_name)
    
    logging.info(f"Move fallbacks (copy+delete likely): {_MOVE_FALLBACKS}")

    # 7) Manifest + stats + run config
    extra = {}
    if patient_entropy_df is not None and split_data.get("objective_score") is not None:
        extra["objective_definition"] = {
            "metric": cfg.objective.metric,
            "patient_entropy": "median(patch_entropy)",
            "split_score": f"median(patient_entropy) over {cfg.objective.score_split.upper()} patients",
            "maximize": cfg.objective.maximize,
        }

    write_manifest_and_log_stats(
        output_dir=output_run_dir,
        run_id=f"{cfg.normalization_method}_seed_{cfg.random_state}",
        normalization_method=cfg.normalization_method,
        is_normalized=(cfg.normalization_method != "NOT_NORMALIZED"),
        data_directory=data_dir,
        split_data=split_data,
        manifest_df=manifest_df,           # <-- ADD THIS
        calc_checksums=cfg.calc_checksums,
        extra=extra,
    )

    logging.info("=== DONE ===")


if __name__ == "__main__":
    cfg = RunConfig(
        normalization_method="NOT_NORMALIZED",
        data_directory=r"D:\Usuario\Desktop\Base_de_dados\CAMELYON16\PATCHES",
        overwrite_output_dir=True,
        random_state=42,
        constraints=SplitConstraints(
            min_test_patients=20,
            min_val_patients=5,
            min_train_patients=5,
            test_ratio=0.10,
            val_ratio=0.10,
            require_train_image_dominance=True,
            require_both_classes_if_possible=True,
            max_tries=1000,
            adaptive=True,  # key: works across datasets
        ),
        objective=ObjectiveConfig(
            enable_objective=True,
            score_split="TRAIN",
            maximize=True,
            num_workers=max(1, (os.cpu_count() or 1) - 1),
            chunksize=128,
        ),
        calc_checksums=False,
        save_entropy_cache_csv=True,
    )
    main(cfg)
