# -----------------------------------------------------------------------------
# UNIFIED DATA PREPARATION PIPELINE (SIMPLIFIED)
# -----------------------------------------------------------------------------
# This script integrates stain normalization and data splitting for deep learning.
#
# Scientific Methodology:
# 1.  Creates a single, reproducible 80% TRAIN, 10% VALIDATION, 10% TEST split
#     at the patient level to prevent data leakage. The split is controlled by a
#     single RANDOM_STATE seed.
# 2.  Stain normalization is applied according to the selected method.
# 3.  CRITICAL: Normalization statistics are calculated *ONLY* from high-entropy
#     images within the TRAIN set.
# 4.  The fitted normalizer is then used to transform all three sets (TRAIN,
#     VALIDATION, TEST) for consistency without leakage.
# 5.  The TRAIN set is balanced via augmentation on the normalized images.
# 6.  A manifest and statistics file are generated for full traceability.
#
# Configuration is done in the `if __name__ == '__main__':` block.
# -----------------------------------------------------------------------------

import os
import shutil
import re
import concurrent.futures
import itertools
import logging
import random
import sys
import json
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tiatoolbox.tools import stainnorm
from tqdm import tqdm

# --- Logger Setup ---
def setup_logging(output_dir):
    """Configures the root logger to output to both a file and the console."""
    log_file = os.path.join(output_dir, 'data_preparation.log')
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    logging.info(f"Logging configured. Output will be saved to {log_file}")

# --- Helper Functions (Unchanged) ---
def seed_worker(seed):
    random.seed(seed)
    np.random.seed(seed)

def calculate_image_entropy(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        if hist.sum() == 0: return 0.0
        prob_dist = hist / hist.sum()
        entropy = -np.sum(prob_dist[prob_dist > 0] * np.log2(prob_dist[prob_dist > 0]))
        return entropy
    except Exception as e:
        logging.warning(f"Could not calculate entropy for {image_path}: {e}")
        return 0.0

# --- Data Loading (Unchanged) ---
def load_data(data_dir):
    """Loads image and mask data deterministically and robustly, returning a DataFrame."""
    logging.info(f"Loading data from: {data_dir}")
    data = []
    image_base_dir = os.path.join(data_dir, "images")
    mask_base_dir = os.path.join(data_dir, "masks")

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(image_base_dir, label_name)
        mask_dir = os.path.join(mask_base_dir, label_name)
        label = 1 if label_name == "CANCER" else 0
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            logging.warning(f"Missing directories for {label_name}. Skipping.")
            continue
        
        logging.info(f"Scanning {label_name} images...")
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        mask_files  = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        
        for image_name in tqdm(image_files, desc=f"Processing {label_name} files"):
            if image_name not in mask_files:
                logging.warning(f"Mask not found for image '{image_name}'. Skipping.")
                continue
            match = re.search(r'PATIENT_(\d+)_', image_name)
            if not match:
                logging.warning(f"Could not extract patient ID from {image_name}. Skipping.")
                continue
            
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)
            if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
                logging.warning(f"File path(s) invalid for '{image_name}'. Skipping.")
                continue
            if cv2.imread(image_path) is None or cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) is None:
                logging.warning(f"Could not read image/mask file '{image_path}'. Skipping.")
                continue

            data.append({
                'patient_id': int(match.group(1)),
                'image_path': image_path,
                'mask_path': mask_path,
                'label': label,
                'filename': image_name
            })
            
    if not data:
        raise ValueError("No valid, readable image/mask pairs were found.")
    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} image/mask pairs for {df['patient_id'].nunique()} patients.")
    return df

# <<< CHANGE: This function is now much simpler. ---
def create_train_val_test_split(df, random_state=42):
    """
    Creates a single, stratified, patient-aware 80/10/10 split.
    """
    logging.info(f"--- Creating a single 80% TRAIN / 10% VAL / 10% TEST split with seed {random_state} ---")
    patient_df = df.groupby('patient_id')['label'].max().reset_index()
    
    # We use a 10-fold splitter to get 10% chunks easily.
    # 8 folds for train, 1 for val, 1 for test.
    sgkf_master = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    try:
        # First split gets the 10% test set
        train_val_idx, test_idx = next(sgkf_master.split(patient_df, y=patient_df['label'], groups=patient_df['patient_id']))
        
        # Sub-split the remaining 90% to get train (8 parts) and validation (1 part)
        train_val_patient_df = patient_df.iloc[train_val_idx]
        sgkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=random_state)
        train_idx_inner, val_idx_inner = next(sgkf_inner.split(train_val_patient_df, y=train_val_patient_df['label'], groups=train_val_patient_df['patient_id']))
        
        # Get original dataframe indices
        train_idx = train_val_patient_df.index[train_idx_inner]
        val_idx = train_val_patient_df.index[val_idx_inner]
    except (StopIteration, ValueError) as e:
        logging.error(f"Could not generate the data split: {e}.")
        return None

    train_patients = set(patient_df.iloc[train_idx]['patient_id'])
    val_patients = set(patient_df.iloc[val_idx]['patient_id'])
    test_patients = set(patient_df.iloc[test_idx]['patient_id'])

    # Ensure no patient overlap between splits (the core scientific safeguard)
    assert train_patients.isdisjoint(val_patients) and train_patients.isdisjoint(test_patients) and val_patients.isdisjoint(test_patients)
    
    logging.info(f"Split created: Train patients={len(train_patients)}, Val patients={len(val_patients)}, Test patients={len(test_patients)}")
    
    # Create the final dataframes
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)
    
    return {'train_df': train_df, 'val_df': val_df, 'test_df': test_df}

# --- Normalization Functions (Unchanged) ---
def make_aggregate_target(image_paths):
    """Creates a median aggregate target image from a list of image paths."""
    images_rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths if cv2.imread(p) is not None]
    if not images_rgb: raise ValueError("No valid images found to build aggregate target.")
    stack = np.stack(images_rgb, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median

def fit_normalizer_on_train_set(train_df, method_name):
    """Fits a stain normalizer by selecting high-entropy images from the training set."""
    logging.info(f"Fitting '{method_name}' normalizer using the TRAIN set...")
    patient_files = train_df.groupby('patient_id')['image_path'].apply(list).to_dict()
    if not patient_files:
        raise ValueError("No patient images found in the training set to fit the normalizer.")
    logging.info(f"Selecting one high-entropy image per patient from {len(patient_files)} training patients...")
    template_paths = [max(file_list, key=calculate_image_entropy) for file_list in patient_files.values()]
    logging.info(f"Creating aggregate target from {len(template_paths)} images...")
    target_rgb = make_aggregate_target(template_paths)
    logging.info(f"Fitting normalizer...")
    normalizer = stainnorm.get_normalizer(method_name)
    normalizer.fit(target_rgb)
    logging.info(f"'{method_name}' normalizer fitted successfully.")
    return normalizer

def save_normalizer_stats(normalizer, method_name, output_file):
    """Saves the key statistics of a fitted normalizer to a JSON file for reproducibility."""
    stats = {"method": method_name}
    try:
        if method_name == "Reinhard":
            stats["target_means"] = normalizer.target_means.tolist()
            stats["target_stds"] = normalizer.target_stds.tolist()
        elif method_name in ["Macenko", "Vahadane"]:
            stats["stain_matrix_target"] = normalizer.stain_matrix_target.tolist()
        else: # Ruifrok may not have simple public params
             stats["info"] = "Fitted using tiatoolbox. No simple parameters to save."
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Normalization stats saved to {output_file}")
    except Exception as e:
        logging.error(f"Could not save normalizer stats: {e}")

# --- File Operations (Unchanged) ---
def process_and_copy_image(src_path, dest_path, normalizer):
    """Worker function: reads, optionally normalizes, and writes an image."""
    try:
        image_bgr = cv2.imread(src_path)
        if image_bgr is None: raise IOError(f"Could not read image: {src_path}")
        if normalizer:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            normalized_rgb = normalizer.transform(image_rgb)
            output_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
        else:
            output_bgr = image_bgr
        if not cv2.imwrite(dest_path, output_bgr): raise IOError(f"Failed to write image to {dest_path}")
        return True, None
    except Exception as e:
        return False, f"Error on {os.path.basename(src_path)}: {e}"

def process_and_write_split_files(split_df, output_dir, split_name, normalizer):
    """Processes and copies all files for a given split (train, val, or test)."""
    if split_df.empty:
        logging.info(f"Skipping file processing for empty split: {split_name}")
        return
    logging.info(f"Processing and writing {len(split_df)} files for {split_name}...")
    tasks = []
    for _, row in split_df.iterrows():
        label_dir = 'CANCER' if row['label'] == 1 else 'NOT_CANCER'
        img_dest = os.path.join(output_dir, split_name, label_dir, row['filename'])
        tasks.append((row['image_path'], img_dest, normalizer))
        mask_dest = os.path.join(output_dir, split_name, f"{label_dir}_MASK", row['filename'])
        tasks.append((row['mask_path'], mask_dest, None))
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
        futures = {executor.submit(process_and_copy_image, src, dst, norm) for src, dst, norm in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
            success, msg = future.result()
            if not success: errors.append(msg)
    if errors:
        for error in errors: logging.error(f"Processing error: {error}")
        raise RuntimeError(f"Failed to process {len(errors)} files for {split_name}.")

# --- Augmentation & Verification (Unchanged) ---
TRANSFORM_CODES = ["HF", "VF", "RR", "GB", "HED", "HSV"]
def _transform_factory(code: str):
    if code == "HF": return A.HorizontalFlip(p=1.0)
    if code == "VF": return A.VerticalFlip(p=1.0)
    if code == "RR": return A.RandomRotate90(p=1.0)
    if code == "GB": return A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    if code == "HED": return A.HEStain(method="random_preset", intensity_shift_range=(-0.2, 0.2), intensity_scale_range=(0.7, 1.3), p=1.0)
    if code == "HSV": return A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=60, val_shift_limit=50, p=1.0)
    raise ValueError(f"Unknown transform code: {code}")

def apply_single_augmentation(image_path, mask_path, output_image_path, output_mask_path, transform_code, seed):
    try:
        seed_worker(seed)
        transform = _transform_factory(transform_code)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise IOError(f"Failed to read image/mask: {image_path}")
        augmented = transform(image=image, mask=mask)
        if not (cv2.imwrite(output_image_path, augmented['image']) and cv2.imwrite(output_mask_path, augmented['mask'])):
            raise IOError("cv2.imwrite() failed to save.")
        return True, None
    except Exception as e:
        return False, f"Error on {os.path.basename(image_path)} [{transform_code}, seed={seed}]: {e}"

def augment_and_balance_train_set(output_dir, train_df, random_state, num_workers=None):
    logging.info("Augmenting and balancing TRAIN set...")
    label_counts = train_df['label'].value_counts()
    n_cancer, n_nocancer = label_counts.get(1, 0), label_counts.get(0, 0)
    if n_cancer == n_nocancer:
        logging.info("Classes are already balanced.")
        return
    minority_label = 1 if n_cancer < n_nocancer else 0
    needed_augs = abs(n_cancer - n_nocancer)
    logging.info(f"Target: Augmenting {'CANCER' if minority_label == 1 else 'NOT_CANCER'} to generate {needed_augs} new samples.")
    minority_df = train_df[train_df['label'] == minority_label]
    files_by_patient = minority_df.groupby('patient_id')['filename'].apply(list).to_dict()
    minority_patients = sorted(files_by_patient.keys())
    patient_cycler, transform_cycler = itertools.cycle(minority_patients), itertools.cycle(TRANSFORM_CODES)
    rng = np.random.default_rng(random_state)
    augmentation_tasks = []
    for _ in range(needed_augs):
        patient_id = next(patient_cycler)
        original_filename = rng.choice(files_by_patient[patient_id])
        transform_code = next(transform_cycler)
        seed = int(rng.integers(0, 2**31 - 1))
        base_name, _ = os.path.splitext(original_filename)
        output_filename = f"{base_name}_aug_{transform_code}_{seed}.png"
        label_dir = 'CANCER' if minority_label == 1 else 'NOT_CANCER'
        img_dir = os.path.join(output_dir, 'TRAIN', label_dir)
        mask_dir = os.path.join(output_dir, 'TRAIN', f"{label_dir}_MASK")
        task = (os.path.join(img_dir, original_filename), os.path.join(mask_dir, original_filename), os.path.join(img_dir, output_filename), os.path.join(mask_dir, output_filename), transform_code, seed)
        augmentation_tasks.append(task)
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(apply_single_augmentation, *task) for task in augmentation_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Applying Augmentations"):
            success, msg = future.result()
            if not success: errors.append(msg)
    if errors:
        for error in errors: logging.error(f"Augmentation error: {error}")
        raise RuntimeError(f"{len(errors)} augmentation errors occurred.")

def verify_split_integrity(output_dir, split_name):
    logging.info(f"Verifying integrity of files for {split_name} split...")
    split_dir = os.path.join(output_dir, split_name)
    if not os.path.isdir(split_dir):
        logging.warning(f"Verification skipped: Directory not found for split '{split_name}'.")
        return
    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(split_dir, label_name)
        mask_dir = os.path.join(split_dir, f"{label_name}_MASK")
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)): continue
        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        if image_files != mask_files:
            raise ValueError(f"Integrity check FAILED for {split_name}: Image and mask file lists do not match for {label_name}.")
    logging.info(f"Integrity verification PASSED for {split_name}.")

def write_manifest_and_log_stats(output_dir):
    logging.info("Generating manifest and stats...")
    rows = []
    # <<< CHANGE: We don't need fold_num anymore, it's always 1 for a given run
    for split in ["TRAIN", "VALIDATION", "TEST"]:
        for label_name in ["CANCER", "NOT_CANCER"]:
            img_dir = os.path.join(output_dir, split, label_name)
            if not os.path.isdir(img_dir): continue
            for f in sorted(os.listdir(img_dir)):
                if not f.lower().endswith('.png'): continue
                match = re.search(r'PATIENT_(\d+)', f)
                pid = int(match.group(1)) if match else None
                aug_tag = 'AUG' if '_aug_' in f else 'ORIGINAL'
                # The 'fold' column is removed as it's no longer relevant
                rows.append({"split": split, "label": 1 if label_name == "CANCER" else 0, "patient_id": pid, "filename": f, "source": aug_tag})
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    # Patient-level stats
    for split in ["TRAIN", "VALIDATION", "TEST"]:
        split_df = manifest_df[manifest_df['split'] == split]
        if split_df.empty: continue
        counts = split_df.drop_duplicates(['patient_id', 'label']).groupby('label')['patient_id'].nunique()
        logging.info(f"{split} patient counts -> NOT_CANCER={counts.get(0, 0)}, CANCER={counts.get(1, 0)}")

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. CONFIGURATION ---
    DATA_DIRECTORY = r'D:\Usuario\Desktop\Base_de_dados\MASTER'
    OUTPUT_BASE_DIR = r'D:\Usuario\Desktop\Base_de_dados\ABLATION\NORMALIZED_SPLITS'
    
    # --- Select Normalization Method ---
    # Options: "NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"
    NORMALIZATION_METHOD = "REINHARD"
    
    # <<< CHANGE: This is now the primary control for reproducibility ---
    # Change this integer to generate a different 80/10/10 split
    RANDOM_STATE = 42
    
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)

    # --- 2. SETUP ---
    VALID_METHODS = ["NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"]
    if NORMALIZATION_METHOD not in VALID_METHODS:
        raise ValueError(f"Invalid NORMALIZATION_METHOD: '{NORMALIZATION_METHOD}'. Must be one of {VALID_METHODS}")
    
    # <<< CHANGE: Output directory now includes the seed for better tracking ---
    output_run_dir = os.path.join(OUTPUT_BASE_DIR, f"{NORMALIZATION_METHOD}_seed_{RANDOM_STATE}")
    
    if os.path.exists(output_run_dir):
        if input(f"Output directory '{output_run_dir}' exists. Delete and proceed? (yes/no): ").lower() != 'yes':
            sys.exit("Exiting.")
        shutil.rmtree(output_run_dir)
    os.makedirs(output_run_dir, exist_ok=True)
    setup_logging(output_run_dir)
    
    try:
        # --- 3. DATA LOADING AND SPLITTING ---
        logging.info("--- Starting Data Preparation Script ---")
        logging.info(f"Normalization Method: {NORMALIZATION_METHOD}")
        logging.info(f"Random State (Seed): {RANDOM_STATE}")
        
        data_df = load_data(DATA_DIRECTORY)
        
        # <<< CHANGE: Simplified function call ---
        split_data = create_train_val_test_split(data_df, random_state=RANDOM_STATE)
        if not split_data: 
            raise RuntimeError("Data split generation failed.")

        # --- 4. PROCESSING ---
        # <<< CHANGE: Removed the outer 'for' loop, as we only process one split ---
        logging.info(f"--- Processing Data Split ---")
        
        # 4.1. Setup Directories
        split_dfs = {'TRAIN': split_data['train_df'], 'VALIDATION': split_data['val_df'], 'TEST': split_data['test_df']}
        for split_name in split_dfs.keys():
            for label in ['CANCER', 'NOT_CANCER']:
                os.makedirs(os.path.join(output_run_dir, split_name, label), exist_ok=True)
                os.makedirs(os.path.join(output_run_dir, split_name, f"{label}_MASK"), exist_ok=True)

        # 4.2. Fit Normalizer on TRAIN set ONLY
        normalizer = None
        if NORMALIZATION_METHOD != 'NOT_NORMALIZED':
            normalizer = fit_normalizer_on_train_set(split_data['train_df'], NORMALIZATION_METHOD)
            stats_file = os.path.join(output_run_dir, 'normalization_stats.json')
            save_normalizer_stats(normalizer, NORMALIZATION_METHOD, stats_file)

        # 4.3. Process and Write Files for all splits
        for split_name, df in split_dfs.items():
            process_and_write_split_files(df, output_run_dir, split_name, normalizer)
            verify_split_integrity(output_run_dir, split_name)

        # 4.4. Augment and Balance TRAIN set
        augment_and_balance_train_set(output_run_dir, split_data['train_df'], RANDOM_STATE, num_workers=NUM_WORKERS)
        
        # 4.5. Post-Augmentation Verification
        logging.info(f"--- Post-Augmentation Verification ---")
        verify_split_integrity(output_run_dir, "TRAIN") 
        n_pos = len(os.listdir(os.path.join(output_run_dir, "TRAIN", "CANCER")))
        n_neg = len(os.listdir(os.path.join(output_run_dir, "TRAIN", "NOT_CANCER")))
        assert n_pos == n_neg, f"TRAIN set imbalance detected: CANCER={n_pos}, NOT_CANCER={n_neg}"
        logging.info(f"TRAIN set balance confirmed: {n_pos} images per class.")

        # 4.6. Create Manifest
        write_manifest_and_log_stats(output_run_dir)
        logging.info(f"--- Data Split processed successfully. ---")

        logging.info("--- All processing tasks completed! ---")

    except Exception as e:
        logging.critical(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
        
    logging.info("--- Script Finished ---")