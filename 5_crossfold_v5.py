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
# 5.  NOTE: Data augmentation must be applied *on-the-fly* during training (not written to disk here).
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

from dotenv import load_dotenv

# --- NEW: OpenSlide Dependency Setup (FOR WINDOWS) ---
# This block MUST run before importing tiatoolbox or openslide.
load_dotenv() # Load variables from .env file
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')

try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        # Temporarily add the OpenSlide bin directory to the DLL search path
        with os.add_dll_directory(OPENSLIDE_PATH):
            # Now that the path is configured, we can safely import libraries that depend on it.
            from tiatoolbox.tools import stainnorm
    else:
        # On non-Windows systems or if path isn't set, just import directly.
        from tiatoolbox.tools import stainnorm
except (ImportError, FileNotFoundError) as e:
    print(f"FATAL ERROR: Could not initialize OpenSlide, which TIAtoolbox depends on.")
    print(f"1. Make sure you have downloaded the OpenSlide binaries for Windows.")
    print(f"2. Ensure the OPENSLIDE_PATH in your .env file points to the 'bin' folder.")
    print(f"   Current Path: {OPENSLIDE_PATH}")
    print(f"   Error Details: {e}")
    sys.exit(1)
# --- END OF NEW BLOCK ---


import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# --- Logger Setup ---
def setup_logging(output_dir):
    """Configures the root logger to output to both a file and the console."""
    log_file = 'data_preparation.log'
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

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = data_dir + "\\" + label_name
        image_dir=os.path.normpath(image_dir)
        mask_dir = data_dir + "\\" + label_name+"_MASK"
        mask_dir=os.path.normpath(mask_dir)
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

def create_train_val_test_split(df, create_test_set=True, random_state=42):
    """
    Creates a single, stratified, patient-aware split.
    - If create_test_set is True: Creates an ~80/10/10 TRAIN/VAL/TEST split.
    - If create_test_set is False: Creates an ~80/20 TRAIN/VAL split.
    Dynamically adapts for datasets with few patients.
    """
    patient_df = df.groupby('patient_id')['label'].max().reset_index()
    n_patients = len(patient_df)

    if create_test_set:
        # --- SCENARIO 1: Create TRAIN, VALIDATION, and TEST sets (80/10/10) ---
        logging.info(f"--- Creating a best-effort ~80/10/10 TRAIN/VAL/TEST split with seed {random_state} ---")
        
        # Minimum of 3 patients needed for 3 sets
        if n_patients < 3:
            raise ValueError(f"Cannot create a TRAIN/VAL/TEST split with fewer than 3 patients. Found only {n_patients}.")

        n_splits_master = min(10, n_patients)
        n_splits_inner = n_splits_master - 1
        logging.info(f"Found {n_patients} patients. Using a {n_splits_master}-fold master split.")
        
        sgkf_master = StratifiedGroupKFold(n_splits=n_splits_master, shuffle=True, random_state=random_state)
        
        try:
            train_val_idx, test_idx = next(sgkf_master.split(patient_df, y=patient_df['label'], groups=patient_df['patient_id']))
            train_val_patient_df = patient_df.iloc[train_val_idx]
            
            if n_splits_inner < 2: raise ValueError("Cannot create inner split with fewer than 2 folds.")
            
            sgkf_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
            train_idx_inner, val_idx_inner = next(sgkf_inner.split(train_val_patient_df, y=train_val_patient_df['label'], groups=train_val_patient_df['patient_id']))
            
            train_idx, val_idx = train_val_patient_df.index[train_idx_inner], train_val_patient_df.index[val_idx_inner]
        except (StopIteration, ValueError) as e:
            logging.error(f"Could not generate the 3-way data split: {e}.")
            return None

        train_patients = set(patient_df.iloc[train_idx]['patient_id'])
        val_patients = set(patient_df.iloc[val_idx]['patient_id'])
        test_patients = set(patient_df.iloc[test_idx]['patient_id'])
        
        test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

    else:
        # --- SCENARIO 2: Create only TRAIN and VALIDATION sets (80/20) ---
        logging.info(f"--- Creating a best-effort ~80/20 TRAIN/VAL split with seed {random_state} ---")
        
        # Minimum of 2 patients needed for 2 sets
        if n_patients < 2:
            raise ValueError(f"Cannot create a TRAIN/VAL split with fewer than 2 patients. Found only {n_patients}.")

        n_splits = min(5, n_patients) # A 5-fold split creates 20% chunks
        logging.info(f"Found {n_patients} patients. Using a {n_splits}-fold split.")
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        try:
            # Take the first fold for train/validation
            train_idx, val_idx = next(sgkf.split(patient_df, y=patient_df['label'], groups=patient_df['patient_id']))
        except (StopIteration, ValueError) as e:
            logging.error(f"Could not generate the 2-way data split: {e}.")
            return None

        train_patients = set(patient_df.iloc[train_idx]['patient_id'])
        val_patients = set(patient_df.iloc[val_idx]['patient_id'])
        
        # CRITICAL: Create an empty DataFrame for the test set to ensure downstream compatibility
        test_patients = set()
        test_df = pd.DataFrame(columns=df.columns)

    # --- Common Logic for Both Scenarios ---
    # Ensure no patient overlap between the created splits
    assert train_patients.isdisjoint(val_patients) and train_patients.isdisjoint(test_patients) and val_patients.isdisjoint(test_patients)
    
    logging.info(f"Split created: Train patients={len(train_patients)}, Val patients={len(val_patients)}, Test patients={len(test_patients)}")
    
    # Create the final DataFrames
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    
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
    """Fits a stain normalizer and returns the normalizer and template paths.""" # <-- Docstring updated
    logging.info(f"Fitting '{method_name}' normalizer using the TRAIN set...")
    patient_files = train_df.groupby('patient_id')['image_path'].apply(list).to_dict()
    if not patient_files:
        raise ValueError("No patient images found in the training set to fit the normalizer.")
    logging.info(f"Selecting one high-entropy image per patient from {len(patient_files)} training patients...")
    
    # This list is what we need to save
    template_paths = [max(file_list, key=calculate_image_entropy) for file_list in patient_files.values()]
    
    logging.info(f"Creating aggregate target from {len(template_paths)} images...")
    target_rgb = make_aggregate_target(template_paths)
    logging.info(f"Fitting normalizer...")
    normalizer = stainnorm.get_normalizer(method_name)
    normalizer.fit(target_rgb)
    logging.info(f"'{method_name}' normalizer fitted successfully.")
    
    return normalizer, template_paths # <-- RETURN BOTH

def save_normalizer_stats(normalizer, method_name, output_dir, template_paths):
    """
    Saves the essential, reproducible statistics of a fitted normalizer to a JSON file.
    Also saves the template images used to generate these statistics.
    
    FINAL VERSION: Based on direct inspection of the tiatoolbox source code.
    Uses the correct attribute names for each normalizer class and robustly
    handles JSON serialization of numpy types.
    """
    stats = {"method": method_name}
    logging.info(f"Extracting stats for '{method_name}'...")
    
    try:
        # The base class for Macenko, Vahadane, and Ruifrok
        if isinstance(normalizer, stainnorm.StainNormalizer):
            # These normalizers all share these core attributes from the parent class.
            if hasattr(normalizer, 'stain_matrix_target'):
                stats["stain_matrix_target"] = normalizer.stain_matrix_target.tolist()
            if hasattr(normalizer, 'maxC_target'): # Correct attribute name is maxC_target
                stats["maxC_target"] = normalizer.maxC_target.tolist()

            # Macenko has an additional useful property on its extractor
            if method_name == "MACENKO" and hasattr(normalizer.extractor, 'stains'):
                stats["stain_vectors_source_estimate"] = normalizer.extractor.stains.tolist()
        
        # Reinhard is a separate class
        elif isinstance(normalizer, stainnorm.ReinhardNormalizer):
            # These are stored as tuples, so converting to list is safe.
            stats["target_means"] = list(normalizer.target_means)
            stats["target_stds"] = list(normalizer.target_stds)

        else:
            stats["info"] = "Unknown or unsupported normalizer type."
            logging.warning(f"Did not recognize normalizer type for '{method_name}'.")

        logging.info(f"Successfully extracted stats for '{method_name}'.")

    except AttributeError as e:
        error_msg = f"Could not extract parameters for '{method_name}'. Attribute missing: {e}"
        stats["error"] = error_msg
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred extracting stats for '{method_name}': {e}"
        stats["error"] = error_msg
        logging.error(error_msg, exc_info=True)


    # --- Save the JSON metadata file ---
    output_file = os.path.join(output_dir, 'normalization_stats.json')
    try:
        with open(output_file, 'w') as f:
            # Custom encoder to handle any numpy types robustly
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(stats, f, indent=4, cls=NumpyEncoder)
        logging.info(f"Normalization stats saved to {output_file}")
    except Exception as e:
        logging.error(f"Could not save normalization_stats.json: {e}")

    # --- Save the template images for full reproducibility ---
    template_dir = os.path.join(output_dir, 'normalization_templates')
    os.makedirs(template_dir, exist_ok=True)
    logging.info(f"Saving {len(template_paths)} template images to '{template_dir}'...")
    try:
        for i, src_path in enumerate(template_paths):
            dest_path = os.path.join(template_dir, f"template_{i:03d}_{os.path.basename(src_path)}")
            shutil.copy(src_path, dest_path)
    except Exception as e:
        logging.error(f"Could not save template images: {e}")


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

    # --- NEW FEATURE FLAG ---
    # Set to True to create an 80/10/10 split (TRAIN/VAL/TEST).
    # Set to False to create an 80/20 split (TRAIN/VAL only).
    CREATE_TEST_SET = False

    # --- Select Normalization Method ---
    # Options: "NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"
    NORMALIZATION_METHOD = "NOT_NORMALIZED"

    DATA_DIRECTORY = r'D:\Usuario\Desktop\Base_de_dados\CAMELYON16\PATCHES'
    DATA_DIRECTORY=os.path.normpath(DATA_DIRECTORY)
    OUTPUT_BASE_DIR = DATA_DIRECTORY + '\\' + NORMALIZATION_METHOD
    OUTPUT_BASE_DIR=os.path.normpath(OUTPUT_BASE_DIR)
        
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
        
        # --- MODIFICATION: Pass the new flag to the split function ---
        split_data = create_train_val_test_split(
            df=data_df, 
            create_test_set=CREATE_TEST_SET, # Pass the flag here
            random_state=RANDOM_STATE
        )
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
            # --- MODIFIED: Capture both return values ---
            normalizer, template_paths = fit_normalizer_on_train_set(split_data['train_df'], NORMALIZATION_METHOD)

            # --- MODIFIED: Pass the new arguments ---
            save_normalizer_stats(
                normalizer=normalizer,
                method_name=NORMALIZATION_METHOD,
                output_dir=output_run_dir, # Save in the main run directory
                template_paths=template_paths
            )

        # 4.3. Process and Write Files for all splits
        for split_name, df in split_dfs.items():
            process_and_write_split_files(df, output_run_dir, split_name, normalizer)
            verify_split_integrity(output_run_dir, split_name)

        # 4.4. Offline augmentation/balancing REMOVED (scientific: augmentation must be on-the-fly)
        logging.info("Skipping offline augmentation/balancing. Apply augmentations on-the-fly during training.")

        # 4.5. Post-write sanity check (counts only; no enforced balancing here)
        verify_split_integrity(output_run_dir, "TRAIN")
        train_pos = len([f for f in os.listdir(os.path.join(output_run_dir, "TRAIN", "CANCER")) if f.lower().endswith(".png")])
        train_neg = len([f for f in os.listdir(os.path.join(output_run_dir, "TRAIN", "NOT_CANCER")) if f.lower().endswith(".png")])
        logging.info(f"TRAIN set written: CANCER={train_pos}, NOT_CANCER={train_neg}")

        # 4.6. Create Manifest
        write_manifest_and_log_stats(output_run_dir)
        logging.info(f"--- Data Split processed successfully. ---")

        logging.info("--- All processing tasks completed! ---")

    except Exception as e:
        logging.critical(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
        
    logging.info("--- Script Finished ---")