import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import shutil
import re
import concurrent.futures
from tqdm import tqdm
import cv2
import itertools
from collections import defaultdict
import albumentations as A
import logging
import sys
import random

# --- Logger Setup Function ---
def setup_logging():
    """Configures the root logger to output to both a file and the console."""
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler('data_preparation.log')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    logging.info("Logging configured. Output will be saved to data_preparation.log")

# --- Seeding function for worker processes ---
def seed_worker(seed):
    """Seeds the random number generators for a worker process."""
    random.seed(seed)
    np.random.seed(seed)

# --- 1. Data Loading and Preparation (REFACTORED for Robustness) ---
def load_data(data_dir):
    """Loads image and mask data deterministically and robustly, returning a DataFrame."""
    logging.info(f"Loading data from: {data_dir}")
    data = []
    for image_dir, mask_dir in zip(
        [os.path.join(data_dir, "CANCER"), os.path.join(data_dir, "NOT_CANCER")],
        [os.path.join(data_dir, "CANCER_MASK"), os.path.join(data_dir, "NOT_CANCER_MASK")]
    ):
        label_name = os.path.basename(image_dir)
        label = 1 if label_name == "CANCER" else 0
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            logging.warning(f"Missing directories for {label_name}. Skipping.")
            continue
        
        logging.info(f"Scanning {label_name} images...")
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        # CRITICAL FIX: Filter mask files for .png to avoid mismatches
        mask_files  = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        
        for image_name in tqdm(image_files, desc=f"Processing {label_name} files"):
            if image_name not in mask_files:
                logging.warning(f"Mask not found for image '{image_name}'. Skipping.")
                continue

            match = re.search(r'PATIENT_(\d+)_', image_name)
            if not match:
                logging.warning(f"Could not extract patient ID from {image_name}. Skipping.")
                continue
            
            # CRITICAL FIX: Add back file existence and readability checks
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)
            if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
                logging.warning(f"File path(s) invalid for '{image_name}'. Skipping.")
                continue
            if cv2.imread(image_path) is None:
                logging.warning(f"Could not read image file '{image_path}'. Skipping.")
                continue
            if cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) is None:
                logging.warning(f"Could not read mask file '{mask_path}'. Skipping.")
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

# --- 2. Cross-Validation Splitting (Patient-Level) ---
def _get_patient_labels(df: pd.DataFrame):
    return df.groupby('patient_id')['label'].max().reset_index()

def create_cross_val_splits(df, n_splits=5, random_state=42):
    # This function is already scientifically sound.
    logging.info(f"Creating {n_splits} patient-level stratified group splits...")
    patient_df = _get_patient_labels(df)
    sgkf_outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    final_splits = []
    for i, (train_val_idx, test_idx) in enumerate(sgkf_outer.split(patient_df, y=patient_df['label'], groups=patient_df['patient_id'])):
        train_val_patients = set(patient_df.iloc[train_val_idx]['patient_id'])
        test_patients = set(patient_df.iloc[test_idx]['patient_id'])
        train_val_df = df[df['patient_id'].isin(train_val_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]
        
        inner_patient_df = _get_patient_labels(train_val_df)
        inner_n_splits = min(5, max(2, inner_patient_df.groupby('label')['patient_id'].nunique().min()))
        sgkf_inner = StratifiedGroupKFold(n_splits=inner_n_splits, shuffle=True, random_state=random_state)
        
        try:
            inner_train_idx, inner_val_idx = next(sgkf_inner.split(inner_patient_df, y=inner_patient_df['label'], groups=inner_patient_df['patient_id']))
        except (StopIteration, ValueError) as e:
            logging.error(f"Could not generate inner split for Fold {i+1}: {e}. Skipping.")
            continue
            
        train_patients = set(inner_patient_df.iloc[inner_train_idx]['patient_id'])
        val_patients = set(inner_patient_df.iloc[inner_val_idx]['patient_id'])

        train_df = train_val_df[train_val_df['patient_id'].isin(train_patients)].reset_index(drop=True)
        val_df = train_val_df[train_val_df['patient_id'].isin(val_patients)].reset_index(drop=True)
        
        assert train_patients.isdisjoint(val_patients) and train_patients.isdisjoint(test_patients) and val_patients.isdisjoint(test_patients)
        final_splits.append({'train_df': train_df, 'val_df': val_df, 'test_df': test_df.reset_index(drop=True)})
        logging.info(f"Fold {i+1}: Train patients={len(train_patients)}, Val patients={len(val_patients)}, Test patients={len(test_patients)}")
    return final_splits

# --- 3. File Operations & Verification ---
def copy_files_for_split(split_df, fold_output_dir, split_name):
    # This function is already robust.
    logging.info(f"Copying {len(split_df)} original files for {split_name}...")
    copy_tasks = []
    for _, row in split_df.iterrows():
        label_dir = 'CANCER' if row['label'] == 1 else 'NOT_CANCER'
        img_dest = os.path.join(fold_output_dir, split_name, label_dir, row['filename'])
        mask_dest = os.path.join(fold_output_dir, split_name, f"{label_dir}_MASK", row['filename'])
        copy_tasks.extend([(row['image_path'], img_dest), (row['mask_path'], mask_dest)])
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
        futures = {executor.submit(shutil.copy2, src, dst) for src, dst in copy_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Copying {split_name}"):
            try: future.result()
            except Exception as exc: errors.append(str(exc))
    if errors:
        for error in errors: logging.error(f"Copy error: {error}")
        raise RuntimeError(f"Failed to copy {len(errors)} files for {split_name}. Aborting fold.")

def verify_split_integrity(fold_output_dir, split_name, original_df=None):
    # This function is already robust.
    logging.info(f"Verifying integrity of files for {split_name} split...")
    split_dir = os.path.join(fold_output_dir, split_name)
    error_messages = []
    aug_pattern_verify = re.compile(r'_aug_([A-Z]{2,3})_\d+\.png$')
    present_patients = set()
    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(split_dir, label_name)
        mask_dir = os.path.join(split_dir, f"{label_name}_MASK")
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            error_messages.append(f"Missing directory for {split_name}/{label_name}.")
            continue
        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        if image_files != mask_files:
            if missing := image_files - mask_files: error_messages.append(f"{label_name}: {len(missing)} images missing masks (e.g., {list(missing)[:3]})")
            if missing := mask_files - image_files: error_messages.append(f"{label_name}: {len(missing)} masks missing images (e.g., {list(missing)[:3]})")
        for img_file in image_files:
            if split_name in ['VALIDATION', 'TEST'] and aug_pattern_verify.search(img_file):
                error_messages.append(f"FATAL LEAKAGE: Augmented file '{img_file}' in '{split_name}' split!")
            match = re.search(r'PATIENT_(\d+)_', img_file)
            if match: present_patients.add(int(match.group(1)))
            elif not aug_pattern_verify.search(img_file): error_messages.append(f"Cannot parse patient ID from original file: {img_file}")
    if original_df is not None:
        original_patient_ids = set(original_df['patient_id'])
        if present_patients != original_patient_ids:
            if missing := original_patient_ids - present_patients: error_messages.append(f"Patient Mismatch: {len(missing)} patients missing (e.g., {list(missing)[:3]}).")
            if extra := present_patients - original_patient_ids: error_messages.append(f"Patient Leakage: {len(extra)} unexpected patients found (e.g., {list(extra)[:3]}).")
    if error_messages:
        for msg in error_messages: logging.error(f"  - {msg}")
        raise ValueError(f"Integrity check FAILED for {split_name}.")
    logging.info(f"Integrity verification PASSED for {split_name}.")
    return True

# --- 4. Augmentation (Reproducible & Process-Safe) ---
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
    """Worker function with shape validation."""
    try:
        seed_worker(seed)
        transform = _transform_factory(transform_code)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise IOError(f"Failed to read image/mask: {image_path}")
        if image.shape[:2] != mask.shape[:2]: raise ValueError(f"Shape mismatch for {os.path.basename(image_path)}: img{image.shape[:2]} vs mask{mask.shape[:2]}")
        augmented = transform(image=image, mask=mask)
        if not (cv2.imwrite(output_image_path, augmented['image']) and cv2.imwrite(output_mask_path, augmented['mask'])):
            raise IOError("cv2.imwrite() failed to save.")
        return True, None
    except Exception as e:
        return False, f"Error on {os.path.basename(image_path)} [{transform_code}, seed={seed}]: {e}"

def augment_and_balance_train_set(fold_output_dir, train_df, random_state, num_workers=None):
    """Balances the training set using fair, patient-centric oversampling."""
    logging.info("Augmenting and balancing TRAIN set...")
    label_counts = train_df['label'].value_counts()
    n_cancer, n_nocancer = label_counts.get(1, 0), label_counts.get(0, 0)
    if n_cancer == n_nocancer:
        logging.info("Classes are already balanced.")
        return []
    minority_label = 1 if n_cancer < n_nocancer else 0
    needed_augs = abs(n_cancer - n_nocancer)
    logging.info(f"Target: Augmenting {'CANCER' if minority_label == 1 else 'NOT_CANCER'} to generate {needed_augs} new samples.")
    minority_df = train_df[train_df['label'] == minority_label]
    files_by_patient = minority_df.groupby('patient_id')['filename'].apply(list).to_dict()
    minority_patients = sorted(files_by_patient.keys())
    patient_cycler, transform_cycler = itertools.cycle(minority_patients), itertools.cycle(TRANSFORM_CODES)
    rng = np.random.default_rng(random_state)
    augmentation_tasks, augmentation_details, file_aug_counters = [], [], defaultdict(int)
    for _ in range(needed_augs):
        patient_id = next(patient_cycler)
        original_filename = rng.choice(files_by_patient[patient_id])
        transform_code = next(transform_cycler)
        seed = int(rng.integers(0, 2**31 - 1))
        base_name = os.path.splitext(original_filename)[0]
        file_aug_counters[base_name] += 1
        output_filename = f"{base_name}_aug_{transform_code}_{file_aug_counters[base_name]}.png"
        label_dir = 'CANCER' if minority_label == 1 else 'NOT_CANCER'
        img_dir = os.path.join(fold_output_dir, 'TRAIN', label_dir)
        mask_dir = os.path.join(fold_output_dir, 'TRAIN', f"{label_dir}_MASK")
        task = (os.path.join(img_dir, original_filename), os.path.join(mask_dir, original_filename), os.path.join(img_dir, output_filename), os.path.join(mask_dir, output_filename), transform_code, seed)
        augmentation_tasks.append(task)
        augmentation_details.append({"augmented_filename": output_filename, "base_filename": original_filename, "patient_id": patient_id, "transform_code": transform_code, "seed": seed})
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(apply_single_augmentation, *task) for task in augmentation_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Applying Augmentations"):
            success, msg = future.result()
            if not success: errors.append(msg)
    if errors:
        for error in errors: logging.error(f"Augmentation error: {error}")
        raise RuntimeError(f"{len(errors)} augmentation errors occurred. Aborting fold.")
    logging.info("Train set balancing and augmentation complete.")
    return augmentation_details

# --- 5. Manifest Generation ---
def write_manifest_and_log_stats(fold_output_dir, fold_num):
    """Creates a manifest CSV for the fold and logs patient-level statistics."""
    logging.info(f"Generating manifest and stats for Fold {fold_num}...")
    rows = []
    for split in ["TRAIN", "VALIDATION", "TEST"]:
        for label_name in ["CANCER", "NOT_CANCER"]:
            img_dir = os.path.join(fold_output_dir, split, label_name)
            if not os.path.isdir(img_dir): continue
            for f in sorted(os.listdir(img_dir)):
                if not f.lower().endswith('.png'): continue
                match = re.search(r'PATIENT_(\d+)', f)
                pid = int(match.group(1)) if match else None
                aug_tag = None
                if aug_match := re.search(r'_aug_([A-Z]{2,3})_(\d+)\.png$', f): aug_tag = f"{aug_match.group(1)}#{aug_match.group(2)}"
                rows.append({"fold": fold_num, "split": split, "label": 1 if label_name == "CANCER" else 0, "patient_id": pid, "filename": f, "augmentation": aug_tag})
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(os.path.join(fold_output_dir, "manifest.csv"), index=False)
    logging.info(f"--- Patient-Level Stats for Fold {fold_num} ---")
    for split in ["TRAIN", "VALIDATION", "TEST"]:
        split_df = manifest_df[manifest_df['split'] == split]
        if split_df.empty: continue
        counts = split_df.drop_duplicates(['patient_id', 'label']).groupby('label')['patient_id'].nunique()
        logging.info(f"{split} patient counts -> NOT_CANCER={counts.get(0, 0)}, CANCER={counts.get(1, 0)}")
    return manifest_df

# --- 6. Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    data_directory = r'D:\Usuario\Desktop\Base_de_dados\MASTER_BASE'
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\NOT_NORMALIZED_CLEANED\cross_val_splits_final'
    N_SPLITS = 5
    RANDOM_STATE = 45
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)

    if os.path.exists(output_base_dir):
        if input(f"Output directory '{output_base_dir}' exists. Delete and proceed? (yes/no): ").lower() != 'yes':
            sys.exit("Exiting.")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    setup_logging()

    all_manifests, all_aug_details = [], []
    try:
        logging.info("--- Starting Data Preparation Script ---")
        data_df = load_data(data_directory)
        cross_val_splits = create_cross_val_splits(data_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
        if not cross_val_splits: raise RuntimeError("No cross-validation splits were generated.")

        for fold_idx, split_data in enumerate(cross_val_splits):
            fold_num = fold_idx + 1
            fold_output_dir = os.path.join(output_base_dir, f"fold_{fold_num}")
            logging.info(f"--- Processing Fold {fold_num}/{len(cross_val_splits)} ---")
            
            try:
                # 1. Setup, Copy, and Pre-Augment Verify
                split_dfs = {'TRAIN': split_data['train_df'], 'VALIDATION': split_data['val_df'], 'TEST': split_data['test_df']}
                for split_name, df in split_dfs.items():
                    for label in ['CANCER', 'NOT_CANCER']:
                        os.makedirs(os.path.join(fold_output_dir, split_name, label), exist_ok=True)
                        os.makedirs(os.path.join(fold_output_dir, split_name, f"{label}_MASK"), exist_ok=True)
                    copy_files_for_split(df, fold_output_dir, split_name)
                    verify_split_integrity(fold_output_dir, split_name, df)

                # 2. Augment and Balance
                aug_details = augment_and_balance_train_set(fold_output_dir, split_data['train_df'], RANDOM_STATE + fold_num, num_workers=NUM_WORKERS)
                # NICE-TO-HAVE: Log augmentation summary
                if aug_details:
                    df_aug = pd.DataFrame(aug_details)
                    logging.info(f"Augmentation transform mix: {df_aug['transform_code'].value_counts().to_dict()}")
                    logging.info(f"Stats for # of augmentations per patient: {df_aug['patient_id'].value_counts().describe().to_dict()}")

                # 3. Post-Augment Verification
                logging.info(f"--- Post-Augmentation Verification for Fold {fold_num} ---")
                verify_split_integrity(fold_output_dir, "TRAIN") # Pass None for df as we're just checking file correspondence
                
                # CRITICAL FIX: Helper function for accurate PNG count
                def _count_pngs(p): return sum(1 for f in os.listdir(p) if f.lower().endswith('.png')) if os.path.isdir(p) else 0
                n_pos = _count_pngs(os.path.join(fold_output_dir, "TRAIN", "CANCER"))
                n_neg = _count_pngs(os.path.join(fold_output_dir, "TRAIN", "NOT_CANCER"))
                assert n_pos == n_neg, f"TRAIN set imbalance detected: CANCER={n_pos}, NOT_CANCER={n_neg}"
                logging.info(f"TRAIN set balance confirmed: {n_pos} images per class.")

                # 4. Create manifests
                manifest = write_manifest_and_log_stats(fold_output_dir, fold_num)
                all_manifests.append(manifest)
                if aug_details:
                    aug_df = pd.DataFrame(aug_details)
                    aug_df['fold'] = fold_num
                    aug_df.to_csv(os.path.join(fold_output_dir, "augmentations.csv"), index=False)
                    all_aug_details.append(aug_df)
                
                logging.info(f"--- Fold {fold_num} completed successfully. ---")
            
            except Exception as e:
                logging.critical(f"--- Fold {fold_num} FAILED: {e} ---")

        # --- Final Summary ---
        if len(all_manifests) == N_SPLITS:
            pd.concat(all_manifests).to_csv(os.path.join(output_base_dir, "full_manifest.csv"), index=False)
            if all_aug_details: pd.concat(all_aug_details).to_csv(os.path.join(output_base_dir, "full_augmentations.csv"), index=False)
            logging.info("All folds processed successfully! Full manifests have been created.")
        else:
            logging.error(f"Processing failed. Only {len(all_manifests)}/{N_SPLITS} folds were successful.")

    except Exception as e:
        logging.critical(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
        
    logging.info("--- Script Finished ---")