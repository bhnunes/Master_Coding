import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import shutil
import re
import concurrent.futures
from tqdm import tqdm
import math
import cv2
import itertools # NEW: Import itertools for efficient cycling
from collections import defaultdict # NEW: Import defaultdict for unique filename generation

# --- Install and Import Albumentations ---
try:
    import albumentations as A
except ImportError:
    print("Albumentations library not found. Please install it: pip install -U albumentations")
    exit()

# --- 1. Data Loading and Preparation (Robust) ---
def load_data(data_dir):
    """Loads image and mask data, parses filenames, and returns a DataFrame."""
    print(f"Loading data from: {data_dir}")
    data = []
    expected_image_dirs = [os.path.join(data_dir, "CANCER"), os.path.join(data_dir, "NOT_CANCER")]
    expected_mask_dirs = [os.path.join(data_dir, "CANCER_MASK"), os.path.join(data_dir, "NOT_CANCER_MASK")]

    for image_dir, mask_dir in zip(expected_image_dirs, expected_mask_dirs):
        label_name = os.path.basename(image_dir) # CANCER or NOT_CANCER
        label = 1 if label_name == "CANCER" else 0

        if not os.path.isdir(image_dir):
            print(f"Warning: Image directory not found: {image_dir}. Skipping label {label_name}.")
            continue
        if not os.path.isdir(mask_dir):
            print(f"Warning: Mask directory not found: {mask_dir}. Skipping label {label_name}.")
            continue

        print(f"  Scanning {label_name} images in {os.path.basename(image_dir)}...")
        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        print(f"    Found {len(image_files)} images, {len(mask_files)} masks.")

        for image_name in tqdm(image_files, desc=f"Processing {label_name} files"):
            mask_name = image_name # Assume mask has the same name

            if mask_name not in mask_files:
                print(f"Warning: Mask '{mask_name}' not found for image '{image_name}' in {mask_dir}. Skipping image.")
                continue # Skip if mask doesn't exist

            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, mask_name)

            # Attempt to extract patient ID
            match = re.search(r'PATIENT_(\d+)_', image_name)
            if match:
                try:
                    patient_id = int(match.group(1))
                except ValueError:
                     print(f"Warning: Could not convert patient ID to int in {image_name}. Skipping.")
                     continue
            else:
                print(f"Warning: Could not extract patient ID from {image_name}. Skipping.")
                continue

            # Basic check if files are accessible (more robust check later)
            if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                 print(f"Warning: File path invalid for {image_name} or {mask_name}. Skipping.")
                 continue

            data.append({
                'patient_id': patient_id,
                'image_path': image_path, # Store full path for easier access
                'mask_path': mask_path,   # Store full path
                'label': label,
                'filename': image_name # Store filename for checks
            })

    if not data:
        raise ValueError(f"No valid image/mask pairs with extractable patient IDs found in {data_dir}. Check directory structure and filenames.")

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} image/mask pairs for {df['patient_id'].nunique()} patients.")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


# --- 2. Cross-Validation Splitting ---
def create_cross_val_splits(df, n_splits=5, random_state=42):
    """Creates Stratified Group K-Fold cross-validation splits."""
    print(f"\nCreating {n_splits} stratified group K-Fold splits...")
    df['label'] = df['label'].astype(int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    min_groups = df.groupby('label')['patient_id'].nunique().min()
    if min_groups < n_splits:
         raise ValueError(f"Cannot perform {n_splits}-fold stratified group split. The least populated class has only {min_groups} unique patients. Minimum required: {n_splits}.")

    fold_indices = list(sgkf.split(df, df['label'], df['patient_id']))
    inner_n_splits = max(2, min(5, df['patient_id'].nunique())) # Adjust inner dynamically

    group_kfold_inner = GroupKFold(n_splits=inner_n_splits)

    for i, (train_val_idx, test_idx) in enumerate(fold_indices):
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        groups_inner = train_val_df['patient_id']

        # Adjust inner split size if needed
        current_inner_n_splits = inner_n_splits
        if len(train_val_df) < current_inner_n_splits or train_val_df['patient_id'].nunique() < current_inner_n_splits:
             print(f"Warning: Fold {i+1} - Adjusting inner KFold splits due to small train+val size.")
             current_inner_n_splits = max(2, min(inner_n_splits, train_val_df['patient_id'].nunique(), len(train_val_df)))
             group_kfold_inner_adj = GroupKFold(n_splits=current_inner_n_splits)
             inner_split_generator = group_kfold_inner_adj.split(train_val_df, groups=groups_inner)
        else:
             inner_split_generator = group_kfold_inner.split(train_val_df, groups=groups_inner) # Use original if possible

        try:
             train_idx_local, val_idx_local = next(inner_split_generator)
             train_df = train_val_df.iloc[train_idx_local]
             val_df = train_val_df.iloc[val_idx_local]
        except StopIteration:
             print(f"Error: Could not generate inner split for Fold {i+1}. Skipping fold.")
             continue

        splits.append({
            'train_df': train_df.reset_index(drop=True),
            'val_df': val_df.reset_index(drop=True),
            'test_df': test_df.reset_index(drop=True)
        })
        print(f"  Fold {i+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("\nVerifying patient separation across final splits...")
    for fold, split_data in enumerate(splits):
        train_patients = set(split_data['train_df']['patient_id'])
        val_patients = set(split_data['val_df']['patient_id'])
        test_patients = set(split_data['test_df']['patient_id'])
        assert len(train_patients.intersection(val_patients)) == 0, f"FATAL: Patient leakage detected in Fold {fold+1} (train/val overlap)"
        assert len(train_patients.intersection(test_patients)) == 0, f"FATAL: Patient leakage detected in Fold {fold+1} (train/test overlap)"
        assert len(val_patients.intersection(test_patients)) == 0, f"FATAL: Patient leakage detected in Fold {fold+1} (val/test overlap)"
    print("Patient separation verified.")
    return splits


# --- 3. File Operations & Verification ---
def copy_files_for_split(split_df, fold_output_dir, split_name):
    """Copies original images and masks for a given split (train/val/test)."""
    print(f"  Copying original files for {split_name}...")
    split_dir = os.path.join(fold_output_dir, split_name)

    copy_tasks = []
    for _, row in split_df.iterrows():
        label_name = 'CANCER' if row['label'] == 1 else 'NOT_CANCER'
        image_dest_dir = os.path.join(split_dir, label_name)
        mask_dest_dir = os.path.join(split_dir, f"{label_name}_MASK")

        image_src_path = row['image_path']
        mask_src_path = row['mask_path']
        base_filename = row['filename']

        image_dest_path = os.path.join(image_dest_dir, base_filename)
        mask_dest_path = os.path.join(mask_dest_dir, base_filename)

        copy_tasks.append((image_src_path, image_dest_path))
        copy_tasks.append((mask_src_path, mask_dest_path))

    errors = []
    num_copy_workers = min(32, (os.cpu_count() or 1) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_copy_workers) as executor:
        future_to_task = {executor.submit(shutil.copy2, src, dst): (src, dst) for src, dst in copy_tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(copy_tasks), desc=f"  Copying {split_name}"):
            src, dst = future_to_task[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"Failed to copy {src} to {dst}: {exc}")

    if errors:
        print(f"\n  --- Errors during copying for {split_name} ---")
        for error in errors[:10]: print(f"    {error}")
        if len(errors) > 10: print(f"    ... ({len(errors)-10} more errors)")
        print(f"  --- End of Copying Errors ---")
        raise RuntimeError(f"Errors occurred during file copying for {split_name}. Check logs.")


def verify_split_integrity(fold_output_dir, split_name, original_df):
    """Performs checks on a created split directory (patient leakage, counts, file existence)."""
    print(f"  Verifying integrity of {split_name} split...")
    split_dir = os.path.join(fold_output_dir, split_name)
    is_train = (split_name == 'TRAIN')

    all_files_ok = True
    present_patients = set()
    error_messages = []
    aug_pattern_verify = re.compile(r'_aug_([A-Z]{2})_\d+\.png$')

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(split_dir, label_name)
        mask_dir = os.path.join(split_dir, f"{label_name}_MASK")

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
             error_messages.append(f"Missing directory: {image_dir} or {mask_dir}")
             all_files_ok = False
             continue

        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}

        if len(image_files) != len(mask_files):
            error_messages.append(f"{label_name}: Image count ({len(image_files)}) != Mask count ({len(mask_files)})")
            all_files_ok = False

        missing_masks = image_files - mask_files
        missing_images = mask_files - image_files
        if missing_masks:
            error_messages.append(f"{label_name}: Images missing masks (e.g., {list(missing_masks)[:3]})")
            all_files_ok = False
        if missing_images:
            error_messages.append(f"{label_name}: Masks missing images (e.g., {list(missing_images)[:3]})")
            all_files_ok = False

        for img_file in image_files:
             match_patient = re.search(r'PATIENT_(\d+)_', img_file)
             if match_patient:
                 present_patients.add(int(match_patient.group(1)))
             else:
                 is_augmented = bool(aug_pattern_verify.search(img_file))
                 if not is_augmented:
                      error_messages.append(f"{label_name}: Cannot parse patient ID from non-augmented file: {img_file}")
                      all_files_ok = False

             if not is_train and aug_pattern_verify.search(img_file):
                  error_messages.append(f"FATAL LEAKAGE: Augmented file '{img_file}' found in {split_name}/{label_name}!")
                  all_files_ok = False

    original_patient_ids = set(original_df['patient_id'])
    if present_patients != original_patient_ids:
         if len(present_patients) != len(original_patient_ids):
              error_messages.append(f"Patient ID count mismatch: Expected {len(original_patient_ids)} unique patients (from DF), found {len(present_patients)} (in files). Possible copy issue or parsing error.")
              all_files_ok = False

    if not all_files_ok:
        print(f"  --- Verification FAILED for {split_name} ---")
        for msg in error_messages: print(f"    - {msg}")
        print(f"  --- End Verification Errors ---")
        raise ValueError(f"Integrity check failed for {split_name} in fold {os.path.basename(fold_output_dir)}")
    else:
        print(f"  Verification PASSED for {split_name}.")
        return True


# --- 4. Augmentation (MODIFIED FOR GLOBAL CYCLIC DISTRIBUTION) ---

# Define the INDIVIDUAL transformations, each with p=1.0 (always apply if chosen)
INDIVIDUAL_TRANSFORMS = [
    (A.HorizontalFlip(p=1.0), "HP"),
    (A.VerticalFlip(p=1.0), "VF"),
    (A.RandomRotate90(p=1.0), "RF"),
    (A.GaussianBlur(blur_limit=(3, 7), p=1.0), "GB"),
    (A.ColorJitter(brightness=0.25, contrast=0.3, saturation=0.3, hue=0.04, p=1.0), "CJ") # hue=0.04 implies [-0.04, 0.04] for albumentations
]

print(f"Defined {len(INDIVIDUAL_TRANSFORMS)} individual augmentations for deterministic cyclic selection:")
for _, code in INDIVIDUAL_TRANSFORMS:
    print(f"  - {code}")

# --- NEW, SIMPLIFIED WORKER FUNCTION ---
def apply_single_augmentation(image_path, mask_path, output_image_path, output_mask_path, transform):
    """
    Worker function that applies a single, specific augmentation to an image/mask pair.
    This function is designed to be called by a ProcessPoolExecutor.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image file: {image_path}")
        if mask is None:
            raise IOError(f"Could not read mask file: {mask_path}")

        # Apply the specific transform received
        augmented = transform(image=image, mask=mask)
        augmented_img = augmented['image']
        augmented_mask = augmented['mask']

        # Save the results
        cv2.imwrite(output_image_path, augmented_img)
        cv2.imwrite(output_mask_path, augmented_mask)
        return True, None
    except Exception as e:
        error_msg = f"Failed augmenting {os.path.basename(image_path)} -> {os.path.basename(output_image_path)}: {type(e).__name__}: {e}"
        return False, error_msg

# --- HEAVILY MODIFIED MASTER FUNCTION ---
def augment_and_balance_train_set(fold_output_dir, num_workers=None):
    """
    Balances the training set by augmenting the minority class using a globally distributed
    cyclic selection of transforms, ensuring perfect balance and reproducibility.
    """
    print("  Augmenting and balancing TRAIN set (using GLOBAL CYCLIC distribution)...")
    train_dir = os.path.join(fold_output_dir, "TRAIN")
    cancer_img_dir = os.path.join(train_dir, "CANCER")
    cancer_mask_dir = os.path.join(train_dir, "CANCER_MASK")
    nocancer_img_dir = os.path.join(train_dir, "NOT_CANCER")
    nocancer_mask_dir = os.path.join(train_dir, "NOT_CANCER_MASK")

    # This regex is used for identifying augmented files if they already exist
    aug_pattern_balance = re.compile(r'_aug_([A-Z]{2})_\d+\.png$')

    # Filter out any pre-existing augmented files to ensure we only augment original samples
    original_cancer_files = [f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png') and not aug_pattern_balance.search(f)]
    original_nocancer_files = [f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png') and not aug_pattern_balance.search(f)]

    n_cancer, n_nocancer = len(original_cancer_files), len(original_nocancer_files)
    print(f"    Original counts: Cancer={n_cancer}, Not_Cancer={n_nocancer}")

    if n_cancer == n_nocancer:
        print("    Classes are already balanced. No augmentation needed.")
        return True

    # Determine minority/majority class
    if n_cancer < n_nocancer:
        minority_img_dir, minority_mask_dir = cancer_img_dir, cancer_mask_dir
        minority_files, n_minority, n_majority = original_cancer_files, n_cancer, n_nocancer
        print(f"    Target: Augmenting CANCER class.")
    else:
        minority_img_dir, minority_mask_dir = nocancer_img_dir, nocancer_mask_dir
        minority_files, n_minority, n_majority = original_nocancer_files, n_nocancer, n_cancer
        print(f"    Target: Augmenting NOT_CANCER class.")

    if n_minority == 0:
        print(f"    Warning: Minority class has 0 samples. Cannot balance.")
        return True

    needed_total_augmentations = n_majority - n_minority
    print(f"    Need to generate {needed_total_augmentations} new samples for the minority class.")

    # --- Create a globally balanced list of all augmentation tasks ---
    augmentation_tasks = []
    
    # Create infinite iterators that will cycle through the available files and transforms
    file_cycler = itertools.cycle(minority_files)
    transform_cycler = itertools.cycle(INDIVIDUAL_TRANSFORMS)
    
    # Keep track of how many times we've used each base filename to create a unique suffix
    file_aug_counters = defaultdict(int)

    for _ in range(needed_total_augmentations):
        original_filename = next(file_cycler)
        transform_obj, transform_code = next(transform_cycler) # Get both object and code

        base_name_without_ext = os.path.splitext(original_filename)[0]
        file_aug_counters[base_name_without_ext] += 1
        
        # Construct unique, informative filenames for the augmented output
        output_filename = f"{base_name_without_ext}_aug_{transform_code}_{file_aug_counters[base_name_without_ext]}.png"
        
        image_src_path = os.path.join(minority_img_dir, original_filename)
        mask_src_path = os.path.join(minority_mask_dir, original_filename) # Mask has same name as image
        output_image_path = os.path.join(minority_img_dir, output_filename)
        output_mask_path = os.path.join(minority_mask_dir, output_filename)
        
        # Add a single task for the worker function
        augmentation_tasks.append((image_src_path, mask_src_path, output_image_path, output_mask_path, transform_obj))

    # --- Execute Augmentation in Parallel ---
    errors = []
    print(f"    Starting parallel execution of {len(augmentation_tasks)} single augmentation tasks...")
    actual_num_workers = os.cpu_count() if num_workers is None else num_workers
    print(f"    Using {actual_num_workers} worker processes.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_num_workers) as executor:
        # Submit each individual augmentation task to the executor
        future_to_task = {executor.submit(apply_single_augmentation, *task): task for task in augmentation_tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(augmentation_tasks), desc="    Applying Augmentations"):
            try:
                success, msg = future.result()
                if not success:
                    errors.append(msg)
            except Exception as exc:
                task_info = future_to_task[future]
                errors.append(f"Exception for task on {os.path.basename(task_info[0])}: {exc}")

    # --- Report Augmentation Summary ---
    # Recalculate count including newly generated augmented files
    final_minority_images = [f for f in os.listdir(minority_img_dir) if f.lower().endswith('.png')]
    final_minority_count = len(final_minority_images)
    print(f"    Augmentation finished. Final minority class count: {final_minority_count} (Majority: {n_majority})")

    if errors:
        print(f"  --- Errors during Augmentation ---")
        for i, error in enumerate(errors):
            if i < 10: print(f"    - {error}")
            elif i == 10: print(f"    ... (omitting {len(errors)-10} more errors)")
            break
        print(f"  --- End Augmentation Errors ---")
    
    # Downsampling is inherently handled as we generate the exact required number of samples.
    # No need for explicit downsampling logic here.
    
    print("  Train set balancing and augmentation complete.")
    return True


# --- 5. Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # This `data_directory` should point to the output of your normalization pipeline
    # e.g., r'D:\Usuario\Desktop\Base_de_dados\ABLATION\TIATOOLBOX_NORMALIZED\Ruifrok'
    data_directory = r'D:\Usuario\Downloads\unzipped_master' # Example, adjust as needed
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\NOT_NORMALIZED\cross_val_splits_balanced_geometric_aug' # Example, adjust as needed
    N_SPLITS = 5
    RANDOM_STATE = 45 # This RANDOM_STATE affects cross-validation splits, not augmentation selection anymore.
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1) # Adjust workers based on CPU capability; augmentation is CPU-bound

    # --- Prepare Output Directory ---
    if os.path.exists(output_base_dir):
        print(f"Output directory '{output_base_dir}' already exists.")
        user_input = input("  -> Delete existing directory and proceed? (yes/no): ").strip().lower()
        if user_input == 'yes':
            try:
                shutil.rmtree(output_base_dir)
                print("  Existing directory deleted.")
            except Exception as e:
                print(f"Error deleting directory: {e}. Exiting.")
                exit(1)
        else:
            print("Exiting.")
            exit(0)
    try:
         os.makedirs(output_base_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating base output directory {output_base_dir}: {e}")
         exit(1)

    # --- Load Data ---
    try:
        data_df = load_data(data_directory)
    except ValueError as e:
        print(f"Error loading data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Create Splits ---
    try:
        cross_val_splits = create_cross_val_splits(data_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    except ValueError as e:
        print(f"Error creating cross-validation splits: {e}")
        if "Cannot perform" in str(e) or "less than n_splits" in str(e) or "Minimum required" in str(e):
             print(f"  Try reducing N_SPLITS (currently {N_SPLITS}) or ensure sufficient patients per class.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during cross-validation split creation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    if not cross_val_splits:
        print("No cross-validation splits were generated. Exiting.")
        exit(1)

    # --- Process Each Fold ---
    print(f"\n--- Starting Processing for {len(cross_val_splits)} Folds ---")
    all_folds_successful = True
    for fold_idx, split_data in enumerate(cross_val_splits):
        fold_num = fold_idx + 1
        print(f"\n--- Processing Fold {fold_num}/{len(cross_val_splits)} ---")
        fold_output_dir = os.path.join(output_base_dir, f"fold_{fold_num}")

        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']

        fold_successful = True

        # --- Create Subdirectories and Copy Files ---
        print(f"Fold {fold_num}: Setting up directories and copying original files...")
        for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
            split_dir = os.path.join(fold_output_dir, split_name)
            try:
                os.makedirs(os.path.join(split_dir, 'CANCER'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'CANCER_MASK'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'NOT_CANCER'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'NOT_CANCER_MASK'), exist_ok=True)
            except OSError as e:
                print(f"  ERROR creating directories for {split_name} in Fold {fold_num}: {e}")
                fold_successful = False
                break

            if not fold_successful: break

            try:
                copy_files_for_split(split_df_current, fold_output_dir, split_name)
            except Exception as e:
                 print(f"  ERROR during file copying for {split_name} in Fold {fold_num}: {e}")
                 fold_successful = False
                 break

        if not fold_successful:
             all_folds_successful = False
             print(f"--- Skipping further processing for Fold {fold_num} due to setup/copying errors. ---")
             continue

        # --- Augment and Balance TRAINING Set ---
        print(f"\nFold {fold_num}: Augmenting and balancing training set...")
        try:
            balance_success = augment_and_balance_train_set(fold_output_dir, num_workers=NUM_WORKERS)
            if not balance_success:
                 print(f"  Warning: Augmentation/Balancing function reported potential issues for Fold {fold_num}, but proceeding.")
        except Exception as e:
            print(f"  FATAL ERROR during augmentation/balancing for Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            fold_successful = False

        # --- Final Verification for the Fold ---
        if fold_successful:
            print(f"\n--- Final Verification for Fold {fold_num} ---")
            verification_passed = True
            for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
                try:
                    if not verify_split_integrity(fold_output_dir, split_name, split_df_current):
                        verification_passed = False
                        fold_successful = False
                except Exception as e:
                     print(f"  ERROR during verification for {split_name} in Fold {fold_num}: {e}")
                     import traceback
                     traceback.print_exc()
                     verification_passed = False
                     fold_successful = False

            if fold_successful and verification_passed:
                 print(f"--- Fold {fold_num} completed successfully. ---")
            else:
                 all_folds_successful = False
                 print(f"--- Fold {fold_num} completed with VERIFICATION ERRORS or previous errors. ---")
        else:
             all_folds_successful = False
             print(f"--- Fold {fold_num} failed before final verification stage. ---")

    # --- Final Summary ---
    print("\n--- Overall Process Summary ---")
    if all_folds_successful:
        print("All folds processed and verified successfully!")
        print("Each augmented image in the TRAIN sets was generated using a globally deterministic cyclic distribution of transformations.")
        print(f"Output location: {output_base_dir}")
    else:
        print("Processing completed, but ERRORS occurred in one or more folds.")
        print("Please review the output messages and logs above to identify issues.")
        print(f"Output directory (may contain partial or erroneous data): {output_base_dir}")

    print("--- Script Finished ---")