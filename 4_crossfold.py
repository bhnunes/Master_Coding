import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
import shutil
import re
import concurrent.futures
from functools import partial
from tqdm import tqdm
import random
import math
import cv2 # Needed for GaussianBlur backend and image loading in albumentations

# --- Install and Import Albumentations ---
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2 # Optional, if using PyTorch later
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

# --- 2. Cross-Validation Splitting (Keep as is, it's good) ---

def create_cross_val_splits(df, n_splits=5, random_state=42): # Increased default splits to 5
    """Creates Stratified Group K-Fold cross-validation splits."""
    print(f"\nCreating {n_splits} stratified group K-Fold splits...")
    # Ensure label is integer type
    df['label'] = df['label'].astype(int)

    # Outer split: StratifiedGroupKFold for Trainval/Test separation by patient
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    # Check if there are enough groups (patients) for each class for the number of splits
    min_groups = df.groupby('label')['patient_id'].nunique().min()
    if min_groups < n_splits:
         print(f"\nWarning: The least populated class has only {min_groups} unique patients, which is less than n_splits={n_splits}. "
               f"StratifiedGroupKFold might fail or produce invalid splits. Consider reducing n_splits or check data.")
         # Optionally, raise an error here if strict adherence is needed.
         raise ValueError(f"Cannot perform {n_splits}-fold stratified group split. Minimum patients per class ({min_groups}) is too low.")


    fold_indices = list(sgkf.split(df, df['label'], df['patient_id']))

    # Inner split: GroupKFold for Train/Validation separation by patient within Trainval
    # We'll do 5 folds internally, but only use the first for train/val split as before
    inner_n_splits = 5
    if inner_n_splits > n_splits : # Heuristic: ensure inner split is feasible
         inner_n_splits = max(2, n_splits) # Adjust inner splits if outer splits are very few

    group_kfold_inner = GroupKFold(n_splits=inner_n_splits)

    for i, (train_val_idx, test_idx) in enumerate(fold_indices):
        train_val_df = df.iloc[train_val_idx].copy() # Use copy to avoid SettingWithCopyWarning
        test_df = df.iloc[test_idx].copy()

        # Perform inner split on train_val_df
        groups_inner = train_val_df['patient_id']
        y_inner = train_val_df['label'] # Stratification isn't directly possible with GroupKFold, but outer split helps

        # Need to check if train_val_df is large enough for inner split
        if len(train_val_df) < inner_n_splits or train_val_df['patient_id'].nunique() < inner_n_splits:
             print(f"Warning: Fold {i+1} - Train+Val set is too small or has too few patients for inner {inner_n_splits}-fold split. Adjusting inner split.")
             # Fallback: Maybe split 80/20 randomly but respecting groups if possible, or just use the first split of a smaller k
             temp_inner_splits = max(2, min(inner_n_splits, train_val_df['patient_id'].nunique()))
             group_kfold_inner_adj = GroupKFold(n_splits=temp_inner_splits)
             inner_split_generator = group_kfold_inner_adj.split(train_val_df, groups=groups_inner)
        else:
             inner_split_generator = group_kfold_inner.split(train_val_df, groups=groups_inner)

        # Get the first train/validation split from the inner generator
        try:
             train_idx_local, val_idx_local = next(inner_split_generator)
             # Convert local indices back to original DataFrame indices if needed, but iloc works on position
             train_df = train_val_df.iloc[train_idx_local]
             val_df = train_val_df.iloc[val_idx_local]
        except StopIteration:
             print(f"Error: Could not generate inner split for Fold {i+1}. Skipping fold.")
             continue # Skip this fold if inner split fails

        splits.append({
            'train_df': train_df.reset_index(drop=True), # Reset index for consistency
            'val_df': val_df.reset_index(drop=True),
            'test_df': test_df.reset_index(drop=True)
        })
        print(f"  Fold {i+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # --- Final Leakage Check ---
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
    os.makedirs(split_dir, exist_ok=True) # Directories should be created before calling

    copy_tasks = []
    for _, row in split_df.iterrows():
        label_name = 'CANCER' if row['label'] == 1 else 'NOT_CANCER'
        image_dest_dir = os.path.join(split_dir, label_name)
        mask_dest_dir = os.path.join(split_dir, f"{label_name}_MASK")
        # Directories for labels created in the main loop

        image_src_path = row['image_path']
        mask_src_path = row['mask_path']
        base_filename = row['filename'] # Use stored filename

        image_dest_path = os.path.join(image_dest_dir, base_filename)
        mask_dest_path = os.path.join(mask_dest_dir, base_filename)

        copy_tasks.append((image_src_path, image_dest_path))
        copy_tasks.append((mask_src_path, mask_dest_path))

    # --- Parallel Copying ---
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor: # I/O bound -> ThreadPool
        future_to_task = {executor.submit(shutil.copy2, src, dst): (src, dst) for src, dst in copy_tasks} # copy2 preserves metadata
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(copy_tasks), desc=f"  Copying {split_name}"):
            src, dst = future_to_task[future]
            try:
                future.result() # Raise exception if copying failed
            except Exception as exc:
                errors.append(f"Failed to copy {src} to {dst}: {exc}")

    if errors:
        print(f"\n  --- Errors during copying for {split_name} ---")
        for error in errors:
            print(f"    {error}")
        print(f"  --- End of Copying Errors ---")
        # Decide if errors are fatal - maybe raise an exception?
        raise RuntimeError(f"Errors occurred during file copying for {split_name}. Check logs.")

def verify_split_integrity(fold_output_dir, split_name, original_df):
    """Performs checks on a created split directory (patient leakage, counts, file existence)."""
    print(f"  Verifying integrity of {split_name} split...")
    split_dir = os.path.join(fold_output_dir, split_name)
    is_train = (split_name == 'TRAIN') # Augmentation check only for train

    all_files_ok = True
    present_patients = set()
    error_messages = []

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(split_dir, label_name)
        mask_dir = os.path.join(split_dir, f"{label_name}_MASK")

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            # This shouldn't happen if creation was successful, but check anyway
             error_messages.append(f"Missing directory: {image_dir} or {mask_dir}")
             all_files_ok = False
             continue # Cannot proceed with checks for this label

        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}

        # --- Check 1: Image and Mask Counts ---
        if len(image_files) != len(mask_files):
            error_messages.append(f"{label_name}: Image count ({len(image_files)}) != Mask count ({len(mask_files)})")
            all_files_ok = False

        # --- Check 2: File Pairing ---
        missing_masks = image_files - mask_files
        missing_images = mask_files - image_files

        if missing_masks:
            error_messages.append(f"{label_name}: Images missing corresponding masks (first 5): {list(missing_masks)[:5]}")
            all_files_ok = False
        if missing_images:
            error_messages.append(f"{label_name}: Masks missing corresponding images (first 5): {list(missing_images)[:5]}")
            all_files_ok = False

        # --- Check 3: Extract Patient IDs from this split ---
        # Also check for unexpected augmented files in val/test
        for img_file in image_files:
             match = re.search(r'PATIENT_(\d+)_', img_file)
             if match:
                 present_patients.add(int(match.group(1)))
             else:
                 # This check might be too strict if augmented filenames don't preserve ID pattern
                 # Consider adjusting if augmentation filenames change drastically
                 # If it's an augmented file, the original filename should be part of it usually
                 is_augmented = '_aug_' in img_file # Simple check based on planned naming
                 if not is_augmented:
                      error_messages.append(f"{label_name}: Cannot parse patient ID from non-augmented file: {img_file}")
                      all_files_ok = False # Decide if this is fatal

             # Check for augmented files in Val/Test
             if not is_train and '_aug_' in img_file:
                  error_messages.append(f"FATAL: Augmented file '{img_file}' found in {split_name}/{label_name}! Data leakage!")
                  all_files_ok = False # This IS fatal

    # --- Check 4: Compare Patient IDs with original DataFrame split ---
    # This requires passing the corresponding original df slice
    original_patient_ids = set(original_df['patient_id'])
    if present_patients != original_patient_ids:
         # This check might fail if some files failed to copy.
         # Let's report the difference.
         if len(present_patients) != len(original_patient_ids): # Only check count differences due to potential copy errors
              error_messages.append(f"Patient ID mismatch: Expected {len(original_patient_ids)} unique patients based on DataFrame, found {len(present_patients)} unique patients in copied files.")
              all_files_ok = False # Might be due to copy errors noted above

    if not all_files_ok:
        print(f"  --- Verification FAILED for {split_name} ---")
        for msg in error_messages:
            print(f"    - {msg}")
        print(f"  --- End Verification Errors ---")
        # Optionally raise an error
        raise ValueError(f"Integrity check failed for {split_name} in fold {os.path.basename(fold_output_dir)}")
        #return False
    else:
        print(f"  Verification PASSED for {split_name}.")
        return True


# --- 4. Augmentation (Using Albumentations) ---

# Define the augmentation pipeline (can be adjusted)
# Values from the paper: brightness, contrast, saturation, hue = 64/255, 0.75, 1.0 (implied?), 0.04 ?
# Max delta for saturation/hue seems low in paper. Let's use reasonable values.
# Brightness max_delta=64/255 ~= 0.25
# Contrast factor range: 1 +/- 0.75 -> (0.25, 1.75) ? Paper text is ambiguous. Let's use a smaller range.
# Saturation: paper says 0.25 delta? Seems very high. Let's assume it meant factor like contrast.
# Hue: paper says 0.04 delta? Hue delta is usually integer 0-180. Let's use a reasonable hue shift limit.

# Adjusted based on common practices and trying to interpret paper:
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # GaussianBlur might make masks less sharp, apply carefully or skip for masks?
    # Albumentations applies blur only to image by default if mask is uint8.
    A.GaussianBlur(blur_limit=(3, 7), p=0.3), # Apply blur less often
    A.ColorJitter(
        brightness=0.25, # Max delta 64/255
        contrast=0.3,    # Factor range (0.7, 1.3) - More conservative than paper interpretation
        saturation=0.3,  # Factor range (0.7, 1.3) - More conservative
        hue=0.04 * 180,  # Shift limit +/- 7 degrees approx. Max delta 0.04 * 180
        p=0.7 # Apply color jitter frequently
    ),
    # Can add more: ElasticTransform, GridDistortion, ShiftScaleRotate etc.
])

def augment_and_save(image_path, mask_path, output_image_dir, output_mask_dir, num_augmentations):
    """Applies augmentations N times to a single image/mask pair and saves."""
    try:
        # Load image and mask using OpenCV (required by Albumentations)
        # Ensure loading in the correct color order (BGR for OpenCV)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # Loads BGR
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load mask as grayscale

        if image is None:
            raise IOError(f"Could not read image file: {image_path}")
        if mask is None:
            raise IOError(f"Could not read mask file: {mask_path}")

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(num_augmentations):
            augmented = transform(image=image, mask=mask)
            augmented_img = augmented['image'] # This is BGR
            augmented_mask = augmented['mask'] # This is Grayscale

            # Define output filenames
            output_image_filename = f"{base_filename}_aug_{i+1}.png"
            output_mask_filename = f"{base_filename}_aug_{i+1}.png" # Same name for mask

            # Save using OpenCV
            cv2.imwrite(os.path.join(output_image_dir, output_image_filename), augmented_img)
            cv2.imwrite(os.path.join(output_mask_dir, output_mask_filename), augmented_mask)

        return True, None # Indicate success

    except Exception as e:
        error_msg = f"Failed augmenting {os.path.basename(image_path)}: {type(e).__name__}: {e}"
        # print(error_msg) # Can be noisy in parallel
        return False, error_msg # Indicate failure and provide message


def augment_and_balance_train_set(fold_output_dir, num_workers=None):
    """Balances the training set by augmenting the minority class."""
    print("  Augmenting and balancing TRAIN set...")
    train_dir = os.path.join(fold_output_dir, "TRAIN")
    cancer_img_dir = os.path.join(train_dir, "CANCER")
    cancer_mask_dir = os.path.join(train_dir, "CANCER_MASK")
    nocancer_img_dir = os.path.join(train_dir, "NOT_CANCER")
    nocancer_mask_dir = os.path.join(train_dir, "NOT_CANCER_MASK") # Mask dir needed for paths

    # --- Count original files (ignore potential previous augmentations) ---
    # cancer_files = {f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png') and '_aug_' not in f}
    # nocancer_files = {f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png') and '_aug_' not in f}

    # Safer: List all pngs and filter based on whether corresponding non-aug file exists
    all_cancer_imgs = {f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png')}
    all_nocancer_imgs = {f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png')}

    # Identify original files (those without _aug_ suffix assuming our naming convention)
    original_cancer_files = {f for f in all_cancer_imgs if '_aug_' not in f}
    original_nocancer_files = {f for f in all_nocancer_imgs if '_aug_' not in f}

    n_cancer = len(original_cancer_files)
    n_nocancer = len(original_nocancer_files)

    print(f"    Original counts: Cancer={n_cancer}, Not_Cancer={n_nocancer}")

    if n_cancer == n_nocancer:
        print("    Classes are already balanced. No augmentation needed for balancing.")
        return True # Indicate success
    if n_cancer == 0 and n_nocancer > 0:
        print("    Warning: No original cancer images found in training set. Cannot balance.")
        return True # Nothing to do, but not failure state?
    if n_nocancer == 0 and n_cancer > 0:
         print("    Warning: No original non-cancer images found in training set. Skipping balancing (or adjust logic if needed).")
         return True


    # --- Determine which class needs augmentation ---
    if n_cancer < n_nocancer:
        minority_img_dir = cancer_img_dir
        minority_mask_dir = cancer_mask_dir
        minority_files = original_cancer_files
        n_minority = n_cancer
        n_majority = n_nocancer
        print(f"    Target: Augmenting CANCER class.")
    else: # n_nocancer < n_cancer
        minority_img_dir = nocancer_img_dir
        minority_mask_dir = os.path.join(train_dir, "NOT_CANCER_MASK") # Define mask dir here
        minority_files = original_nocancer_files
        n_minority = n_nocancer
        n_majority = n_cancer
        print(f"    Target: Augmenting NOT_CANCER class.")


    # --- Calculate needed augmentations ---
    needed_total_augmentations = n_majority - n_minority
    # Ensure n_minority is not zero before division
    if n_minority > 0:
         augmentations_per_sample = math.ceil(needed_total_augmentations / n_minority)
    else:
         # This case should be caught earlier, but defensively:
         print("    Error: Minority class count is zero, cannot proceed with augmentation calculation.")
         return False # Indicate failure


    print(f"    Need {needed_total_augmentations} additional samples for minority class.")
    print(f"    Applying approx. {augmentations_per_sample} augmentations per original minority sample.")

    # --- Prepare tasks for parallel execution ---
    augmentation_tasks = []
    for filename in minority_files:
        image_path = os.path.join(minority_img_dir, filename)
        mask_path = os.path.join(minority_mask_dir, filename) # Assume same name for mask
        if os.path.exists(image_path) and os.path.exists(mask_path):
            augmentation_tasks.append((image_path, mask_path, minority_img_dir, minority_mask_dir, augmentations_per_sample))
        else:
            print(f"    Warning: Skipping augmentation for {filename} - image or mask file missing.")


    if not augmentation_tasks:
         print("    No valid image/mask pairs found for minority class augmentation.")
         return True # Not an error if no files exist


    # --- Execute Augmentation in Parallel ---
    errors = []
    print(f"    Starting parallel augmentation for {len(augmentation_tasks)} files...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create partial function - NOTE: Cannot directly pass 'transform' if it's complex and not pickleable
        # In this case, A.Compose should be pickleable. If not, define transform inside worker.
        # partial_func = partial(augment_and_save, num_augmentations=augmentations_per_sample) # Simplified if all args passed in task tuple

        # Map tasks to the executor
        future_to_task = {executor.submit(augment_and_save, *task): task for task in augmentation_tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(augmentation_tasks), desc="    Augmenting Minority"):
            task_info = future_to_task[future] # Get original task args if needed
            try:
                success, msg = future.result()
                if not success:
                    errors.append(msg)
            except Exception as exc:
                img_p = task_info[0] # Get image path from task tuple
                errors.append(f"Exception during augmentation for {os.path.basename(img_p)}: {exc}")


    # --- Report Augmentation Summary ---
    final_minority_count = len([f for f in os.listdir(minority_img_dir) if f.lower().endswith('.png')])
    print(f"    Augmentation finished. Final minority class count: {final_minority_count}")

    if errors:
        print(f"  --- Errors during Augmentation ---")
        # Print only a few errors to avoid flooding console
        for i, error in enumerate(errors):
            if i < 10: # Print first 10 errors
                 print(f"    - {error}")
            elif i == 10:
                 print(f"    ... (omitting {len(errors)-10} more errors)")
                 break
        print(f"  --- End Augmentation Errors ---")
        # return False # Decide if augmentation errors are fatal

    # Optional: Add a downsampling step if augmentation produces slightly MORE than needed
    final_minority_files = {f for f in os.listdir(minority_img_dir) if f.lower().endswith('.png')}
    current_minority_count = len(final_minority_files)
    if current_minority_count > n_majority:
        num_to_delete = current_minority_count - n_majority
        print(f"    Downsampling: Augmentation created {num_to_delete} extra samples. Randomly deleting...")
        # Only delete *augmented* files to preserve originals
        augmented_files_to_consider = {f for f in final_minority_files if '_aug_' in f}
        if len(augmented_files_to_consider) >= num_to_delete:
             files_to_delete = random.sample(list(augmented_files_to_consider), num_to_delete)
             for filename in files_to_delete:
                 try:
                     os.remove(os.path.join(minority_img_dir, filename))
                     os.remove(os.path.join(minority_mask_dir, filename)) # Delete corresponding mask
                 except OSError as e:
                     print(f"    Warning: Failed to delete extra augmented file {filename}: {e}")
             print(f"    Downsampling complete. Final count should be {n_majority}.")
        else:
             print(f"    Warning: Not enough augmented files ({len(augmented_files_to_consider)}) to delete {num_to_delete}. Count might be slightly off.")


    print("  Train set balancing and augmentation complete.")
    return True


# --- 5. Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    data_directory = r'D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED' # Input dir with CANCER, NOT_CANCER, CANCER_MASK, NOT_CANCER_MASK
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits_balanced' # Output for folds
    N_SPLITS = 5 # Number of folds (e.g., 5)
    RANDOM_STATE = 42 # For reproducibility
    NUM_WORKERS = os.cpu_count() # Use all available CPU cores for parallel tasks

    # --- Prepare Output Directory ---
    if os.path.exists(output_base_dir):
        print(f"Output directory '{output_base_dir}' already exists.")
        # Decide action: remove it, ask user, or add timestamp?
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


    # --- Create Splits ---
    try:
        cross_val_splits = create_cross_val_splits(data_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    except Exception as e:
        print(f"Error creating cross-validation splits: {e}")
        # Add more specific error handling if StratifiedGroupKFold fails due to class imbalance
        if "received fewer than" in str(e) or "n_splits=" in str(e):
             print("  This might be due to having too few patients in one class for the requested number of splits.")
             print(f"  Try reducing N_SPLITS (currently {N_SPLITS}).")
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
        os.makedirs(fold_output_dir, exist_ok=True)

        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']

        fold_successful = True

        # --- Create Subdirectories and Copy Files ---
        for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
            split_dir = os.path.join(fold_output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            # Create label subdirs
            for label_name in ['CANCER', 'NOT_CANCER']:
                 os.makedirs(os.path.join(split_dir, label_name), exist_ok=True)
                 os.makedirs(os.path.join(split_dir, f"{label_name}_MASK"), exist_ok=True)

            try:
                copy_files_for_split(split_df_current, fold_output_dir, split_name)
            except Exception as e:
                 print(f"  ERROR during file copying for {split_name}: {e}")
                 fold_successful = False
                 break # Stop processing this fold if copying fails

        if not fold_successful:
             all_folds_successful = False
             print(f"--- Skipping further processing for Fold {fold_num} due to errors. ---")
             continue # Move to the next fold

        # --- Augment and Balance TRAINING Set ---
        try:
            balance_success = augment_and_balance_train_set(fold_output_dir, num_workers=NUM_WORKERS)
            if not balance_success:
                 print(f"  Warning: Augmentation/Balancing reported issues for Fold {fold_num}.")
                 # Decide if this should be fatal
                 fold_successful = False
        except Exception as e:
            print(f"  ERROR during augmentation/balancing for Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            fold_successful = False


        # --- Final Verification for the Fold ---
        print(f"\n--- Final Verification for Fold {fold_num} ---")
        verification_passed = True
        for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
            if not verify_split_integrity(fold_output_dir, split_name, split_df_current):
                verification_passed = False
                fold_successful = False # Mark fold as failed if verification fails


        if fold_successful and verification_passed:
             print(f"--- Fold {fold_num} completed successfully. ---")
        else:
             all_folds_successful = False
             print(f"--- Fold {fold_num} completed with ERRORS. Please review logs/output. ---")


    # --- Final Summary ---
    print("\n--- Overall Process Summary ---")
    if all_folds_successful:
        print("All folds processed and verified successfully!")
        print(f"Balanced cross-validation splits are located in: {output_base_dir}")
    else:
        print("Processing completed, but ERRORS occurred in one or more folds.")
        print("Please review the output messages and logs above to identify issues.")
        print(f"Output directory: {output_base_dir}")

    print("--- Script Finished ---")