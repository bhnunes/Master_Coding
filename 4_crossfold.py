import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
# from PIL import Image, UnidentifiedImageError # PIL no longer strictly needed if using cv2 consistently
import shutil
import re
import concurrent.futures
# from functools import partial # No longer needed for partial
from tqdm import tqdm
import random
import math
import cv2 # Needed for GaussianBlur backend and image loading/saving

# --- Install and Import Albumentations ---
try:
    import albumentations as A
    # from albumentations.pytorch import ToTensorV2 # Optional, if using PyTorch later
except ImportError:
    print("Albumentations library not found. Please install it: pip install -U albumentations")
    exit()


# --- 1. Data Loading and Preparation (Robust) ---
# (Keep load_data function as is)
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
# (Keep create_cross_val_splits function as is)
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
# (Keep copy_files_for_split function as is)
def copy_files_for_split(split_df, fold_output_dir, split_name):
    """Copies original images and masks for a given split (train/val/test)."""
    print(f"  Copying original files for {split_name}...")
    split_dir = os.path.join(fold_output_dir, split_name)
    # Directories should be created before calling this function in the main loop

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
    # Use more threads for I/O bound tasks like copying
    num_copy_workers = min(32, (os.cpu_count() or 1) + 4) # Common heuristic for ThreadPool
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
        for error in errors[:10]: print(f"    {error}") # Print first few errors
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
    # Updated regex for single augmentation code: _aug_XX_N.png
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

        # Check patient IDs and augmentation presence
        for img_file in image_files:
             match_patient = re.search(r'PATIENT_(\d+)_', img_file)
             if match_patient:
                 present_patients.add(int(match_patient.group(1)))
             else:
                 # Only flag non-parsing if it's NOT an augmented file
                 is_augmented = bool(aug_pattern_verify.search(img_file))
                 if not is_augmented:
                      error_messages.append(f"{label_name}: Cannot parse patient ID from non-augmented file: {img_file}")
                      all_files_ok = False

             # Check for augmented files in Val/Test
             if not is_train and aug_pattern_verify.search(img_file):
                  error_messages.append(f"FATAL LEAKAGE: Augmented file '{img_file}' found in {split_name}/{label_name}!")
                  all_files_ok = False # This IS fatal

    # Compare Patient IDs with original DataFrame split
    original_patient_ids = set(original_df['patient_id'])
    if present_patients != original_patient_ids:
         # Report difference only if counts mismatch, allowing for potential copy errors of *some* files
         if len(present_patients) != len(original_patient_ids):
              error_messages.append(f"Patient ID count mismatch: Expected {len(original_patient_ids)} unique patients (from DF), found {len(present_patients)} (in files). Possible copy issue or parsing error.")
              # Decide if this is fatal - depends on tolerance for copy errors
              all_files_ok = False # Let's consider it an error for now

    if not all_files_ok:
        print(f"  --- Verification FAILED for {split_name} ---")
        for msg in error_messages: print(f"    - {msg}")
        print(f"  --- End Verification Errors ---")
        raise ValueError(f"Integrity check failed for {split_name} in fold {os.path.basename(fold_output_dir)}")
    else:
        print(f"  Verification PASSED for {split_name}.")
        return True


# --- 4. Augmentation (Using Albumentations - Single Random Transform) ---

# Define the INDIVIDUAL transformations, each with p=1.0 (always apply if chosen)
# Keep parameters as before, or adjust if needed for the specific transform
INDIVIDUAL_TRANSFORMS = [
    (A.HorizontalFlip(p=1.0), "HP"),
    (A.VerticalFlip(p=1.0), "VF"),
    (A.RandomRotate90(p=1.0), "RF"),
    (A.GaussianBlur(blur_limit=(3, 7), p=1.0), "GB"),
    (A.ColorJitter(
        brightness=0.25,
        contrast=0.3,
        saturation=0.3,
        hue=0.04 * 180, # Use degrees for OpenCV backend
        p=1.0 # Apply jitter parameters if this transform is chosen
    ), "CJ")
    # Add more individual transforms here if desired, e.g.:
    # (A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=1.0), "SSR"),
    # (A.ElasticTransform(p=1.0, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), "ET")
]

print(f"Defined {len(INDIVIDUAL_TRANSFORMS)} individual augmentations for random selection:")
for _, code in INDIVIDUAL_TRANSFORMS:
    print(f"  - {code}")

def augment_and_save(image_path, mask_path, output_image_dir, output_mask_dir, num_augmentations):
    """
    Applies ONE randomly selected augmentation N times to a single image/mask pair
    and saves using filenames indicating the single applied transform.
    """
    try:
        # Load original image and mask ONCE using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise IOError(f"Could not read image file: {image_path}")
        if mask is None:
            raise IOError(f"Could not read mask file: {mask_path}")

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Generate N augmented versions, each with ONE random transform applied
        for i in range(num_augmentations):
            # *** Randomly select ONE transform for this iteration ***
            chosen_transform, chosen_code = random.choice(INDIVIDUAL_TRANSFORMS)

            # Apply the single chosen transform
            augmented = chosen_transform(image=image.copy(), mask=mask.copy()) # Apply to copies
            augmented_img = augmented['image']
            augmented_mask = augmented['mask']

            # Define output filenames indicating the SINGLE applied transform code
            # Format: basename_aug_CODE_N.png
            output_image_filename = f"{base_filename}_aug_{chosen_code}_{i+1}.png"
            output_mask_filename = f"{base_filename}_aug_{chosen_code}_{i+1}.png" # Use same name format

            # Save the result of this single transformation
            img_save_path = os.path.join(output_image_dir, output_image_filename)
            mask_save_path = os.path.join(output_mask_dir, output_mask_filename)

            cv2.imwrite(img_save_path, augmented_img)
            # Ensure mask is saved correctly (consider checking dtype if issues arise)
            cv2.imwrite(mask_save_path, augmented_mask)

        return True, None # Indicate success for the original image pair

    except Exception as e:
        # Capture more specific errors if possible
        error_msg = f"Failed augmenting {os.path.basename(image_path)} (iter {i+1 if 'i' in locals() else 'N/A'}, transform {chosen_code if 'chosen_code' in locals() else 'N/A'}): {type(e).__name__}: {e}"
        return False, error_msg # Indicate failure and provide message


def augment_and_balance_train_set(fold_output_dir, num_workers=None):
    """Balances the training set by augmenting the minority class using single random transforms."""
    print("  Augmenting and balancing TRAIN set (using single random transform per generated image)...")
    train_dir = os.path.join(fold_output_dir, "TRAIN")
    cancer_img_dir = os.path.join(train_dir, "CANCER")
    cancer_mask_dir = os.path.join(train_dir, "CANCER_MASK")
    nocancer_img_dir = os.path.join(train_dir, "NOT_CANCER")
    nocancer_mask_dir = os.path.join(train_dir, "NOT_CANCER_MASK")

    # Updated regex for single augmentation code: _aug_XX_N.png
    aug_pattern_balance = re.compile(r'_aug_([A-Z]{2})_\d+\.png$')

    original_cancer_files = {f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png') and not aug_pattern_balance.search(f)}
    original_nocancer_files = {f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png') and not aug_pattern_balance.search(f)}

    n_cancer = len(original_cancer_files)
    n_nocancer = len(original_nocancer_files)

    print(f"    Original counts: Cancer={n_cancer}, Not_Cancer={n_nocancer}")

    if n_cancer == n_nocancer:
        print("    Classes are already balanced. No augmentation needed.")
        return True
    if n_minority := min(n_cancer, n_nocancer) == 0:
        print(f"    Warning: Minority class ({'CANCER' if n_cancer == 0 else 'NOT_CANCER'}) has 0 samples. Cannot balance.")
        return True # Nothing to augment

    # Determine minority/majority
    if n_cancer < n_nocancer:
        minority_img_dir, minority_mask_dir = cancer_img_dir, cancer_mask_dir
        minority_files, n_minority, n_majority = original_cancer_files, n_cancer, n_nocancer
        print(f"    Target: Augmenting CANCER class.")
    else:
        minority_img_dir, minority_mask_dir = nocancer_img_dir, nocancer_mask_dir
        minority_files, n_minority, n_majority = original_nocancer_files, n_nocancer, n_cancer
        print(f"    Target: Augmenting NOT_CANCER class.")

    # Calculate needed augmentations
    needed_total_augmentations = n_majority - n_minority
    # This is the number of times we need to call augment_and_save *per original minority image* on average
    augmentations_per_original_sample = math.ceil(needed_total_augmentations / n_minority)

    print(f"    Need {needed_total_augmentations} additional samples for minority class.")
    print(f"    Applying {augmentations_per_original_sample} randomly chosen single augmentations per original minority sample.")

    # Prepare tasks for parallel execution
    augmentation_tasks = []
    for filename in minority_files:
        image_path = os.path.join(minority_img_dir, filename)
        mask_path = os.path.join(minority_mask_dir, filename) # Assume same name for mask
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Pass only the number of augmentations needed per original file
            augmentation_tasks.append((image_path, mask_path, minority_img_dir, minority_mask_dir, augmentations_per_original_sample))
        else:
            print(f"    Warning: Skipping augmentation for {filename} - image or mask file missing.")

    if not augmentation_tasks:
         print("    No valid image/mask pairs found for minority class augmentation.")
         return True # Not an error if no files exist

    # --- Execute Augmentation in Parallel ---
    errors = []
    print(f"    Starting parallel augmentation for {len(augmentation_tasks)} original files...")
    # Use ProcessPoolExecutor for CPU-bound augmentation tasks
    actual_num_workers = os.cpu_count() if num_workers is None else num_workers
    print(f"    Using {actual_num_workers} worker processes.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_num_workers) as executor:
        # No need for partial, just submit the task tuple
        future_to_task = {executor.submit(augment_and_save, *task): task for task in augmentation_tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(augmentation_tasks), desc="    Augmenting Minority"):
            task_info = future_to_task[future]
            try:
                success, msg = future.result()
                if not success:
                    errors.append(msg)
            except Exception as exc:
                img_p = task_info[0] # Get image path from task tuple
                errors.append(f"Exception during augmentation task for {os.path.basename(img_p)}: {exc}")

    # --- Report Augmentation Summary ---
    final_minority_images = {f for f in os.listdir(minority_img_dir) if f.lower().endswith('.png')}
    final_minority_count = len(final_minority_images)
    print(f"    Augmentation finished. Final minority class count: {final_minority_count}")

    if errors:
        print(f"  --- Errors during Augmentation ---")
        for i, error in enumerate(errors):
            if i < 10: print(f"    - {error}")
            elif i == 10: print(f"    ... (omitting {len(errors)-10} more errors)")
            break
        print(f"  --- End Augmentation Errors ---")
        # return False # Decide if augmentation errors are fatal

    # Optional: Downsampling to exact count (more precise now)
    if final_minority_count > n_majority:
        num_to_delete = final_minority_count - n_majority
        print(f"    Downsampling: Augmentation created {num_to_delete} extra samples. Randomly deleting...")
        # Only delete *augmented* files
        augmented_files_to_consider = {f for f in final_minority_images if aug_pattern_balance.search(f)}

        if len(augmented_files_to_consider) >= num_to_delete:
             # Convert set to list for random.sample
             files_to_delete = random.sample(list(augmented_files_to_consider), num_to_delete)
             delete_errors = 0
             for filename in files_to_delete:
                 try:
                     os.remove(os.path.join(minority_img_dir, filename))
                     os.remove(os.path.join(minority_mask_dir, filename)) # Delete corresponding mask
                 except OSError as e:
                     print(f"    Warning: Failed to delete extra augmented file {filename}: {e}")
                     delete_errors += 1
             print(f"    Downsampling complete. Aiming for {n_majority} samples. {delete_errors} errors during deletion.")
        else:
             # This case is less likely with ceil but possible if many originals failed processing
             print(f"    Warning: Not enough augmented files ({len(augmented_files_to_consider)}) to delete {num_to_delete}. Final count might be slightly off.")

    print("  Train set balancing and augmentation complete.")
    # You can add a final count verification here if needed
    final_count_after_downsample = len([f for f in os.listdir(minority_img_dir) if f.lower().endswith('.png')])
    print(f"    Final minority count after potential downsampling: {final_count_after_downsample} (Majority count: {n_majority})")
    return True


# --- 5. Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    data_directory = r'D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED'
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits_balanced_geometric_aug'
    N_SPLITS = 5
    RANDOM_STATE = 42
    # Adjust workers based on CPU capability; augmentation is CPU-bound
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1) # Leave one core free

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
        # Base fold dir created implicitly by makedirs below if needed

        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']

        fold_successful = True

        # --- Create Subdirectories and Copy Files ---
        print(f"Fold {fold_num}: Setting up directories and copying original files...")
        for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
            split_dir = os.path.join(fold_output_dir, split_name)
            # Create directories including label subdirs *before* copying
            try:
                os.makedirs(os.path.join(split_dir, 'CANCER'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'CANCER_MASK'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'NOT_CANCER'), exist_ok=True)
                os.makedirs(os.path.join(split_dir, 'NOT_CANCER_MASK'), exist_ok=True)
            except OSError as e:
                print(f"  ERROR creating directories for {split_name} in Fold {fold_num}: {e}")
                fold_successful = False
                break # Cannot proceed with this fold

            if not fold_successful: break

            try:
                copy_files_for_split(split_df_current, fold_output_dir, split_name)
            except Exception as e:
                 print(f"  ERROR during file copying for {split_name} in Fold {fold_num}: {e}")
                 fold_successful = False
                 break # Stop processing this fold if copying fails

        if not fold_successful:
             all_folds_successful = False
             print(f"--- Skipping further processing for Fold {fold_num} due to setup/copying errors. ---")
             # Optional: Clean up partially created fold directory
             # shutil.rmtree(fold_output_dir, ignore_errors=True)
             continue # Move to the next fold

        # --- Augment and Balance TRAINING Set ---
        print(f"\nFold {fold_num}: Augmenting and balancing training set...")
        try:
            balance_success = augment_and_balance_train_set(fold_output_dir, num_workers=NUM_WORKERS)
            if not balance_success:
                 print(f"  Warning: Augmentation/Balancing function reported potential issues for Fold {fold_num}, but proceeding.")
                 # Decide if this should stop the fold processing
                 # fold_successful = False # Or just log it
        except Exception as e:
            print(f"  FATAL ERROR during augmentation/balancing for Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            fold_successful = False


        # --- Final Verification for the Fold ---
        if fold_successful: # Only verify if previous steps seemed okay
            print(f"\n--- Final Verification for Fold {fold_num} ---")
            verification_passed = True
            for split_name, split_df_current in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):
                try:
                    if not verify_split_integrity(fold_output_dir, split_name, split_df_current):
                        verification_passed = False
                        fold_successful = False # Mark fold as failed if verification fails
                except Exception as e:
                     print(f"  ERROR during verification for {split_name} in Fold {fold_num}: {e}")
                     import traceback
                     traceback.print_exc()
                     verification_passed = False
                     fold_successful = False # Treat verification error as fold failure

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
        print("Each augmented image in the TRAIN sets was generated using exactly ONE randomly selected transformation.")
        print(f"Output location: {output_base_dir}")
    else:
        print("Processing completed, but ERRORS occurred in one or more folds.")
        print("Please review the output messages and logs above to identify issues.")
        print(f"Output directory (may contain partial or erroneous data): {output_base_dir}")

    print("--- Script Finished ---")