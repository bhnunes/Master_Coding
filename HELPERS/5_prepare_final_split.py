import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit # Use this for single Train/Val split by group
import shutil
import re
import concurrent.futures
from tqdm import tqdm
import random
import math
import cv2

# --- Install and Import Albumentations ---
try:
    import albumentations as A
except ImportError:
    print("Albumentations library not found. Please install it: pip install -U albumentations")
    exit()

# --- 1. Data Loading and Preparation ---
# (Keep load_data function as is)
def load_data(data_dir):
    # ... (load_data function code remains the same) ...
    print(f"Loading data from: {data_dir}")
    data = [] 
    expected_image_dirs = [os.path.join(data_dir, "CANCER"), os.path.join(data_dir, "NOT_CANCER")] 
    expected_mask_dirs = [os.path.join(data_dir, "CANCER_MASK"), os.path.join(data_dir, "NOT_CANCER_MASK")]
    
    for image_dir, mask_dir in zip(expected_image_dirs, expected_mask_dirs):
        label_name = os.path.basename(image_dir) 
        label = 1 if label_name == "CANCER" else 0
        if not os.path.isdir(image_dir): 
            print(f"Warn: Img dir miss: {image_dir}") 
            continue
        if not os.path.isdir(mask_dir): 
            print(f"Warn: Mask dir miss: {mask_dir}") 
            continue
        print(f" Scan {label_name} in {os.path.basename(image_dir)}...") 
        
        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')} 
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')} 
        
        print(f"  Found {len(image_files)} imgs, {len(mask_files)} masks.")
        
        for image_name in tqdm(image_files, desc=f"Proc {label_name}"):
            mask_name = image_name

            if mask_name not in mask_files: 
                print(f"Warn: Mask '{mask_name}' miss for '{image_name}'. Skip.") 
                continue
            
            image_path = os.path.join(image_dir, image_name) 
            mask_path = os.path.join(mask_dir, image_name)
            match = re.search(r'PATIENT_(\d+)_', image_name)
            
            if match: 
                try: 
                    patient_id = int(match.group(1)) 
                except ValueError: 
                    print(f"Warn: PID conv fail {image_name}. Skip.") 
                    continue
            else: 
                print(f"Warn: PID parse fail {image_name}. Skip.") 
                continue
            
            if not os.path.isfile(image_path) or not os.path.isfile(mask_path): 
                print(f"Warn: Path invalid {image_name}. Skip.") 
                continue
            
            data.append({'patient_id':patient_id, 'image_path':image_path, 'mask_path':mask_path, 'label':label, 'filename':image_name})
    
    if not data: 
        raise ValueError(f"No valid data found in {data_dir}.")
    
    df = pd.DataFrame(data) 
    
    print(f"Loaded {len(df)} pairs for {df['patient_id'].nunique()} patients.") 
    print(f"Class dist:\n{df['label'].value_counts()}") 
    return df


# --- 2. Create Single Train/Validation Split ---
def create_train_val_split(df, train_size=0.8, random_state=42):
    """Creates a single Train/Validation split based on patient IDs."""
    print(f"\nCreating single Train/Validation split (Train Size: {train_size:.1%}) using patient groups...")
    df['label'] = df['label'].astype(int) # Ensure label type

    # We need only one split, so n_splits=1
    # test_size is the proportion for the validation set here
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)

    # Get the indices for the split
    try:
        train_idx, val_idx = next(gss.split(df, df['label'], groups=df['patient_id']))
    except ValueError as e:
         print(f"Error during GroupShuffleSplit: {e}")
         if "cannot be greater than the number of groups" in str(e):
              print(" Ensure train_size is appropriate and there are enough unique patients.")
         raise # Re-raise the error

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    print(f"Split complete: Train={len(train_df)} ({train_df['patient_id'].nunique()} patients), Validation={len(val_df)} ({val_df['patient_id'].nunique()} patients)")

    # --- Verification: Patient Leakage ---
    print("Verifying patient separation between Train and Validation...")
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise ValueError(f"FATAL: Patient leakage detected between Train and Validation! Overlapping IDs: {overlap}")
    else:
        print("Patient separation verified.")

    return train_df, val_df


# --- 3. File Operations & Verification ---
# (Keep copy_files_for_split function as is)
def copy_files_for_split(split_df, output_split_dir): # Modified args
    """Copies original images and masks for a given split DataFrame to the output dir."""
    print(f"  Copying original files for split...")
    # Directories (TRAIN/CANCER, TRAIN/NOT_CANCER etc.) should be created before calling

    copy_tasks = []
    for _, row in split_df.iterrows():
        label_name = 'CANCER' if row['label'] == 1 else 'NOT_CANCER'
        image_dest_dir = os.path.join(output_split_dir, label_name) # Directly use output_split_dir
        mask_dest_dir = os.path.join(output_split_dir, f"{label_name}_MASK")

        image_src_path = row['image_path']
        mask_src_path = row['mask_path']
        base_filename = row['filename']

        image_dest_path = os.path.join(image_dest_dir, base_filename)
        mask_dest_path = os.path.join(mask_dest_dir, base_filename)

        # Ensure destination directories exist (should be created outside this function)
        if not os.path.isdir(image_dest_dir): 
            os.makedirs(image_dest_dir, exist_ok=True)
        if not os.path.isdir(mask_dest_dir): 
            os.makedirs(mask_dest_dir, exist_ok=True)

        copy_tasks.append((image_src_path, image_dest_path))
        copy_tasks.append((mask_src_path, mask_dest_path))

    # ... (Keep the parallel copying logic with ThreadPoolExecutor and error handling) ...
    errors = []
    num_copy_workers = min(32, (os.cpu_count() or 1) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_copy_workers) as executor:
        future_to_task = {executor.submit(shutil.copy2, src, dst): (src, dst) for src, dst in copy_tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(copy_tasks), desc=f"  Copying"):
            src, dst = future_to_task[future]
            try: 
                future.result()
            except Exception as exc: 
                errors.append(f"Failed copy {src} -> {dst}: {exc}")
    if errors: 
        print(f"\n Copy Errors:\n" + "\n".join(errors[:10]) + ("..." if len(errors)>10 else "")) 
        raise RuntimeError("Errors during file copying.")


def verify_split_integrity(output_split_dir, original_df): # Modified args
    """Performs checks on a created split directory (counts, file existence, NO augmentation check needed here)."""
    print(f"  Verifying integrity of split: {os.path.basename(output_split_dir)}...")
    all_files_ok = True
    error_messages = []
    # Updated regex for expected augmentation format (used by balancer)
    aug_pattern_verify = re.compile(r'_aug_([A-Z]{2})_\d+\.png$')

    split_file_count = 0 # Count total files copied
    present_patients = set() # Track patients in this split

    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(output_split_dir, label_name)
        mask_dir = os.path.join(output_split_dir, f"{label_name}_MASK")

        if not os.path.isdir(image_dir): 
            error_messages.append(f"Missing: {image_dir}") 
            all_files_ok=False 
            continue
        if not os.path.isdir(mask_dir): 
            error_messages.append(f"Missing: {mask_dir}") 
            all_files_ok=False 
            continue

        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}

        split_file_count += len(image_files) # Add images to total count for this split

        if len(image_files) != len(mask_files):
            error_messages.append(f"{label_name}: Img count ({len(image_files)}) != Mask count ({len(mask_files)})")
            all_files_ok = False

        missing_masks = image_files - mask_files
        missing_images = mask_files - image_files
        if missing_masks: 
            error_messages.append(f"{label_name}: Imgs miss masks (e.g., {list(missing_masks)[:3]})") 
            all_files_ok=False
        if missing_images: 
            error_messages.append(f"{label_name}: Masks miss imgs (e.g., {list(missing_images)[:3]})") 
            all_files_ok=False

        # Extract patient IDs from this split
        for img_file in image_files:
             match_patient = re.search(r'PATIENT_(\d+)_', img_file)
             if match_patient: 
                present_patients.add(int(match_patient.group(1)))
             else: # Only flag if it's NOT expected to be augmented (check TRAIN later)
                  is_augmented = bool(aug_pattern_verify.search(img_file))
                  if not is_augmented:
                       error_messages.append(f"{label_name}: Cannot parse patient ID: {img_file}")
                       all_files_ok = False

    # Compare total file count and patient count with original DataFrame slice
    if split_file_count != len(original_df):
        error_messages.append(f"File count mismatch: Expected {len(original_df)} images (from DF), found {split_file_count} (in files). Possible copy issue.")
        all_files_ok = False
    original_patient_ids = set(original_df['patient_id'])
    if len(present_patients) != len(original_patient_ids):
        error_messages.append(f"Patient ID count mismatch: Expected {len(original_patient_ids)} unique patients (from DF), found {len(present_patients)} (in files). Possible copy/parsing issue.")
        all_files_ok = False


    if not all_files_ok:
        print(f"  --- Verification FAILED for {os.path.basename(output_split_dir)} ---")
        for msg in error_messages: print(f"    - {msg}")
        print(f"  --- End Verification Errors ---")
        raise ValueError(f"Integrity check failed for {os.path.basename(output_split_dir)}")
    else:
        print(f"  Verification PASSED for {os.path.basename(output_split_dir)}.")
        return True

def verify_train_balance_and_masks(train_output_dir):
    """Verifies final balance and mask integrity in the TRAIN directory after augmentation."""
    print(f"  Verifying final balance and mask integrity for: TRAIN")
    all_files_ok = True
    error_messages = []
    aug_pattern_verify = re.compile(r'_aug_([A-Z]{2})_\d+\.png$')

    cancer_img_dir = os.path.join(train_output_dir, 'CANCER')
    cancer_mask_dir = os.path.join(train_output_dir, 'CANCER_MASK')
    nocancer_img_dir = os.path.join(train_output_dir, 'NOT_CANCER')
    nocancer_mask_dir = os.path.join(train_output_dir, 'NOT_CANCER_MASK')

    cancer_images = {f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png')}
    cancer_masks = {f for f in os.listdir(cancer_mask_dir) if f.lower().endswith('.png')}
    nocancer_images = {f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png')}
    nocancer_masks = {f for f in os.listdir(nocancer_mask_dir) if f.lower().endswith('.png')}

    # Check Balance
    if len(cancer_images) != len(nocancer_images):
        error_messages.append(f"BALANCE CHECK FAILED: Cancer images ({len(cancer_images)}) != Non-Cancer images ({len(nocancer_images)})")
        all_files_ok = False
    else:
         print(f"    Balance check PASSED: {len(cancer_images)} images per class.")

    # Check Mask Correspondence per class
    for label_name, img_set, mask_set in [("CANCER", cancer_images, cancer_masks), ("NOT_CANCER", nocancer_images, nocancer_masks)]:
        if len(img_set) != len(mask_set):
            error_messages.append(f"{label_name} MASK CHECK FAILED: Img count ({len(img_set)}) != Mask count ({len(mask_set)})")
            all_files_ok = False

        missing_masks = img_set - mask_set
        missing_images = mask_set - img_set
        if missing_masks: 
            error_messages.append(f"{label_name} MASK CHECK FAILED: Images miss masks (e.g., {list(missing_masks)[:3]})") 
            all_files_ok=False
        if missing_images: 
            error_messages.append(f"{label_name} MASK CHECK FAILED: Masks miss images (e.g., {list(missing_images)[:3]})") 
            all_files_ok=False

    if not all_files_ok:
        print(f"  --- FINAL TRAIN Verification FAILED ---")
        for msg in error_messages: print(f"    - {msg}")
        print(f"  --- End Verification Errors ---")
        raise ValueError(f"Final train set verification failed in {train_output_dir}")
    else:
        print(f"  Final TRAIN set verification PASSED.")
        return True


# --- 4. Augmentation ---
# (Keep INDIVIDUAL_TRANSFORMS definition)
INDIVIDUAL_TRANSFORMS = [ # ... (same definition as before) ...
    (A.HorizontalFlip(p=1.0), "HP"), (A.VerticalFlip(p=1.0), "VF"), (A.RandomRotate90(p=1.0), "RF"),
    (A.GaussianBlur(blur_limit=(3,7),p=1.0),"GB"),
    (A.ColorJitter(brightness=0.25,contrast=0.3,saturation=0.3,hue=0.04*180,p=1.0),"CJ") ]

# (Keep augment_and_save function as is)
def augment_and_save(image_path, mask_path, output_image_dir, output_mask_dir, num_augmentations):
    # ... (augment_and_save function code remains the same) ...
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None: 
            raise IOError(f"Img read fail: {image_path}")
        if mask is None: 
            raise IOError(f"Mask read fail: {mask_path}")
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        for i in range(num_augmentations):
            chosen_transform, chosen_code = random.choice(INDIVIDUAL_TRANSFORMS)
            try: 
                augmented = chosen_transform(image=image.copy(), mask=mask.copy())
            except Exception as aug_e: 
                print(f"Warn: Aug {chosen_code} failed for {base_filename}: {aug_e}") 
                continue # Skip this aug attempt
            augmented_img, augmented_mask = augmented['image'], augmented['mask']
            out_img_fn = f"{base_filename}_aug_{chosen_code}_{i+1}.png" 
            out_mask_fn = out_img_fn
            img_save_path = os.path.join(output_image_dir, out_img_fn) 
            mask_save_path = os.path.join(output_mask_dir, out_mask_fn)
            cv2.imwrite(img_save_path, augmented_img) 
            cv2.imwrite(mask_save_path, augmented_mask)
        return True, None
    except Exception as e: 
        return False, f"Aug ERR {os.path.basename(image_path)}: {e}"

# (Keep augment_and_balance_train_set function as is)
def augment_and_balance_train_set(train_output_dir, num_workers=None): # Takes TRAIN dir as input
    # ... (augment_and_balance_train_set function code remains the same) ...
    # (It calculates minority/majority within the train_output_dir and augments/downsamples)
    print(" Balancing TRAIN set...") 
    train_dir=train_output_dir # Use passed dir
    cancer_img_dir=os.path.join(train_dir,"CANCER") 
    cancer_mask_dir=os.path.join(train_dir,"CANCER_MASK") 
    nocancer_img_dir=os.path.join(train_dir,"NOT_CANCER") 
    nocancer_mask_dir=os.path.join(train_dir,"NOT_CANCER_MASK")
    aug_pattern=re.compile(r'_aug_([A-Z]{2})_\d+\.png$')
    orig_cancer={f for f in os.listdir(cancer_img_dir) if f.lower().endswith('.png') and not aug_pattern.search(f)} 
    orig_nocancer={f for f in os.listdir(nocancer_img_dir) if f.lower().endswith('.png') and not aug_pattern.search(f)}
    n_cancer,n_nocancer=len(orig_cancer),len(orig_nocancer) 
    print(f"  Orig counts: C={n_cancer}, NC={n_nocancer}")
    
    if n_cancer==n_nocancer: 
        print("  Already balanced.") 
        return True
    
    n_minority=min(n_cancer,n_nocancer)
    if n_minority==0: 
        print(" Warn: Minority class has 0 samples. Cannot balance.") 
        return True
    if n_cancer<n_nocancer: 
        minor_img_dir,minor_mask_dir,minor_files,n_minor,n_major=cancer_img_dir,cancer_mask_dir,orig_cancer,n_cancer,n_nocancer 
        print("  Augmenting CANCER.")
    else: 
        minor_img_dir,minor_mask_dir,minor_files,n_minor,n_major=nocancer_img_dir,nocancer_mask_dir,orig_nocancer,n_nocancer,n_cancer 
        print("  Augmenting NOT_CANCER.")
    needed=n_major-n_minor 
    augs_per_orig=math.ceil(needed/n_minor) 
    print(f"  Need {needed} more samples. Apply ~{augs_per_orig} augs per orig.")
    
    tasks=[(os.path.join(minor_img_dir,f),os.path.join(minor_mask_dir,f),minor_img_dir,minor_mask_dir,augs_per_orig) for f in minor_files if os.path.exists(os.path.join(minor_img_dir,f)) and os.path.exists(os.path.join(minor_mask_dir,f))]
    
    if not tasks: 
        print("  No valid files found for augmentation.") 
        return True
    
    errors=[] 
    print(f"  Starting parallel aug for {len(tasks)} files...") 
    workers=os.cpu_count() if num_workers is None else num_workers
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task={executor.submit(augment_and_save,*task):task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task),total=len(tasks),desc="  Augmenting"): 
            task_info=future_to_task[future] 
            try: 
                success,msg=future.result() 
                errors.append(msg) if not success else None 
            except Exception as exc: 
                errors.append(f"Exception aug {os.path.basename(task_info[0])}: {exc}")
    
    final_minor_files={f for f in os.listdir(minor_img_dir) if f.lower().endswith('.png')} 
    final_minor_count=len(final_minor_files) 
    print(f"  Aug finished. Final minority count: {final_minor_count}")
    
    if errors: 
        print(f"  Aug Errors:{len(errors)}") 
        print('\n'.join(errors[:5])) 
        print("...") if len(errors)>5 else None
    
    if final_minor_count>n_major:
        num_del=final_minor_count-n_major 
        print(f"  Downsampling: {num_del} extra samples. Deleting...") 
        aug_files_del={f for f in final_minor_files if aug_pattern.search(f)}
        
        if len(aug_files_del)>=num_del: 
            files_to_del=random.sample(list(aug_files_del),num_del) 
            del_errs=0
        
            for f in files_to_del: 
                try: 
                    os.remove(os.path.join(minor_img_dir,f)) 
                    os.remove(os.path.join(minor_mask_dir,f)) 
                except OSError as e: 
                    print(f" Warn: Del fail {f}: {e}") 
                    del_errs+=1
            
            print(f"  Downsampling done. Aim {n_major} samples. {del_errs} errors.")
        
        else: print(f" Warn: Not enough augmented files ({len(aug_files_del)}) to delete {num_del}.")
    
    print("  Balancing complete.") 
    final_count_ds=len([f for f in os.listdir(minor_img_dir) if f.lower().endswith('.png')]) 
    print(f"  Final minority count: {final_count_ds} (Majority: {n_major})") 
    return True

# --- 5. Main Execution (Modified for Single Split) ---
if __name__ == '__main__':
    # --- Configuration ---
    data_directory = r'D:\Usuario\Desktop\Base_de_dados\TEST_ADJUSTED'
    # Output base directory for the single split
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\FINAL_SPLIT_DATA'
    TRAIN_SIZE = 0.8 # 80% for training
    RANDOM_STATE = 42
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)

    # --- Prepare Output Directory ---
    output_train_dir = os.path.join(output_base_dir, "TRAIN")
    output_val_dir = os.path.join(output_base_dir, "VALIDATION")

    if os.path.exists(output_base_dir):
        print(f"Output directory '{output_base_dir}' already exists.")
        user_input = input("  -> Delete existing directory and proceed? (yes/no): ").strip().lower()
        if user_input == 'yes':
            try: 
                shutil.rmtree(output_base_dir) 
                print("  Deleted.")
            except Exception as e: 
                print(f"Err delete: {e}. Exit.") 
                exit(1)
        else: 
            print("Exiting.") 
            exit(0)
    try:
        # Create base, train, and validation dirs (including label subdirs)
        for split_dir in [output_train_dir, output_val_dir]:
             os.makedirs(os.path.join(split_dir, 'CANCER'), exist_ok=True)
             os.makedirs(os.path.join(split_dir, 'CANCER_MASK'), exist_ok=True)
             os.makedirs(os.path.join(split_dir, 'NOT_CANCER'), exist_ok=True)
             os.makedirs(os.path.join(split_dir, 'NOT_CANCER_MASK'), exist_ok=True)
        print(f"Created output structure in {output_base_dir}")
    except OSError as e: 
        print(f"Err creating output dirs: {e}") 
        exit(1)


    # --- Load Data ---
    try: 
        data_df = load_data(data_directory)
    except Exception as e: 
        print(f"Load Data Err: {e}") 
        exit(1)


    # --- Create Single Train/Validation Split ---
    try:
        final_train_df, final_val_df = create_train_val_split(
            data_df, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"Error during Train/Val split: {e}") 
        exit(1)


    # --- Process Splits (Copy Files) ---
    overall_success = True
    try:
        # Copy Training Files
        copy_files_for_split(final_train_df, output_train_dir)
        # Verify Training Files (Before Augmentation)
        verify_split_integrity(output_train_dir, final_train_df)

        # Copy Validation Files
        copy_files_for_split(final_val_df, output_val_dir)
        # Verify Validation Files
        verify_split_integrity(output_val_dir, final_val_df)

    except Exception as e:
        print(f"--- ERROR during file copying or initial verification: {e} ---")
        import traceback
        traceback.print_exc()
        overall_success = False


    # --- Augment and Balance TRAINING Set ---
    if overall_success:
        print(f"\n--- Augmenting and balancing FINAL training set ---")
        try:
            balance_success = augment_and_balance_train_set(output_train_dir, num_workers=NUM_WORKERS)
            if not balance_success:
                 print(f"  Warning: Augmentation/Balancing function reported issues.")
                 # Decide if this is fatal
                 # overall_success = False
        except Exception as e:
            print(f"  FATAL ERROR during augmentation/balancing: {e}")
            import traceback
            traceback.print_exc()
            overall_success = False

    # --- Final Verification of TRAIN set (After Augmentation) ---
    if overall_success:
        print(f"\n--- Final Verification of Balanced TRAIN Set ---")
        try:
            verify_train_balance_and_masks(output_train_dir)
        except Exception as e:
            print(f"  ERROR during final TRAIN set verification: {e}")
            import traceback
            traceback.print_exc()
            overall_success = False


    # --- Final Summary ---
    print("\n--- Overall Process Summary ---")
    if overall_success:
        print("Final Train/Validation split created, copied, augmented, balanced, and verified successfully!")
        print(f"Output location: {output_base_dir}")
    else:
        print("Processing completed, but ERRORS occurred.")
        print("Please review the output messages and logs above.")
        print(f"Output directory (may contain partial or erroneous data): {output_base_dir}")

    print("--- Script Finished ---")