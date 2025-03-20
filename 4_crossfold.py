import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from PIL import Image
import shutil
import re
import concurrent.futures
from functools import partial

# --- 1. Data Loading and Preparation (Modified) ---

def load_data(data_dir):
    """Loads image and mask data, parses filenames, and returns a DataFrame."""
    data = []
    for label_name in ["CANCER", "NOT_CANCER"]:  # Iterate directly over label names
        image_dir = os.path.join(data_dir, label_name)
        mask_dir = os.path.join(data_dir, f"{label_name}_MASK")

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            print(f"Warning: Image or mask directory not found for {label_name}, skipping.")
            continue

        label = 1 if label_name == "CANCER" else 0

        for image_name in os.listdir(image_dir):
            # --- CRITICAL: Check for corresponding mask ---
            mask_name = image_name  # Mask name is the *same* as image name
            mask_name=mask_name.replace("adj_","")
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {image_name}, skipping.")
                continue

            match = re.search(r'PATIENT_(\d+)_', image_name)
            if match:
                patient_id = int(match.group(1))
            else:
                print(f"Warning: Could not extract patient ID from {image_name}, skipping.")
                continue

            data.append({
                'patient_id': patient_id,
                'image_path': image_path,
                'mask_path': mask_path,  # Store mask path
                'label': label
            })

    return pd.DataFrame(data)

# --- 2. Cross-Validation Splitting (Same as before, with data leakage check) ---

def create_cross_val_splits(df, n_splits=2, random_state=42):
    """Creates Group K-Fold cross-validation splits, handling stratification."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for _, (train_val_idx, test_idx) in enumerate(sgkf.split(df, df['label'], df['patient_id'])):
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        group_kfold_inner = GroupKFold(n_splits=5)
        groups_inner = train_val_df['patient_id'].to_numpy()
        y_inner = train_val_df['label'].to_numpy()
        train_val_df = train_val_df.reset_index()

        inner_split = list(group_kfold_inner.split(train_val_df, y_inner, groups_inner))
        train_idx, val_idx = inner_split[0]  # Use only the first inner split

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        splits.append({
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        })

    # Data Leakage Check
    for fold, split_data in enumerate(splits):
        train_patients = set(split_data['train_df']['patient_id'])
        val_patients = set(split_data['val_df']['patient_id'])
        test_patients = set(split_data['test_df']['patient_id'])

        assert len(train_patients.intersection(val_patients)) == 0, f"Data leakage in fold {fold} (train/val)"
        assert len(train_patients.intersection(test_patients)) == 0, f"Data leakage in fold {fold} (train/test)"
        assert len(val_patients.intersection(test_patients)) == 0, f"Data leakage in fold {fold} (val/test)"
    return splits

# --- 3. Optimized Augmentation Function (Modified) ---

def augment_single_image_and_mask(image_path, mask_path, output_image_dir, output_mask_dir):
    """Augments a single image and its corresponding mask, saving all versions."""
    try:
        img = Image.open(image_path)
        mask = Image.open(mask_path)
    except (FileNotFoundError, IOError):
        print(f"Skipping image/mask (cannot open): {image_path}, {mask_path}")
        return

    base_filename = os.path.basename(image_path).split('.')[0]

    augmentations = {
        'original': (img, mask),
        'rotated_90': (img.rotate(90, expand=True), mask.rotate(90, expand=True)),
        'rotated_180': (img.rotate(180, expand=True), mask.rotate(180, expand=True)),
        'rotated_270': (img.rotate(270, expand=True), mask.rotate(270, expand=True)),
        'flipped_horizontal': (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)),
        'flipped_vertical': (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)),
    }

    for aug_name, (augmented_img, augmented_mask) in augmentations.items():
        output_image_filename = f"{base_filename}_{aug_name}.png"
        output_mask_filename = f"{base_filename}_{aug_name}.png"  # Same name for mask
        augmented_img.save(os.path.join(output_image_dir, output_image_filename))
        augmented_mask.save(os.path.join(output_mask_dir, output_mask_filename))

def augment_images_and_masks_parallel(image_paths, mask_paths, output_image_dir, output_mask_dir, num_workers=None):
    """Applies augmentations in parallel using ProcessPoolExecutor."""
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_func = partial(augment_single_image_and_mask,
                               output_image_dir=output_image_dir,
                               output_mask_dir=output_mask_dir)
        executor.map(partial_func, image_paths, mask_paths)  # Pass both image and mask paths

# --- 4. Main Execution (Modified) ---
if __name__ == '__main__':
    data_directory = r'D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED'  # Use raw string or double backslashes
    output_base_dir = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits'

    data_df = load_data(data_directory)
    cross_val_splits = create_cross_val_splits(data_df)

    for fold, split_data in enumerate(cross_val_splits):
        print(f"Processing Fold {fold + 1}...")
        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']

        fold_output_dir = os.path.join(output_base_dir, f"dataset_{fold + 1}")  # Use "dataset_" prefix
        os.makedirs(fold_output_dir, exist_ok=True)

        for split_name, split_df in zip(['TRAIN', 'VALIDATION', 'TEST'], [train_df, val_df, test_df]):  # Uppercase
            split_dir = os.path.join(fold_output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for label in [0, 1]:
                label_name = 'CANCER' if label == 1 else 'NOT_CANCER'  # Uppercase
                image_dest_dir = os.path.join(split_dir, label_name)
                mask_dest_dir = os.path.join(split_dir, f"{label_name}_MASK")  # Create mask directory
                os.makedirs(image_dest_dir, exist_ok=True)
                os.makedirs(mask_dest_dir, exist_ok=True)  # Create mask directory

            for _, row in split_df.iterrows():
                image_path = row['image_path']
                mask_path = row['mask_path']  # Get mask path
                label = row['label']
                label_name = 'CANCER' if label == 1 else 'NOT_CANCER'

                image_dest_dir = os.path.join(fold_output_dir, split_name, label_name)
                mask_dest_dir = os.path.join(fold_output_dir, split_name, f"{label_name}_MASK") # Get mask folder
                image_dest_path = os.path.join(image_dest_dir, os.path.basename(image_path))
                mask_dest_path = os.path.join(mask_dest_dir, os.path.basename(mask_path))  # Mask destination

                try:
                    shutil.copy(image_path, image_dest_path)
                    shutil.copy(mask_path, mask_dest_path)  # Copy mask
                except FileNotFoundError:
                    print(f"Warning: Source image/mask not found: {image_path} or {mask_path}")
                    continue

        # --- Parallel Augmentation (for the training set, including masks) ---
        train_cancer_df = train_df[train_df['label'] == 1]
        train_no_cancer_df = train_df[train_df['label'] == 0]

        train_cancer_image_paths = train_cancer_df['image_path'].tolist()
        train_cancer_mask_paths = train_cancer_df['mask_path'].tolist()  # Get mask paths

        train_no_cancer_image_paths = train_no_cancer_df['image_path'].tolist()
        train_no_cancer_mask_paths = train_no_cancer_df['mask_path'].tolist() # Get mask paths

        train_cancer_output_image_dir = os.path.join(fold_output_dir, "TRAIN", "CANCER")
        train_cancer_output_mask_dir = os.path.join(fold_output_dir, "TRAIN", "CANCER_MASK")  # Output mask dir

        train_no_cancer_output_image_dir = os.path.join(fold_output_dir, "TRAIN", "NOT_CANCER")
        train_no_cancer_output_mask_dir = os.path.join(fold_output_dir, "TRAIN", "NOT_CANCER_MASK")  # Output mask dir

        num_workers = os.cpu_count()
        print(f"Augmenting cancer images and masks with {num_workers} workers...")
        augment_images_and_masks_parallel(train_cancer_image_paths, train_cancer_mask_paths,
                                          train_cancer_output_image_dir, train_cancer_output_mask_dir,
                                          num_workers=num_workers)
        print(f"Augmenting non-cancer images and masks with {num_workers} workers...")
        augment_images_and_masks_parallel(train_no_cancer_image_paths, train_no_cancer_mask_paths,
                                         train_no_cancer_output_image_dir, train_no_cancer_output_mask_dir,
                                         num_workers=num_workers)
    print("Cross-validation splits created and augmented (in parallel).")