import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from PIL import Image
import shutil
import re
from joblib import Parallel, delayed  # For parallel processing
import concurrent.futures  # Another option for parallel processing (using ProcessPoolExecutor)
from functools import partial  # For passing arguments to parallelized functions


# --- 1. Data Loading and Preparation (Same as before) ---

def load_data(data_dir):
    """Loads image data, parses filenames, and returns a DataFrame."""
    data = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        label = 1 if label_name == "CANCER" else 0

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)

            match = re.search(r'PATIENT_(\d+)_', image_name)
            if match:
                patient_id = int(match.group(1))
            else:
                print(f"Warning: Could not extract patient ID from {image_name}, skipping.")
                continue

            data.append({
                'patient_id': patient_id,
                'image_path': image_path,
                'label': label
            })

    return pd.DataFrame(data)


# --- 2. Cross-Validation Splitting (Same as before) ---

def create_cross_val_splits(df, n_splits=10, random_state=42):
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
        train_idx, val_idx = inner_split[0]

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        splits.append({
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        })
    return splits


# --- 3. Optimized Augmentation Function (Parallel Processing) ---

def augment_single_image(image_path, output_dir):
    """Augments a single image and saves all versions."""
    try:
        img = Image.open(image_path)
    except (FileNotFoundError, IOError):
        print(f"Skipping image (cannot open): {image_path}")
        return  # Return early if the image can't be opened

    base_filename = os.path.basename(image_path).split('.')[0]

    augmentations = {
        'original': img,
        'rotated_90': img.rotate(90, expand=True),
        'rotated_180': img.rotate(180, expand=True),
        'rotated_270': img.rotate(270, expand=True),
        'flipped_horizontal': img.transpose(Image.FLIP_LEFT_RIGHT),
        'flipped_vertical': img.transpose(Image.FLIP_TOP_BOTTOM),
    }

    for aug_name, augmented_img in augmentations.items():
        output_filename = f"{base_filename}_{aug_name}.png"
        augmented_img.save(os.path.join(output_dir, output_filename))



def augment_images_parallel(image_paths, output_dir, num_workers=None):
    """
    Applies augmentations in parallel using ProcessPoolExecutor.

    Args:
        image_paths: List of image paths.
        output_dir: Output directory.
        num_workers: Number of worker processes.  If None, uses the number of CPU cores.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ProcessPoolExecutor for true parallelism (handles GIL limitations of Python)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use functools.partial to pass the output_dir to the augment_single_image function
        partial_func = partial(augment_single_image, output_dir=output_dir)
        executor.map(partial_func, image_paths)


# --- 4. Main Execution (Parallel Augmentation) ---

if __name__ == '__main__':
    data_directory = 'path/to/your/data'  # REPLACE
    output_base_dir = 'path/to/output/cross_val_splits'  # REPLACE

    data_df = load_data(data_directory)
    cross_val_splits = create_cross_val_splits(data_df)

    for fold, split_data in enumerate(cross_val_splits):
        print(f"Processing Fold {fold + 1}...")
        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']

        fold_output_dir = os.path.join(output_base_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            split_dir = os.path.join(fold_output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for label in [0, 1]:
                label_name = 'cancer' if label == 1 else 'no_cancer'
                label_dir = os.path.join(split_dir, label_name)
                os.makedirs(label_dir, exist_ok=True)

            for _, row in split_df.iterrows():
                image_path = row['image_path']
                label = row['label']
                label_name = 'cancer' if label == 1 else 'no_cancer'

                dest_dir = os.path.join(fold_output_dir, split_name, label_name)
                dest_path = os.path.join(dest_dir, os.path.basename(image_path))
                try:
                    shutil.copy(image_path, dest_path)
                except FileNotFoundError:
                    print(f"Warning: Source image not found: {image_path}")
                    continue

        # Parallel Augmentation (for the training set)
        train_cancer_paths = train_df[train_df['label'] == 1]['image_path'].tolist()
        train_no_cancer_paths = train_df[train_df['label'] == 0]['image_path'].tolist()

        train_cancer_output_dir = os.path.join(fold_output_dir, "train", "cancer")
        train_no_cancer_output_dir = os.path.join(fold_output_dir, "train", "no_cancer")
        
        # Use a reasonable number of workers.  os.cpu_count() is a good default.
        num_workers = os.cpu_count()
        print(f"Augmenting cancer images with {num_workers} workers...")
        augment_images_parallel(train_cancer_paths, train_cancer_output_dir, num_workers=num_workers)
        print(f"Augmenting non-cancer images with {num_workers} workers...")
        augment_images_parallel(train_no_cancer_paths, train_no_cancer_output_dir, num_workers=num_workers)
        

    print("Cross-validation splits created and augmented (in parallel).")