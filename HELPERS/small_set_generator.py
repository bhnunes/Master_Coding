import os
import shutil
import pandas as pd
import re
from tqdm import tqdm

def create_small_dataset(base_dataset_dir, output_dir, sample_percentage=0.1):
    """
    Creates a smaller dataset, ensuring NO data leakage between original splits.

    Args:
        base_dataset_dir: Path to the original dataset.
        output_dir: Path to save the smaller dataset.
        sample_percentage: Percentage of data to sample (from each original split).
    """

    os.makedirs(output_dir, exist_ok=True)

    total_iterations = len(['TRAIN', 'VALIDATION', 'TEST'])
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar_overall:
        for split_name in ['TRAIN', 'VALIDATION', 'TEST']:
            split_dir = os.path.join(base_dataset_dir, split_name)
            output_split_dir = os.path.join(output_dir, split_name)
            os.makedirs(output_split_dir, exist_ok=True)

            # --- 1. Collect ALL patient IDs and filenames for THIS split ---
            all_data = []
            for label_name in ['CANCER', 'NOT_CANCER']:
                image_dir = os.path.join(split_dir, label_name)
                mask_dir = os.path.join(split_dir, f"{label_name}_MASK")

                if not os.path.isdir(image_dir):  # Check if directory exists
                    print(f"Warning: Directory not found: {image_dir}, skipping.")
                    continue

                image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
                for image_name in image_files:
                    match = re.search(r'PATIENT_(\d+)_', image_name)
                    patient_id = int(match.group(1)) if match else None
                    if patient_id is None:
                        print(f"Warning: Could not extract patient ID from {image_name}, skipping.")
                        continue
                    all_data.append({
                        'patient_id': patient_id,
                        'label_name': label_name,
                        'image_name': image_name,
                        'split_name': split_name  # Store the original split
                    })

            if not all_data:
                print(f"Warning: No data found for split {split_name}, skipping.")
                pbar_overall.update(1)
                continue

            df = pd.DataFrame(all_data)

            # --- 2. Sample PATIENTS (not images) ---
            unique_patients = df['patient_id'].unique()
            num_patients_to_sample = int(len(unique_patients) * sample_percentage)

            # Handle edge case:  Not enough patients to sample
            if num_patients_to_sample == 0:
                print(f"Warning:  Not enough patients in split {split_name} to sample. Skipping.")
                pbar_overall.update(1)
                continue

            sampled_patients = set(random.sample(list(unique_patients), num_patients_to_sample))

            # --- 3. Filter the DataFrame to include ONLY sampled patients ---
            sampled_df = df[df['patient_id'].isin(sampled_patients)]

            # --- 4. Copy files (now guaranteed to be leakage-free) ---
            for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc=f"Copying {split_name}"):
                label_name = row['label_name']
                image_name = row['image_name']
                mask_name = image_name

                image_path = os.path.join(base_dataset_dir, split_name, label_name, image_name)
                mask_path = os.path.join(base_dataset_dir, split_name, f"{label_name}_MASK", mask_name)
                output_image_path = os.path.join(output_dir, split_name, label_name, image_name)
                output_mask_path = os.path.join(output_dir, split_name, f"{label_name}_MASK", mask_name)

                # Create output directories if they don't exist
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

                try:
                    shutil.copy(image_path, output_image_path)
                    shutil.copy(mask_path, output_mask_path)
                except FileNotFoundError:
                    print(f"Warning: Image or mask file not found: {image_path} or {mask_path}")

            # --- 5. ASSERT NO DATA LEAKAGE ---
            # Collect patient IDs from the *newly created* small dataset's splits
            small_train_patients = set()
            small_val_patients = set()
            small_test_patients = set()

            for label_name in ['CANCER', 'NOT_CANCER']:
                for small_split_name, patient_set in zip(['TRAIN', 'VALIDATION', 'TEST'],
                                                         [small_train_patients, small_val_patients, small_test_patients]):
                    small_image_dir = os.path.join(output_dir, small_split_name, label_name)
                    if os.path.isdir(small_image_dir):  # Check if dir exists
                        for small_image_name in os.listdir(small_image_dir):
                            match = re.search(r'PATIENT_(\d+)_', small_image_name)
                            if match:
                                patient_id = int(match.group(1))
                                patient_set.add(patient_id)

            assert len(small_train_patients.intersection(small_val_patients)) == 0, "Data leakage (train/val) in small dataset!"
            assert len(small_train_patients.intersection(small_test_patients)) == 0, "Data leakage (train/test) in small dataset!"
            assert len(small_val_patients.intersection(small_test_patients)) == 0, "Data leakage (val/test) in small dataset!"

            pbar_overall.update(1)
            print(f'Small set for {split_name} created and verified.')

import random  # Make sure random is imported at the top

if __name__ == '__main__':
    base_dataset_path = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits\dataset_2'
    small_dataset_output_path = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits\small_dataset_2'
    create_small_dataset(base_dataset_path, small_dataset_output_path)
    print("Small dataset created successfully.")