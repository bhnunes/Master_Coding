import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm  # Import tqdm

def create_small_dataset(base_dataset_dir, output_dir, sample_percentage=0.1):
    """
    Creates a smaller dataset by sampling a percentage of the original dataset.

    Args:
        base_dataset_dir: Path to the original dataset (e.g., 'dataset_1').
        output_dir: Path to save the smaller dataset.
        sample_percentage: Percentage of data to sample (e.g., 0.1 for 10%).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Total number of iterations for the outer loops (for progress bar)
    total_iterations = len(['TRAIN', 'VALIDATION', 'TEST']) * len(['CANCER', 'NOT_CANCER'])

    with tqdm(total=total_iterations, desc="Overall Progress") as pbar_overall:  # Outer progress bar
        for split_name in ['TRAIN', 'VALIDATION', 'TEST']:
            split_dir = os.path.join(base_dataset_dir, split_name)
            output_split_dir = os.path.join(output_dir, split_name)
            os.makedirs(output_split_dir, exist_ok=True)

            for label_name in ['CANCER', 'NOT_CANCER']:
                image_dir = os.path.join(split_dir, label_name)
                mask_dir = os.path.join(split_dir, f"{label_name}_MASK")
                output_image_dir = os.path.join(output_split_dir, label_name)
                output_mask_dir = os.path.join(output_split_dir, f"{label_name}_MASK")
                os.makedirs(output_image_dir, exist_ok=True)
                os.makedirs(output_mask_dir, exist_ok=True)

                # Get list of images (and implicitly, masks)
                image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

                # --- Stratified Sampling ---
                if not image_files:  # Handle empty directories
                    print(f"Warning: No images found in {image_dir}, skipping.")
                    pbar_overall.update(1)  # Update outer progress bar even if skipping
                    continue

                # Create a dataframe for stratified sampling
                data = []
                for image_name in image_files:
                    match = re.search(r'PATIENT_(\d+)_', image_name)
                    patient_id = int(match.group(1)) if match else None
                    if patient_id is None:
                        print(f"Warning: Could not extract patient ID from {image_name}, this image won't be sampled.")
                        continue
                    data.append({'image_name': image_name, 'patient_id': patient_id})

                df = pd.DataFrame(data)

                # Split data, stratifying by patient ID.
                _, sample_df = train_test_split(
                    df,
                    test_size=sample_percentage,
                    random_state=42,
                    stratify=df['patient_id']
                )
                # --- Copy Sampled Images and Masks ---

                # Inner progress bar for copying files
                with tqdm(total=len(sample_df), desc=f"Copying {label_name} ({split_name})", leave=False) as pbar_inner:
                    for _, row in sample_df.iterrows():
                        image_name = row['image_name']
                        mask_name = image_name  # Mask names are the *same* as image names
                        image_path = os.path.join(image_dir, image_name)
                        mask_path = os.path.join(mask_dir, mask_name)
                        output_image_path = os.path.join(output_image_dir, image_name)
                        output_mask_path = os.path.join(output_mask_dir, mask_name)

                        try:
                            shutil.copy(image_path, output_image_path)
                            shutil.copy(mask_path, output_mask_path)
                        except FileNotFoundError:
                            print(f"Warning: Image or mask file not found: {image_path} or {mask_path}")
                        pbar_inner.update(1)  # Update inner progress bar

                pbar_overall.update(1)  # Update outer progress bar after each label/split
                print(f'Small set for {label_name} of {split_name} created.')

if __name__ == '__main__':
    base_dataset_path = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits\dataset_1'
    small_dataset_output_path = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits\small_dataset'
    create_small_dataset(base_dataset_path, small_dataset_output_path)
    print("Small dataset created successfully.")