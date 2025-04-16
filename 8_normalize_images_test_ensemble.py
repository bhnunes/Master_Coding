import cv2
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm

def normalize_single_image(img_filename, input_dir, output_dir, target_mean, target_std):
    """Applies Reinhard normalization to a single image using pre-calculated target stats."""
    input_path = os.path.join(input_dir, img_filename)
    output_path = os.path.join(output_dir, img_filename) # Save with original name or add prefix
    smooth = 1e-6 # Epsilon for std dev

    try:
        source_img_bgr = cv2.imread(input_path)
        if source_img_bgr is None:
            print(f"Warning: Could not read image {img_filename}. Skipping.")
            return

        source_img_lab = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate source image stats
        source_mean, source_std = cv2.meanStdDev(source_img_lab)
        source_mean = np.hstack(source_mean)
        source_std = np.hstack(source_std)
        source_std = np.maximum(source_std, smooth) # Avoid division by zero

        # Apply Reinhard normalization
        normalized_lab = ((source_img_lab - source_mean) * (target_std / source_std)) + target_mean
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

        # Convert back to BGR and save
        output_img_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, output_img_bgr)

    except Exception as e:
        print(f"Error processing image {img_filename}: {e}")


def normalize_new_data(input_image_dir, output_image_dir, stats_file_path):
    """Normalizes all images in input_dir using saved stats."""

    if not os.path.exists(stats_file_path):
        raise FileNotFoundError(f"Normalization stats file not found: {stats_file_path}")
    if not os.path.isdir(input_image_dir):
         raise FileNotFoundError(f"Input directory not found: {input_image_dir}")

    os.makedirs(output_image_dir, exist_ok=True)

    # Load the normalization statistics
    try:
        stats = np.load(stats_file_path)
        target_mean = stats['template_mean']
        target_std = stats['template_std']
        print("Loaded normalization statistics.")
        print(f"  Target Mean: {target_mean}")
        print(f"  Target Std:  {target_std}")
    except Exception as e:
        raise RuntimeError(f"Error loading normalization stats from {stats_file_path}: {e}")

    image_list = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_list:
        print("No images found in the input directory.")
        return

    print(f"Normalizing {len(image_list)} images from {input_image_dir}...")
    Parallel(n_jobs=-1)( # Use all available cores
        delayed(normalize_single_image)(img_file, input_image_dir, output_image_dir, target_mean, target_std)
        for img_file in tqdm(image_list, desc="Normalizing New Images")
    )
    print(f"Normalization complete. Output saved to: {output_image_dir}")


# --- Example Usage ---
hold_out_test_dir = r'D:\Usuario\Desktop\Base_de_dados\TEST\NOT_CANCER'
normalized_hold_out_test_dir = r'D:\Usuario\Desktop\Base_de_dados\TEST\NOT_CANCER_NORM'
stats_file = r'D:\Usuario\Desktop\Master_Coding\Master_Coding\Slice_Template\normalization_stats.npz' # The file saved previously

normalize_new_data(hold_out_test_dir, normalized_hold_out_test_dir, stats_file)