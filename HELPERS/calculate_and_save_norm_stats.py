import numpy as np
import cv2
import os
from PIL import Image, UnidentifiedImageError
import logging
import sys

# --- Configuration ---
# *** MUST Point this to your EXISTING template directory ***
TEMPLATE_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED\Slice_Template"

# *** Define WHERE to save the calculated stats ***
OUTPUT_STATS_DIR = r"D:\Usuario\Desktop\Master_Coding\Master_Coding\Slice_Template" # Or any other desired location
STATS_FILENAME = "normalization_stats.npz" # Standard name for the stats file

# --- Configure Logging (Optional but Recommended) ---
LOG_FILE = os.path.join(OUTPUT_STATS_DIR, "calculate_stats_log.txt")
os.makedirs(OUTPUT_STATS_DIR, exist_ok=True) # Ensure output dir exists
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO, # Log info level for this script
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
# Also print info messages to console
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_and_save_stats(template_dir, output_stats_path):
    """
    Calculates the average LAB mean and std dev from images in template_dir
    and saves them to output_stats_path.
    """
    all_means = []
    all_stds = []

    if not os.path.isdir(template_dir):
        logging.error(f"Template directory not found: {template_dir}")
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    template_files = [f for f in os.listdir(template_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not template_files:
        logging.error(f"No valid template image files found in: {template_dir}")
        raise ValueError(f"No valid template image files found in: {template_dir}")

    logging.info(f"Calculating average stats from {len(template_files)} templates in {template_dir}...")
    processed_count = 0
    skipped_count = 0

    for fname in template_files:
        template_path = os.path.join(template_dir, fname)
        try:
            # Use OpenCV directly as PIL was mainly for robustness check before
            template_img_bgr = cv2.imread(template_path)
            if template_img_bgr is None:
                logging.warning(f"Skipping unreadable template: {fname}")
                skipped_count += 1
                continue

            # Convert to LAB
            template_img_lab = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2LAB)
            if template_img_lab is None: # Check conversion result
                logging.warning(f"Skipping template due to LAB conversion issue: {fname}")
                skipped_count += 1
                continue

            # Calculate stats
            mean, std = cv2.meanStdDev(template_img_lab)
            all_means.append(np.hstack(mean)) # Flatten channels (L, A, B)
            all_stds.append(np.hstack(std))
            processed_count += 1

        except Exception as e:
            logging.warning(f"Error processing template {fname}: {e}")
            skipped_count += 1
            continue # Skip this template on error

    if processed_count == 0:
         message = "Could not process any valid template images for averaging."
         logging.error(message)
         raise ValueError(message)

    logging.info(f"Successfully processed {processed_count} templates, skipped {skipped_count}.")

    # Calculate average stats
    avg_mean = np.mean(np.array(all_means), axis=0)
    avg_std = np.mean(np.array(all_stds), axis=0)

    # Prevent division by zero in normalization later
    epsilon = 1e-6
    avg_std = np.maximum(avg_std, epsilon)

    logging.info(f"Average Mean (LAB): {avg_mean}")
    logging.info(f"Average Std Dev (LAB): {avg_std}")

    # Save the calculated statistics
    try:
        np.savez(output_stats_path, template_mean=avg_mean, template_std=avg_std)
        logging.info(f"Successfully saved normalization statistics to: {output_stats_path}")
    except Exception as e:
        logging.error(f"Error saving normalization stats to {output_stats_path}: {e}")
        raise RuntimeError(f"Error saving normalization stats: {e}")

    return avg_mean, avg_std

# --- Main Execution Block ---
if __name__ == "__main__":
    output_file_path = os.path.join(OUTPUT_STATS_DIR, STATS_FILENAME)
    logging.info(f"--- Starting Normalization Stat Calculation ---")
    logging.info(f"Template Directory: {TEMPLATE_DIR}")
    logging.info(f"Output Stats File: {output_file_path}")

    try:
        calculate_and_save_stats(TEMPLATE_DIR, output_file_path)
        logging.info(f"--- Stat Calculation Finished Successfully ---")
    except Exception as e:
        logging.critical(f"--- Stat Calculation Failed: {e} ---")
        print(f"Error: {e}. Check log file '{LOG_FILE}' for details.", file=sys.stderr)
        exit(1)