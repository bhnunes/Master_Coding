import numpy as np
import cv2
import os
import random
from PIL import Image, UnidentifiedImageError
from joblib import Parallel, delayed
import logging
import re  # For parsing patient IDs
import shutil # For copying files
from collections import defaultdict # For grouping files by patient
import sys # For printing to stderr


# --- Configuration ---
# Base directories (Adjust as needed)
INPUT_BASE_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER"
OUTPUT_BASE_DIR = r"D:\Usuario\Desktop\Base_de_dados\ABLATION\ADJUSTED_RANDOM"

# Subdirectories for data and templates
CANCER_INPUT_DIR = os.path.join(INPUT_BASE_DIR, "CANCER")
NOT_CANCER_INPUT_DIR = os.path.join(INPUT_BASE_DIR, "NOT_CANCER")
TEMPLATE_DIR = os.path.join(OUTPUT_BASE_DIR, "Slice_Template") # Template dir inside output base

# Log file location
LOG_FILE = os.path.join(OUTPUT_BASE_DIR, "normalization_error_log.txt")

# --- Configure logging ---
# Ensure the output base directory exists for the log file
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w", # Overwrite log file each run
)

# --- Functions ---

def create_template_directory(source_dirs, template_dir):
    """
    Creates the template directory by selecting one random image per patient
    from the combined source directories.

    Args:
        source_dirs (list): List of paths to source directories (e.g., [CANCER_DIR, NOT_CANCER_DIR]).
        template_dir (str): Path to the target template directory.
    """
    print(f"Creating template directory: {template_dir}")
    logging.info(f"Starting template directory creation. Target: {template_dir}")

    patient_files = defaultdict(list)
    patient_id_pattern = re.compile(r"PATIENT_(\d+)_") # Regex to find PATIENT_ID_

    # 1. Gather all image files grouped by patient ID
    print("Scanning source directories for patient images...")
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found: {source_dir}. Skipping.", file=sys.stderr)
            logging.warning(f"Source directory not found during template creation: {source_dir}")
            continue

        print(f"  Scanning: {source_dir}")
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                match = patient_id_pattern.search(filename)
                if match:
                    patient_id = match.group(1)
                    full_path = os.path.join(source_dir, filename)
                    patient_files[patient_id].append(full_path)
                # else: Optional: log files that don't match the pattern

    if not patient_files:
        message = "No patient images found matching the pattern in source directories. Cannot create templates."
        print(f"Error: {message}", file=sys.stderr)
        logging.error(message)
        raise ValueError(message)

    print(f"Found images for {len(patient_files)} unique patients.")

    # 2. Clear existing template directory (optional, but recommended for consistency)
    if os.path.exists(template_dir):
        print(f"Clearing existing template directory: {template_dir}")
        try:
            shutil.rmtree(template_dir)
        except OSError as e:
            message = f"Error clearing existing template directory: {e}. Check permissions or if files are in use."
            print(f"Error: {message}", file=sys.stderr)
            logging.error(message)
            raise OSError(message)

    # 3. Create the template directory
    try:
        os.makedirs(template_dir)
    except OSError as e:
         message = f"Error creating template directory: {e}"
         print(f"Error: {message}", file=sys.stderr)
         logging.error(message)
         raise OSError(message)


    # 4. Select one random image per patient and copy to template directory
    print("Selecting and copying one random image per patient...")
    templates_copied_count = 0
    for patient_id, file_list in patient_files.items():
        if file_list:
            selected_file = random.choice(file_list)
            try:
                # Copy file, preserving metadata (like modification time)
                shutil.copy2(selected_file, template_dir)
                templates_copied_count += 1
            except Exception as e:
                print(f"Warning: Could not copy file {selected_file} for patient {patient_id}: {e}", file=sys.stderr)
                logging.warning(f"Could not copy template file {selected_file} for patient {patient_id}: {e}")

    if templates_copied_count == 0:
         message = "Failed to copy any template files. Please check permissions and file paths."
         print(f"Error: {message}", file=sys.stderr)
         logging.error(message)
         raise RuntimeError(message)

    print(f"Successfully copied {templates_copied_count} template images (one per patient) to {template_dir}")
    logging.info(f"Successfully copied {templates_copied_count} template images.")


def calculate_average_template_stats(template_dir):
    """Calculates the average LAB mean and std dev across all valid templates."""
    all_means = []
    all_stds = []
    template_files = [f for f in os.listdir(template_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not template_files:
        raise ValueError("No template images found in the directory for averaging.")

    print(f"Calculating average stats from {len(template_files)} templates in {template_dir}...")
    logging.info(f"Calculating average stats from {len(template_files)} templates.")
    processed_count = 0
    for fname in template_files:
        template_path = os.path.join(template_dir, fname)
        try:
            # Use PIL to open first to catch non-image files reliably
            with Image.open(template_path) as img_pil:
                # Convert PIL Image (often RGB or RGBA) to NumPy array, then to BGR for OpenCV
                template_img_np = np.array(img_pil)
                if template_img_np.shape[2] == 4: # Handle RGBA
                     template_img_np = cv2.cvtColor(template_img_np, cv2.COLOR_RGBA2BGR)
                elif template_img_np.shape[2] == 3: # Handle RGB
                     template_img_np = cv2.cvtColor(template_img_np, cv2.COLOR_RGB2BGR)
                else: # Handle Grayscale or other formats if necessary
                    # Decide how to handle: convert to BGR or skip? Skipping is safer.
                    print(f"Warning: Skipping template {fname} with unsupported channel count: {template_img_np.shape}", file=sys.stderr)
                    logging.warning(f"Skipping template {fname} with unsupported channel count: {template_img_np.shape}")
                    continue

            # Now template_img_np is definitely BGR
            template_img_lab = cv2.cvtColor(template_img_np, cv2.COLOR_BGR2LAB)
            mean, std = cv2.meanStdDev(template_img_lab)
            all_means.append(np.hstack(mean))
            all_stds.append(np.hstack(std))
            processed_count += 1

        except UnidentifiedImageError:
            print(f"Warning: Skipping non-image or corrupt template file: {fname}", file=sys.stderr)
            logging.warning(f"Skipping non-image or corrupt template file during averaging: {template_path}")
        except Exception as e:
            print(f"Warning: Error processing template {fname}: {e}", file=sys.stderr)
            logging.warning(f"Error processing template {template_path} during averaging: {e}")
            continue # Skip this template on error

    if processed_count == 0 or not all_means:
         raise ValueError("Could not process any valid template images for averaging.")

    # Calculate average, ensuring stability for std dev near zero
    avg_mean = np.mean(np.array(all_means), axis=0)
    avg_std = np.mean(np.array(all_stds), axis=0)
    # Prevent division by zero - set near-zero std dev to a small epsilon
    epsilon = 1e-6
    avg_std = np.maximum(avg_std, epsilon)

    print(f"Average template stats calculated from {processed_count} valid templates.")
    logging.info(f"Average template stats calculated from {processed_count} valid templates.")

    stats_file = os.path.join(OUTPUT_BASE_DIR, 'normalization_stats.npz') # Define path
    try:
        np.savez(stats_file, template_mean=avg_mean, template_std=avg_std)
        print(f"Saved normalization statistics to: {stats_file}")
        logging.info(f"Saved normalization statistics to: {stats_file}")
    except Exception as e:
        print(f"Error saving normalization stats: {e}", file=sys.stderr)
        logging.error(f"Error saving normalization stats: {e}")
    # --- End Modification ---

    return avg_mean, avg_std


def process_image(img_filename, input_dir, output_dir, template_mean, template_std):
    """Applies Reinhard normalization to a single image."""
    input_path = os.path.join(input_dir, img_filename)
    # Create output filename (e.g., CANCER_PATIENT_...)
    output_filename = f"{img_filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Open with PIL for robust loading and format handling
        with Image.open(input_path) as input_pil:
            # Convert PIL (RGB/RGBA etc) to NumPy array -> OpenCV BGR
             input_np = np.array(input_pil)
             if input_np.ndim == 2: # Grayscale
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_GRAY2BGR)
             elif input_np.shape[2] == 3: # RGB
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
             elif input_np.shape[2] == 4: # RGBA
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGBA2BGR)
             else:
                 raise ValueError(f"Unsupported channel count: {input_np.shape}")

        # Check if conversion to numpy array was successful
        if input_bgr.size == 0:
            print(f"Warning: Skipping empty image after conversion: {img_filename}", file=sys.stderr)
            logging.error(f"Skipping empty image after conversion: {input_path}")
            return

        # Convert to LAB color space
        input_lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)

        # Calculate source image statistics
        img_mean, img_std = cv2.meanStdDev(input_lab)
        img_mean = np.hstack(img_mean) # Flatten means
        img_std = np.hstack(img_std)   # Flatten stds
        # Prevent division by zero for std dev
        epsilon = 1e-6
        img_std = np.maximum(img_std, epsilon)

        # --- Reinhard color normalization (vectorized) ---
        # Ensure LAB image is float for calculation
        input_lab = input_lab.astype(np.float32)
        # Apply formula
        normalized_lab = ((input_lab - img_mean) * (template_std / img_std)) + template_mean
        # Clip values to valid LAB range (approx 0-255) and convert back to uint8
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        # --- End Normalization ---

        # Convert back to BGR
        output_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        # Save the normalized image using OpenCV
        success = cv2.imwrite(output_path, output_bgr)
        if not success:
             raise IOError(f"Failed to write image to {output_path}")

    except UnidentifiedImageError:
        print(f"Warning: Skipping non-image file or corrupted image: {img_filename}", file=sys.stderr)
        logging.error(f"Skipping non-image file or corrupted image: {input_path}")
    except FileNotFoundError:
         print(f"Error: Input image not found: {input_path}", file=sys.stderr)
         logging.error(f"Input image not found: {input_path}")
    except Exception as e:
        print(f"Error processing image {img_filename}: {e}", file=sys.stderr)
        logging.exception(f"Error processing image: {input_path} - Type: {type(e).__name__}, Error: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Define Input and Output Directories ---
    source_directories = [CANCER_INPUT_DIR, NOT_CANCER_INPUT_DIR]
    # Automatically create corresponding output directories
    output_directories = [
        os.path.join(OUTPUT_BASE_DIR, os.path.basename(src)) for src in source_directories
    ]

    # --- Step 1: Create Template Directory ---
    try:
        create_template_directory(source_directories, TEMPLATE_DIR)
    except (ValueError, OSError, RuntimeError) as e:
         print(f"Fatal Error during template creation: {e}. Exiting.", file=sys.stderr)
         logging.critical(f"Fatal Error during template creation: {e}. Exiting.")
         exit(1) # Exit if template creation fails

    # --- Step 2: Calculate Average Template Stats ---
    try:
        template_mean_avg, template_std_avg = calculate_average_template_stats(TEMPLATE_DIR)
    except ValueError as ve:
        print(f"Fatal Error calculating average template stats: {ve}. Exiting.", file=sys.stderr)
        logging.critical(f"Fatal Error calculating average template stats: {ve}. Exiting.")
        exit(1) # Exit if stats calculation fails
    except Exception as e:
        print(f"Fatal Unexpected error calculating average template stats: {e}. Exiting.", file=sys.stderr)
        logging.critical(f"Fatal Unexpected error calculating average template stats: {e}. Exiting.", exc_info=True)
        exit(1)

    # --- Step 3: Process Input Directories Sequentially ---
    print("\n--- Starting Image Normalization ---")
    for input_dir, output_dir in zip(source_directories, output_directories):

        print(f"\nProcessing Directory: {input_dir}")
        logging.info(f"Starting processing for directory: {input_dir}")

        if not os.path.isdir(input_dir):
            print(f"Warning: Input directory '{input_dir}' not found. Skipping.", file=sys.stderr)
            logging.warning(f"Input directory '{input_dir}' not found. Skipping.")
            continue

        # Create the corresponding output directory if it doesn't exist
        if not os.path.isdir(output_dir):
             try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
             except OSError as e:
                 print(f"Error: Could not create output directory {output_dir}: {e}. Skipping this directory.", file=sys.stderr)
                 logging.error(f"Could not create output directory {output_dir}: {e}. Skipping.")
                 continue


        input_image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not input_image_list:
            print("  No images found to process in this directory.")
            logging.warning(f"No images found to process in directory: {input_dir}")
            continue

        print(f"  Found {len(input_image_list)} images to normalize.")

        # Process images in batches for the current directory
        batch_size = 500 # Adjust batch size as needed based on memory/performance
        total_batches = (len(input_image_list) + batch_size - 1) // batch_size

        for i in range(0, len(input_image_list), batch_size):
            batch_num = (i // batch_size) + 1
            image_batch = input_image_list[i:i + batch_size]

            print(f"  Processing Batch {batch_num} of {total_batches} ({len(image_batch)} images)...")

            # Use the SAME average stats for all batches and all directories
            Parallel(n_jobs=-1)(
                delayed(process_image)(img_file, input_dir, output_dir, template_mean_avg, template_std_avg) for img_file in image_batch
            )

        print(f"Finished processing directory: {input_dir}")
        logging.info(f"Finished processing directory: {input_dir}")

    print(f"\n--- Normalization Complete ---")
    print(f"Check '{LOG_FILE}' for any errors logged during the process.")
    logging.info("Normalization process finished.")

