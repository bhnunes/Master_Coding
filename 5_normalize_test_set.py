# -----------------------------------------------------------------------------
# APPLY NORMALIZATION TO A STANDALONE TEST SET
# -----------------------------------------------------------------------------
# This script applies a pre-existing stain normalization to a hold-out test set.
#
# Scientific Methodology:
# 1.  It reads the `normalization_stats.json` file from a previously completed
#     training data preparation run.
# 2.  It "rehydrates" the exact same tiatoolbox normalizer object using the
#     saved parameters (stain matrix, target means, etc.). This is a loading
#     step, NOT a fitting step, thus preventing any data leakage.
# 3.  It processes all images in the specified input TEST directory, applying
#     the loaded normalizer's `transform` method.
# 4.  Mask files are copied directly without modification.
# 5.  The output is a new, self-contained, normalized TEST set directory,
#     ready for use in the inference script.
#
# Configuration is done in the `if __name__ == '__main__':` block.
# -----------------------------------------------------------------------------

import os
import shutil
import json
import logging
import sys
import concurrent.futures

from dotenv import load_dotenv

# --- OpenSlide Dependency Setup ---
load_dotenv()
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            from tiatoolbox.tools import stainnorm
    else:
        from tiatoolbox.tools import stainnorm
except (ImportError, FileNotFoundError) as e:
    # ... (same error handling as your previous script) ...
    sys.exit(1)

import cv2
import numpy as np
from tqdm import tqdm

# --- Logger Setup ---
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'test_set_normalization.log')
    # ... (same logging setup as your previous script) ...
    logging.info(f"Logging configured. Output will be saved to {log_file}")

# --- Core Normalization Logic ---
def load_normalizer_from_stats(stats_dir):
    """
    Loads and re-instantiates a tiatoolbox normalizer from a saved stats file.
    """
    stats_file = os.path.join(stats_dir, 'normalization_stats.json')
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"normalization_stats.json not found in '{stats_dir}'")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    method_name = stats.get("method")
    if not method_name or method_name == "NOT_NORMALIZED":
        logging.info("Normalization method is 'NOT_NORMALIZED'. No normalizer will be applied.")
        return None, "NOT_NORMALIZED"

    logging.info(f"Rehydrating '{method_name}' normalizer from saved stats...")
    normalizer = stainnorm.get_normalizer(method_name)
    
    # Re-populate the normalizer with the saved parameters
    try:
        if isinstance(normalizer, stainnorm.ReinhardNormalizer):
            normalizer.target_means = np.array(stats["target_means"])
            normalizer.target_stds = np.array(stats["target_stds"])
        elif isinstance(normalizer, stainnorm.StainNormalizer): # Macenko, Vahadane, Ruifrok
            normalizer.stain_matrix_target = np.array(stats["stain_matrix_target"])
            normalizer.maxC_target = np.array(stats["maxC_target"])
        else:
            raise TypeError(f"Unsupported normalizer type found in stats file: {type(normalizer)}")
        
        logging.info(f"Successfully rehydrated '{method_name}' normalizer.")
        return normalizer, method_name
    except KeyError as e:
        raise KeyError(f"Stats file is missing a required key for '{method_name}': {e}")


# --- File Processing (Worker Function) ---
def process_file(src_path, dest_path, is_mask, normalizer):
    """Worker function: copies a mask or normalizes and writes an image."""
    try:
        # For masks or if no normalization is needed, just copy the file
        if is_mask or normalizer is None:
            shutil.copy(src_path, dest_path)
        else: # It's an image that needs normalization
            image_bgr = cv2.imread(src_path)
            if image_bgr is None: raise IOError("Could not read image")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            normalized_rgb = normalizer.transform(image_rgb)
            output_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
            
            if not cv2.imwrite(dest_path, output_bgr):
                raise IOError("Failed to write normalized image")
        return True, None
    except Exception as e:
        return False, f"Error on {os.path.basename(src_path)}: {e}"

def verify_integrity(output_dir):
    """Verifies that image and mask file counts match in the output directory."""
    # ... (This can be a simplified version of your previous verification function) ...
    logging.info(f"Verifying integrity of output files...")
    for label_name in ["CANCER", "NOT_CANCER"]:
        image_dir = os.path.join(output_dir, "TEST", label_name)
        mask_dir = os.path.join(output_dir, "TEST", f"{label_name}_MASK")
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)): continue
        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        if image_files != mask_files:
            raise ValueError(f"Integrity check FAILED: Image and mask file lists do not match for {label_name}.")
    logging.info(f"Integrity verification PASSED.")

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. CONFIGURATION ---

    # Path to the directory containing the original, unnormalized TEST set patches.
    # This directory should contain 'CANCER', 'NOT_CANCER', 'CANCER_MASK', etc.
    TEST_SET_DIR = r'D:\Usuario\Desktop\Base_de_dados\CATCH\RAW_TEST_SET'
    
    # Path to the directory from the TRAINING run that contains the 'normalization_stats.json' file.
    STATS_DIR = r'D:\Usuario\Desktop\Base_de_dados\CATCH\PATCHES_SUBSET_10\VAHADANE\VAHADANE_seed_42'
    
    # Path to the new directory where the normalized TEST set will be saved.
    OUTPUT_DIR = r'D:\Usuario\Desktop\Base_de_dados\CATCH\NORMALIZED_TEST_SET'
    
    NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)

    # --- 2. SETUP ---
    if os.path.exists(OUTPUT_DIR):
        if input(f"Output directory '{OUTPUT_DIR}' exists. Delete and proceed? (yes/no): ").lower() != 'yes':
            sys.exit("Exiting.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    try:
        logging.info("--- Starting Test Set Normalization Script ---")
        
        # --- 3. LOAD NORMALIZER ---
        normalizer, method_name = load_normalizer_from_stats(STATS_DIR)

        # --- 4. PREPARE FILE LIST AND OUTPUT DIRECTORIES ---
        tasks = []
        logging.info(f"Scanning input directory: {TEST_SET_DIR}")
        for label_name in ["CANCER", "NOT_CANCER"]:
            for folder_type in ["", "_MASK"]:
                subfolder = f"{label_name}{folder_type}"
                src_dir = os.path.join(TEST_SET_DIR, subfolder)
                dest_dir = os.path.join(OUTPUT_DIR, "TEST", subfolder) # Create a 'TEST' subfolder for consistency
                
                if not os.path.isdir(src_dir):
                    logging.warning(f"Source sub-directory not found, skipping: {src_dir}")
                    continue
                
                os.makedirs(dest_dir, exist_ok=True)
                is_mask = "_MASK" in subfolder
                
                for filename in os.listdir(src_dir):
                    if filename.lower().endswith('.png'):
                        src_path = os.path.join(src_dir, filename)
                        dest_path = os.path.join(dest_dir, filename)
                        tasks.append((src_path, dest_path, is_mask, normalizer))
        
        if not tasks:
            raise FileNotFoundError("No valid files found to process in the input directory.")
            
        logging.info(f"Found {len(tasks)} total files (images and masks) to process.")

        # --- 5. PROCESS FILES IN PARALLEL ---
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Create a dictionary to map futures to their source paths for better error reporting
            future_to_path = {executor.submit(process_file, *task): task[0] for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks), desc="Normalizing Test Set"):
                src_path = future_to_path[future]
                try:
                    success, msg = future.result()
                    if not success:
                        errors.append(msg)
                except Exception as exc:
                    error_msg = f"Error processing {os.path.basename(src_path)}: {exc}"
                    errors.append(error_msg)

        if errors:
            for error in errors:
                logging.error(f"Processing error: {error}")
            raise RuntimeError(f"Failed to process {len(errors)} files. See log for details.")

        # --- 6. FINAL VERIFICATION ---
        verify_integrity(os.path.join(OUTPUT_DIR, "TEST"))
        
        logging.info("--- Test set normalization completed successfully! ---")

    except Exception as e:
        logging.critical(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
        
    logging.info("--- Script Finished ---")