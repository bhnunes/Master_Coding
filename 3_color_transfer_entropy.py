import numpy as np
import cv2
import os
import shutil
from collections import defaultdict
from joblib import Parallel, delayed
import logging
import re
import sys
from PIL import Image, UnidentifiedImageError

# --- NEW: Import Tiatoolbox ---
from tiatoolbox.tools import stainnorm

# --- Configuration ---
# Base directories (Adjust as needed)
INPUT_BASE_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER"
OUTPUT_BASE_DIR = r"D:\Usuario\Desktop\Base_de_dados\ABLATION\TIATOOLBOX_NORMALIZED"

# --- NEW: List of normalization methods to test ---
METHODS_TO_TEST = ["Reinhard", "Ruifrok", "Macenko", "Vahadane"]

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

def calculate_image_entropy(image_path):
    """
    Calcula a entropia de Shannon para um dado arquivo de imagem.
    Uma entropia mais alta indica mais 'informação' ou 'complexidade' na imagem.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        if hist.sum() == 0:
            return 0.0
        prob_dist = hist / hist.sum()
        entropy = -np.sum(prob_dist[prob_dist > 0] * np.log2(prob_dist[prob_dist > 0]))
        return entropy
    except Exception as e:
        logging.warning(f"Could not calculate entropy for {image_path}: {e}")
        return 0.0

def create_template_directory(source_dirs, template_dir):
    """
    MODIFICADO: Cria o diretório de templates selecionando a imagem de MAIOR ENTROPIA
    por paciente das pastas de origem combinadas.
    """
    print(f"Creating template directory: {template_dir}")
    logging.info(f"Starting template directory creation. Target: {template_dir}")
    patient_files = defaultdict(list)
    patient_id_pattern = re.compile(r"PATIENT_(\d+)_")
    print("Scanning source directories for patient images...")
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found: {source_dir}. Skipping.", file=sys.stderr)
            continue
        print(f"  Scanning: {source_dir}")
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                match = patient_id_pattern.search(filename)
                if match:
                    patient_id = match.group(1)
                    patient_files[patient_id].append(os.path.join(source_dir, filename))
    if not patient_files:
        raise ValueError("No patient images found. Cannot create templates.")
    print(f"Found images for {len(patient_files)} unique patients.")
    if os.path.exists(template_dir):
        print(f"Clearing existing template directory: {template_dir}")
        shutil.rmtree(template_dir)
    os.makedirs(template_dir)
    print("Selecting template with max entropy for each patient and copying...")
    templates_copied_count = 0
    for patient_id, file_list in patient_files.items():
        if not file_list:
            continue
        best_file = max(file_list, key=lambda f: calculate_image_entropy(f))
        if best_file:
            try:
                shutil.copy2(best_file, template_dir)
                templates_copied_count += 1
            except Exception as e:
                print(f"Warning: Could not copy file {best_file} for patient {patient_id}: {e}", file=sys.stderr)
                logging.warning(f"Could not copy template file {best_file} for patient {patient_id}: {e}")
    if templates_copied_count == 0:
         raise RuntimeError("Failed to copy any template files.")
    print(f"Successfully copied {templates_copied_count} template images (one per patient) to {template_dir}")
    logging.info(f"Successfully copied {templates_copied_count} template images.")


def make_aggregate_target(images_rgb):
    resized = []
    for img in images_rgb:
        if img is None:
            continue
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        resized.append(img)
    if not resized:
        raise ValueError("No valid images to build aggregate target.")
    stack = np.stack(resized, axis=0)               # (N, H, W, 3)
    median = np.median(stack, axis=0).astype(np.uint8)  # (H, W, 3)
    return median

# --- NEW: Generic function to fit normalizers from the template directory ---
def fit_normalizers(template_dir, methods):
    """
    Fits stain normalizers for a list of methods using all images in the template directory.
    
    This function creates an aggregated 'target' from all template images, which is
    scientifically more robust than using a single image.

    Args:
        template_dir (str): Path to the directory containing template images.
        methods (list): A list of method names (e.g., ["Reinhard", "Macenko"]).

    Returns:
        dict: A dictionary mapping method names to their fitted normalizer objects.
    """
    template_files = [os.path.join(template_dir, f) for f in os.listdir(template_dir) 
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not template_files:
        raise ValueError("No template images found in the directory for fitting normalizers.")

    print(f"Loading {len(template_files)} template images for fitting...")
    
    # Load all template images. Tiatoolbox expects RGB, but cv2 reads as BGR.
    # We convert them to RGB before fitting.
    try:
        template_images_rgb = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in template_files]
    except Exception as e:
        raise IOError(f"Failed to read template images. Error: {e}")

    target_rgb = make_aggregate_target(template_images_rgb)

    fitted_normalizers = {}
    print("Fitting normalizers for all specified methods...")
    for method_name in methods:
        try:
            print(f"  - Fitting for '{method_name}'...")
            # Get the appropriate normalizer from the factory
            normalizer = stainnorm.get_normalizer(method_name)
            
            # Fit the normalizer on the entire set of template images.
            # Tiatoolbox handles the aggregation internally to create a robust target.
            normalizer.fit(target_rgb)
            
            fitted_normalizers[method_name] = normalizer
            print(f"    '{method_name}' normalizer fitted successfully.")
        except Exception as e:
            print(f"Error fitting normalizer for method {method_name}: {e}", file=sys.stderr)
            logging.error(f"Error fitting normalizer for method {method_name}: {e}")
            
    return fitted_normalizers

# --- MODIFIED: process_image now uses a normalizer object ---
def process_image(img_filename, input_dir, output_dir, normalizer):
    """
    Applies stain normalization to a single image using a fitted normalizer object.
    """
    input_path = os.path.join(input_dir, img_filename)
    output_path = os.path.join(output_dir, img_filename)

    try:
        # Open with PIL for robust loading
        with Image.open(input_path) as input_pil:
             input_np = np.array(input_pil)
             if input_np.ndim == 2: # Grayscale
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_GRAY2BGR)
             elif input_np.shape[2] == 3: # RGB
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
             elif input_np.shape[2] == 4: # RGBA
                 input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGBA2BGR)
             else:
                 raise ValueError(f"Unsupported channel count: {input_np.shape}")

        if input_bgr.size == 0:
            logging.error(f"Skipping empty image after conversion: {input_path}")
            return
            
        # --- Normalization using Tiatoolbox ---
        # 1. Convert source from BGR to RGB for tiatoolbox
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

        # 2. Apply the pre-fitted normalizer
        normalized_rgb = normalizer.transform(input_rgb)

        # 3. Convert normalized image back to BGR for saving with OpenCV
        output_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
        # --- End Normalization ---

        success = cv2.imwrite(output_path, output_bgr)
        if not success:
             raise IOError(f"Failed to write image to {output_path}")

    except UnidentifiedImageError:
        logging.error(f"Skipping non-image file or corrupted image: {input_path}")
    except FileNotFoundError:
         logging.error(f"Input image not found: {input_path}")
    except Exception as e:
        logging.exception(f"Error processing image: {input_path} - Type: {type(e).__name__}, Error: {e}")


# --- Main Execution Block (MODIFIED) ---
if __name__ == "__main__":
    
    source_directories = [CANCER_INPUT_DIR, NOT_CANCER_INPUT_DIR]

    # --- Step 1: Create Template Directory (No change) ---
    try:
        create_template_directory(source_directories, TEMPLATE_DIR)
    except (ValueError, OSError, RuntimeError) as e:
         print(f"Fatal Error during template creation: {e}. Exiting.", file=sys.stderr)
         logging.critical(f"Fatal Error during template creation: {e}. Exiting.")
         sys.exit(1)

    # --- Step 2: Fit all Normalizers using the Template Directory ---
    try:
        fitted_normalizers = fit_normalizers(TEMPLATE_DIR, METHODS_TO_TEST)
        if not fitted_normalizers:
            raise ValueError("No normalizers could be fitted. Check logs for errors.")
    except (ValueError, IOError) as e:
        print(f"Fatal Error fitting normalizers: {e}. Exiting.", file=sys.stderr)
        logging.critical(f"Fatal Error fitting normalizers: {e}. Exiting.")
        sys.exit(1)

    # --- Step 3: Process Images Sequentially for Each Normalization Method ---
    print("\n--- Starting Full Image Normalization Pipeline ---")
    for method_name, normalizer in fitted_normalizers.items():
        
        print(f"\n===== PROCESSING WITH METHOD: {method_name} =====")
        logging.info(f"Starting processing run for method: {method_name}")

        # Define method-specific output directories
        method_output_base = os.path.join(OUTPUT_BASE_DIR, method_name)
        
        for input_dir in source_directories:
            output_dir = os.path.join(method_output_base, os.path.basename(input_dir))
            
            print(f"\nProcessing Directory: {input_dir}")
            print(f"Output will be saved to: {output_dir}")
            
            if not os.path.isdir(input_dir):
                print(f"Warning: Input directory '{input_dir}' not found. Skipping.", file=sys.stderr)
                logging.warning(f"Input directory '{input_dir}' not found. Skipping.")
                continue

            os.makedirs(output_dir, exist_ok=True)
            
            input_image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not input_image_list:
                print("  No images found to process in this directory.")
                logging.warning(f"No images found to process in directory: {input_dir}")
                continue

            print(f"  Found {len(input_image_list)} images to normalize.")

            batch_size = 500
            total_batches = (len(input_image_list) + batch_size - 1) // batch_size

            for i in range(0, len(input_image_list), batch_size):
                batch_num = (i // batch_size) + 1
                image_batch = input_image_list[i:i + batch_size]
                print(f"  Processing Batch {batch_num} of {total_batches} ({len(image_batch)} images)...")

                Parallel(n_jobs=-1)(
                    delayed(process_image)(img_file, input_dir, output_dir, normalizer) for img_file in image_batch
                )

            print(f"Finished processing directory: {input_dir} for method {method_name}")
            logging.info(f"Finished processing directory: {input_dir} for method {method_name}")

    print(f"\n--- Normalization Complete for All Methods ---")
    print(f"Check '{OUTPUT_BASE_DIR}' for normalized image subdirectories.")
    print(f"Check '{LOG_FILE}' for any errors logged during the process.")
    logging.info("Normalization process finished for all methods.")

