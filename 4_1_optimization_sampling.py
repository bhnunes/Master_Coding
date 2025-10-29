import os
import cv2
import numpy as np
import random
import math
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Scientific Logging Setup (Unchanged) ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler("experiment_setup.log", mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- MODULE 1: Image Overlay Generation (Unchanged) ---
# This function is now called on-demand for a much smaller number of images.

def overlay_mask_edges(
    image_path: str,
    mask_path: str,
    out_path: str,
    color=(0, 0, 255),
    thickness: int = 2,
    alpha: float = 1.0
):
    """
    Draws the boundary of a binary mask on the image and saves it to out_path.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            logging.error(f"Could not read image: {image_path}")
            return False

        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            logging.error(f"Could not read mask: {mask_path}")
            return False

        if m.ndim == 3 and m.shape[2] == 4:
            mask_gray = m[:, :, 3]
        elif m.ndim == 3:
            mask_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = m

        _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

        if mask_bin.shape[:2] != img.shape[:2]:
            logging.error(f"Size mismatch: image={img.shape[:2]} vs mask={mask_bin.shape[:2]} for {Path(image_path).name}")
            return False

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        overlay = img.copy()
        cv2.drawContours(overlay, contours, contourIdx=-1, color=color, thickness=thickness)

        result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0) if alpha < 1.0 else overlay
        
        # Ensure the destination directory exists before writing
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, result)
        return True
    except Exception as e:
        logging.error(f"Failed to process and save overlay for {Path(image_path).name}: {e}")
        return False

def _process_overlay_task(args):
    """Helper function to unpack arguments for multiprocessing."""
    return overlay_mask_edges(*args)

# --- MODULE 2: Statistical Sampling (Unchanged) ---

def calculate_cochran_sample_size(confidence_level=0.95, margin_of_error=0.05, proportion=0.5):
    """
    Calculates the sample size using Cochran's formula.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level)
    if z is None:
        raise ValueError("Confidence level must be one of 0.90, 0.95, or 0.99.")
    p = proportion
    e = margin_of_error
    n = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n)

# --- NEW CONSOLIDATED WORKFLOW ---

def generate_samples_on_the_fly(image_folder, mask_folder, base_output_path, color=(0,0,255), thickness=2, alpha=1.0):
    """
    Consolidated function to efficiently create sample pools without storing all overlays.
    It identifies all pairs, selects a random subset, and then generates overlays
    only for that subset, saving them directly to the final destination folders.
    """
    logging.info("--- Experiment Setup Initiated: On-the-Fly Generation ---")
    
    # STEP 1: Discover all valid image-mask pairs to define the total population
    logging.info(f"Scanning for images in: {image_folder}")
    logging.info(f"Scanning for masks in: {mask_folder}")
    
    try:
        image_files = {Path(f).stem: os.path.join(image_folder, f) for f in os.listdir(image_folder)}
        mask_files = {Path(f).stem: os.path.join(mask_folder, f) for f in os.listdir(mask_folder)}
    except OSError as e:
        logging.error(f"Could not read source directories: {e}")
        return

    common_stems = set(image_files.keys()) & set(mask_files.keys())
    
    if not common_stems:
        logging.error("No matching image and mask files found. Please ensure filenames are identical.")
        return
        
    total_images = len(common_stems)
    logging.info(f"Identified {total_images} valid image-mask pairs as the total population.")
    
    # STEP 2: Calculate required sample sizes based on the total population
    required_sample_size = calculate_cochran_sample_size()
    pilot_sample_size = 100
    master_pool_size = math.ceil(total_images * 0.10)

    logging.info("--- Experiment Parameters ---")
    logging.info(f"Statistically Required Sample Size (95% confidence, 5% error): {required_sample_size}")
    logging.info(f"Pilot Sample Size (for estimating proportions): {pilot_sample_size}")
    logging.info(f"Master Candidate Pool Size (10% of total): {master_pool_size}")

    if total_images < (pilot_sample_size + master_pool_size):
        logging.error("Not enough images to create non-overlapping pilot and master pool samples.")
        return

    # STEP 3: Perform unbiased random selection on the LIST of pairs
    all_pairs = list(common_stems)
    random.shuffle(all_pairs)
    
    master_pool_stems = all_pairs[:master_pool_size]
    pilot_sample_stems = all_pairs[master_pool_size : master_pool_size + pilot_sample_size]

    logging.info(f"Randomly selected {len(master_pool_stems)} pairs for the master pool.")
    logging.info(f"Randomly selected {len(pilot_sample_stems)} pairs for the pilot sample.")

    # STEP 4: Generate overlays ONLY for the selected pairs, directly into their destination
    pilot_folder_path = os.path.join(base_output_path, 'pilot_sample')
    master_pool_path = os.path.join(base_output_path, 'master_candidate_pool')

    # Prepare tasks for parallel processing
    tasks = []
    # Master pool tasks
    for stem in master_pool_stems:
        img_path = image_files[stem]
        msk_path = mask_files[stem]
        out_path = os.path.join(master_pool_path, Path(img_path).name)
        tasks.append((img_path, msk_path, out_path, color, thickness, alpha))
    # Pilot sample tasks
    for stem in pilot_sample_stems:
        img_path = image_files[stem]
        msk_path = mask_files[stem]
        out_path = os.path.join(pilot_folder_path, Path(img_path).name)
        tasks.append((img_path, msk_path, out_path, color, thickness, alpha))
        
    num_processes = cpu_count()
    logging.info(f"Starting on-the-fly overlay generation for {len(tasks)} selected images using {num_processes} processes.")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_process_overlay_task, tasks), total=len(tasks), desc="Generating Samples"))
    
    success_count = sum(results)
    logging.info(f"Successfully generated {success_count} out of {len(tasks)} required overlay images.")

    if success_count < len(tasks):
        logging.warning(f"{len(tasks) - success_count} images failed to process. Check 'experiment_setup.log' for details.")

    # STEP 5: Create subdirectories in pilot_sample for manual labeling
    os.makedirs(os.path.join(pilot_folder_path, 'APPROVED'), exist_ok=True)
    os.makedirs(os.path.join(pilot_folder_path, 'REJECTED'), exist_ok=True)
    logging.info(f"Created subdirectories for manual labeling inside '{pilot_folder_path}'")

    os.makedirs(os.path.join(master_pool_path, 'APPROVED'), exist_ok=True)
    os.makedirs(os.path.join(master_pool_path, 'REJECTED'), exist_ok=True)
    logging.info(f"Created subdirectories for manual labeling inside '{master_pool_path}'") 
    
    # Final user instructions
    print("\n\n--- Setup Complete! (Efficient Method) ---")
    print("Overlays for the pilot and master pools were generated on-the-fly.")
    print("NO intermediate 'overlays' folder was created, saving significant disk space.")
    print("\nA detailed log has been saved to 'experiment_setup.log'.")
    print("\nYour next steps are:")
    print(f"1. Go to the '{os.path.basename(pilot_folder_path)}' folder. Move the 100 generated overlays into the 'APPROVED' or 'REJECTED' subfolders based on your criteria.")
    print("2. Once sorted, calculate the proportion of each class (e.g., 70 Approved -> 70%).")
    print(f"3. Use this proportion to calculate your final target numbers from the {required_sample_size} total samples needed.")
    print(f"   - Example: 0.70 * {required_sample_size} = {math.ceil(0.7 * required_sample_size)} 'Approved' images.")
    print(f"   - Example: 0.30 * {required_sample_size} = {math.ceil(0.3 * required_sample_size)} 'Rejected' images.")
    print(f"4. Go to the '{os.path.basename(master_pool_path)}' folder and sample images until you reach your targets.")
    print("5. This final collection will be your statistically representative dataset for the experiment.")


if __name__ == "__main__":
    # --- Configuration ---
    SOURCE_IMAGE_FOLDER = r"D:\Usuario\Desktop\Base_de_dados\CHILE\PATCHES\CANCER"
    SOURCE_IMAGE_FOLDER = os.path.normpath(SOURCE_IMAGE_FOLDER)
    SOURCE_MASK_FOLDER = r"D:\Usuario\Desktop\Base_de_dados\CHILE\PATCHES\CANCER_MASK"
    SOURCE_MASK_FOLDER = os.path.normpath(SOURCE_MASK_FOLDER)
    OUTPUT_BASE_FOLDER = r"D:\Usuario\Desktop\Base_de_dados\CHILE\Optimization_Test"
    OUTPUT_BASE_FOLDER = os.path.normpath(OUTPUT_BASE_FOLDER)

    # --- Execution ---
    # This single function now handles the entire efficient setup process.
    generate_samples_on_the_fly(
        image_folder=SOURCE_IMAGE_FOLDER,
        mask_folder=SOURCE_MASK_FOLDER,
        base_output_path=OUTPUT_BASE_FOLDER
    )