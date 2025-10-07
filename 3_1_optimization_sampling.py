import os
import cv2
import numpy as np
import random
import shutil
import math
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Scientific Logging Setup ---
# Configure logging to provide a detailed, reproducible record of the experiment setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_setup.log"),
        logging.StreamHandler()
    ]
)

# --- MODULE 1: Image Overlay Generation ---

def overlay_mask_edges(
    image_path: str,
    mask_path: str,
    out_path: str,
    color=(0, 0, 255),   # BGR -> red
    thickness: int = 2,  # edge line thickness in pixels
    alpha: float = 1.0   # 1.0 = solid red line; <1.0 = semi-transparent
):
    """
    Draws the boundary of a binary mask on the image. (This function is from your example)
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

        if alpha >= 1.0:
            result = overlay
        else:
            result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)
        
        cv2.imwrite(out_path, result)
        return True
    except Exception as e:
        logging.error(f"Failed to process {Path(image_path).name}: {e}")
        return False

def _process_overlay_task(args):
    """Helper function to unpack arguments for multiprocessing."""
    image_path, mask_path, out_path, color, thickness, alpha = args
    return overlay_mask_edges(image_path, mask_path, out_path, color, thickness, alpha)

def create_overlay_images_parallel(image_folder, mask_folder, output_folder, color=(0,0,255), thickness=2, alpha=1.0):
    """
    Finds image-mask pairs and generates overlay images using multiprocessing.
    """
    logging.info("--- Step 1: Overlay Image Generation ---")
    logging.info(f"Scanning for images in: {image_folder}")
    logging.info(f"Scanning for masks in: {mask_folder}")
    
    image_files = {Path(f).stem: os.path.join(image_folder, f) for f in os.listdir(image_folder)}
    mask_files = {Path(f).stem: os.path.join(mask_folder, f) for f in os.listdir(mask_folder)}

    # Find common files based on filename stem (e.g., "image1" for "image1.png")
    common_stems = set(image_files.keys()) & set(mask_files.keys())
    
    if not common_stems:
        logging.error("No matching image and mask files found. Please ensure filenames are identical.")
        return None
        
    logging.info(f"Found {len(common_stems)} matching image-mask pairs.")
    
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Created output directory for overlays: {output_folder}")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for stem in common_stems:
        image_path = image_files[stem]
        mask_path = mask_files[stem]
        # Preserve the original filename and extension
        original_filename = Path(image_path).name
        out_path = os.path.join(output_folder, original_filename)
        tasks.append((image_path, mask_path, out_path, color, thickness, alpha))

    # Use multiprocessing Pool for parallel execution
    num_processes = cpu_count()
    logging.info(f"Starting overlay generation with {num_processes} parallel processes.")
    
    with Pool(processes=num_processes) as pool:
        # tqdm provides a real-time progress bar
        results = list(tqdm(pool.imap_unordered(_process_overlay_task, tasks), total=len(tasks), desc="Creating Overlays"))

    success_count = sum(results)
    logging.info(f"Successfully generated {success_count} out of {len(tasks)} overlay images.")
    
    if success_count == 0:
        logging.error("Overlay generation failed for all images. Please check logs for errors.")
        return None
        
    return output_folder

# --- MODULE 2: Statistical Sampling ---

def calculate_cochran_sample_size(confidence_level=0.95, margin_of_error=0.05, proportion=0.5):
    """
    Calculates the sample size using Cochran's formula.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if confidence_level not in z_scores:
        raise ValueError("Confidence level must be one of 0.90, 0.95, or 0.99.")
    z = z_scores[confidence_level]
    p = proportion
    e = margin_of_error
    n = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n)

def create_image_samples(source_folder, base_output_path):
    """
    Analyzes a folder of images, calculates sample sizes, and creates sample folders.
    Now operates on the folder of generated overlays.
    """
    logging.info("--- Step 2: Statistical Sampling ---")
    if not os.path.isdir(source_folder):
        logging.error(f"The provided source folder '{source_folder}' is not a valid directory.")
        return

    logging.info(f"Scanning source folder for sampling: {source_folder}")
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    try:
        all_files = [f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions]
    except OSError as e:
        logging.error(f"Error reading directory: {e}")
        return

    total_images = len(all_files)
    if total_images == 0:
        logging.error("No overlay image files found in the source directory for sampling.")
        return

    logging.info(f"Found {total_images} total overlay images for sampling.")

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

    pilot_folder_path = os.path.join(base_output_path, 'pilot_sample')
    master_pool_path = os.path.join(base_output_path, 'master_candidate_pool')

    os.makedirs(pilot_folder_path, exist_ok=True)
    os.makedirs(master_pool_path, exist_ok=True)
    logging.info(f"Created directory: {pilot_folder_path}")
    logging.info(f"Created directory: {master_pool_path}")

    random.shuffle(all_files)
    master_pool_files = all_files[:master_pool_size]
    pilot_sample_files = all_files[master_pool_size : master_pool_size + pilot_sample_size]
    
    logging.info(f"Copying {len(master_pool_files)} images to '{master_pool_path}'...")
    for filename in tqdm(master_pool_files, desc="Copying to Master Pool"):
        shutil.copy2(os.path.join(source_folder, filename), os.path.join(master_pool_path, filename))

    logging.info(f"Copying {len(pilot_sample_files)} images to '{pilot_folder_path}'...")
    for filename in tqdm(pilot_sample_files, desc="Copying to Pilot Sample"):
        shutil.copy2(os.path.join(source_folder, filename), os.path.join(pilot_folder_path, filename))

    print("\n\n--- Setup Complete! ---")
    print("\nA detailed log of this setup has been saved to 'experiment_setup.log'.")
    print("\nYour next steps are:")
    print(f"1. Go to the '{pilot_folder_path}' folder and manually label the 100 images as 'Approved' or 'Rejected'.")
    print("2. Calculate the proportion of each class (e.g., 70% Approved, 30% Rejected).")
    print(f"3. Use this proportion to determine your final target numbers from the {required_sample_size} total samples needed.")
    print(f"   - Example: 0.70 * {required_sample_size} = {math.ceil(0.7 * required_sample_size)} 'Approved' images.")
    print(f"   - Example: 0.30 * {required_sample_size} = {math.ceil(0.3 * required_sample_size)} 'Rejected' images.")
    print(f"4. Go to the '{master_pool_path}' folder and randomly select/label images until you reach your target numbers for both classes.")
    print("5. This final collection of labeled images will be your statistically representative dataset for the experiment.")

if __name__ == "__main__":
    # --- Configuration ---
    # Please DEFINE your source and output paths here
    
    # Path to the folder containing the original images (e.g., patient scans)
    SOURCE_IMAGE_FOLDER = r"C:\Images_IA_MEDICA\CANCER"
    
    # Path to the folder containing the corresponding masks
    SOURCE_MASK_FOLDER = r"C:\Images_IA_MEDICA\CANCER_MASK"
    
    # Base path where all output folders will be created
    # The script will create: .../OUTPUT_BASE/overlays, .../OUTPUT_BASE/pilot_sample, etc.
    OUTPUT_BASE_FOLDER = r"C:\Images_IA_MEDICA\Optimization_Test"

    # --- Execution ---
    # 1. Create overlay images from the source images and masks
    overlays_folder = os.path.join(OUTPUT_BASE_FOLDER, 'overlays')
    generated_overlays_path = create_overlay_images_parallel(
        image_folder=SOURCE_IMAGE_FOLDER,
        mask_folder=SOURCE_MASK_FOLDER,
        output_folder=overlays_folder
    )

    # 2. If overlay generation was successful, create the sampling pools from them
    if generated_overlays_path:
        create_image_samples(
            source_folder=generated_overlays_path,
            base_output_path=OUTPUT_BASE_FOLDER
        )
    else:
        logging.critical("Halting execution because overlay generation failed.")