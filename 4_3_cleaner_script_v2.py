import os
import sys
import shutil
import time
import numpy as np
import logging
import concurrent.futures
from collections import Counter
from functools import partial
from pathlib import Path

# --- 1. SETUP LOGGING (Unchanged) ---
def setup_logging():
    """Configures logging to output to both console and a file."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)
    
    # Console handler for high-level progress and final results
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler for detailed, granular logs of every operation
    file_handler = logging.FileHandler('production_filtering.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# --- 2. VERIFY DEPENDENCIES (Unchanged) ---
try:
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError("The 'ximgproc' module is missing.")
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Critical libraries are missing. {e}")
    print("Please ensure 'opencv-contrib-python' and 'tqdm' are installed.")
    sys.exit(1)

# --- 3. OPTIMAL PARAMETERS (UPDATED) ---
# These parameters were determined by the rigorous, parallelized grid search
# and validated on a held-out test set. They are the core of our scientific findings.
OPTIMAL_PARAMS = {
    'k': 150,
    'min_size': 100,
    'bg_intensity_thresh': 235,
    'erosion_px': 0,
    'contamination_rate_thresh': 0.05  # This is our decision boundary 'tau'
}

# --- 4. CORE FUNCTION (REPLACED) ---

def get_roi_contamination(image_path, mask_path, params):
    """
    Calculates the ROI Contamination Rate. This scientifically valid metric
    replaces the old, flawed background percentage calculation.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError("Image could not be read.")

        # 1. Load the specialist's annotation mask (M)
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            raise IOError("Mask could not be read.")
        
        _, roi_mask = cv2.threshold(roi_mask, 1, 255, cv2.THRESH_BINARY)

        # 2. Run graph segmenter to get the predicted background (B)
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
        segment_map = segmentator.processImage(image)
        
        background_mask = np.zeros(image.shape[:2], dtype=bool)
        num_segments = np.max(segment_map) + 1
        
        for seg_id in range(num_segments):
            segment_pixel_mask = (segment_map == seg_id)
            # Ensure the segment is not empty before calculating mean
            if np.any(segment_pixel_mask):
                avg_color = cv2.mean(image, mask=segment_pixel_mask.astype(np.uint8))
                avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
                if avg_intensity > params['bg_intensity_thresh']:
                    background_mask[segment_pixel_mask] = True
        
        # 3. Calculate the ROI Contamination Rate: C = |B ∩ M| / |M|
        M = (roi_mask > 0).astype(np.uint8)
        
        # Note: erosion_px is 0 based on our findings, so this step does nothing,
        # but the code is kept for completeness if parameters are ever re-tuned.
        erosion_px = params.get('erosion_px', 0)
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_px+1, 2*erosion_px+1))
            M = cv2.erode(M, kernel)
            
        roi_area = M.sum()
        if roi_area == 0:
            logging.getLogger().debug(f"ROI area for {Path(image_path).name} is zero. Returning NaN.")
            return np.nan

        contamination_pixels = (background_mask & (M.astype(bool))).sum()
        
        return contamination_pixels / roi_area

    except Exception as e:
        logging.getLogger().warning(f"Could not process image {Path(image_path).name}. Error: {e}. Skipping.")
        return None

# --- 5. WORKER FUNCTION FOR PARALLEL PROCESSING (UPDATED) ---

def process_image_worker(image_filename, dirs):
    """
    A single unit of work for one process.
    Analyzes one image using the correct metric and moves it if rejected.
    """
    logger = logging.getLogger()
    try:
        source_image_path = os.path.join(dirs['images'], image_filename)
        source_mask_path = os.path.join(dirs['masks'], image_filename)

        if not os.path.exists(source_mask_path):
            logger.debug(f"Mask not found for image '{image_filename}'. Skipping.")
            return 'skipped_no_mask'

        # Calculate the scientifically validated contamination rate
        contamination_rate = get_roi_contamination(source_image_path, source_mask_path, OPTIMAL_PARAMS)

        if contamination_rate is None or np.isnan(contamination_rate):
            return 'skipped_processing_error'

        # The core decision logic based on our optimization results
        if contamination_rate > OPTIMAL_PARAMS['contamination_rate_thresh']:
            dest_image_path = os.path.join(dirs['rejected_images'], image_filename)
            dest_mask_path = os.path.join(dirs['rejected_masks'], image_filename)
            shutil.move(source_image_path, dest_image_path)
            shutil.move(source_mask_path, dest_mask_path)
            logger.debug(f"Rejected '{image_filename}' with contamination rate: {contamination_rate:.3f}")
            return 'rejected'
        else:
            logger.debug(f"Accepted '{image_filename}' with contamination rate: {contamination_rate:.3f}")
            return 'accepted'
            
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {image_filename}: {e}")
        return 'skipped_unexpected_error'


# --- 6. MAIN SCRIPT LOGIC (MANAGER - UPDATED FOR CLARITY) ---

def filter_images_parallel(image_dir, mask_dir, output_base_dir, max_workers=None):
    """
    Manages the parallel filtering of the entire source dataset.
    """
    logger = logging.getLogger()
    start_time = time.time()
    logger.info("--- Starting Production Image Filtering Process ---")
    logger.info(f"Using optimal parameters: {OPTIMAL_PARAMS}")
    
    if max_workers is None:
        max_workers = os.cpu_count()
    logger.info(f"Distributing work across {max_workers} CPU cores.")

    # Setup output directories
    rejected_images_path = os.path.join(output_base_dir, "REJECTED_IMAGES")
    rejected_masks_path = os.path.join(output_base_dir, "REJECTED_MASKS")
    os.makedirs(rejected_images_path, exist_ok=True)
    os.makedirs(rejected_masks_path, exist_ok=True)
    logger.info(f"Rejected files will be moved to: {output_base_dir}")

    try:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not image_files:
            logger.error(f"No image files found in the source directory: '{image_dir}'.")
            return
    except FileNotFoundError:
        logger.error(f"The source image directory '{image_dir}' does not exist.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    # Bundle directory paths to pass to each worker process
    dirs = {
        'images': image_dir,
        'masks': mask_dir,
        'rejected_images': rejected_images_path,
        'rejected_masks': rejected_masks_path
    }
    
    # Use ProcessPoolExecutor for robust parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        task = partial(process_image_worker, dirs=dirs)
        
        # Use executor.map and tqdm for a clean, real-time progress bar
        results_iterator = executor.map(task, image_files)
        results = list(tqdm(results_iterator, total=len(image_files), desc="Filtering Images", file=sys.stdout))

    result_counts = Counter(results)

    logger.info("\n--- Filtering Complete ---")
    logger.info(f"Total images analyzed: {len(image_files)}")
    logger.info(f"Images Accepted (Kept in source folder): {result_counts['accepted']}")
    logger.info(f"Images Rejected (Moved to output folder): {result_counts['rejected']}")
    skipped_total = result_counts['skipped_no_mask'] + result_counts['skipped_processing_error'] + result_counts['skipped_unexpected_error']
    logger.info(f"Images Skipped (Errors or missing masks): {skipped_total}")
    logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")
    logger.info(f"A detailed log has been saved to: {os.path.abspath('production_filtering.log')}")


# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    setup_logging()
    
    # --- IMPORTANT: Define your source and output paths here ---
    SOURCE_IMAGE_DIR = r"C:\Images_IA_MEDICA\CANCER"
    SOURCE_MASK_DIR = r"C:\Images_IA_MEDICA\CANCER_MASK"
    
    # This is the base directory where rejected files will be placed.
    OUTPUT_DIR = r"C:\Images_IA_MEDICA\FILTERED_OUTPUT"

    # Leave as None to use all available CPU cores.
    NUMBER_OF_WORKERS = None 

    filter_images_parallel(SOURCE_IMAGE_DIR, SOURCE_MASK_DIR, OUTPUT_DIR, max_workers=NUMBER_OF_WORKERS)