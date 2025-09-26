import os
import sys
import shutil
import time
import numpy as np
import logging
import concurrent.futures
from collections import Counter
from functools import partial

# --- 1. SETUP LOGGING ---
def setup_logging():
    """Configures logging to output to both console and a file."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger # Avoid re-configuring if already set up
    logger.setLevel(logging.DEBUG)
    
    # Console handler for high-level progress
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler for detailed, granular logs
    file_handler = logging.FileHandler('filtering_parallel.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# --- 2. VERIFY DEPENDENCIES ---
try:
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError("The 'ximgproc' module is missing.")
    from tqdm import tqdm
except ImportError as e:
    # Logger might not be set up yet, so use print for this critical error
    print(f"Error: Critical libraries are missing. {e}")
    print("Please ensure 'opencv-contrib-python' and 'tqdm' are installed.")
    print("Activate your virtual environment and run 'pip install \"opencv-contrib-python==4.7.0.72\" tqdm'")
    sys.exit(1)

# --- 3. OPTIMAL PARAMETERS (accessible globally) ---
OPTIMAL_PARAMS = {
    'k': 300,
    'min_size': 50,
    'bg_intensity_thresh': 235,
    'bg_percentage_thresh': 0.05
}

# --- 4. CORE FUNCTIONS ---

def _create_graph_segmentator(params):
    """Creates a graph segmentator, handling different OpenCV versions."""
    if hasattr(cv2.ximgproc, 'segmentation') and hasattr(cv2.ximgproc.segmentation, 'createGraphSegmentation'):
        return cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
    if hasattr(cv2.ximgproc, 'createGraphSegmentation'):
        return cv2.ximgproc.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
    raise RuntimeError("Could not find 'createGraphSegmentation' in your OpenCV installation.")

def get_background_percentage(image_path, segmentator):
    """Processes a single image to calculate the percentage of background pixels."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError("Image could not be read.")
        
        height, width = image.shape[:2]
        total_pixels = height * width
        if total_pixels == 0:
            raise ValueError("Image has zero area.")

        segment_map = segmentator.processImage(image)
        num_segments = np.max(segment_map) + 1
        background_pixel_count = 0

        for seg_id in range(num_segments):
            segment_mask_boolean = (segment_map == seg_id)
            if np.sum(segment_mask_boolean) > 0:
                segment_mask_uint8 = segment_mask_boolean.astype(np.uint8)
                avg_color = cv2.mean(image, mask=segment_mask_uint8)
                avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
                if avg_intensity > OPTIMAL_PARAMS['bg_intensity_thresh']:
                    background_pixel_count += np.sum(segment_mask_boolean)

        return background_pixel_count / total_pixels

    except Exception as e:
        # Using logger configured in the main process
        logging.getLogger().warning(f"Could not process image {os.path.basename(image_path)}. Error: {e}. Skipping.")
        return None

# --- 5. WORKER FUNCTION FOR PARALLEL PROCESSING ---

def process_image_worker(patch_filename, dirs):
    """
    A single unit of work for one process.
    Analyzes one image and moves it if it's rejected.
    Returns a status string: 'accepted', 'rejected', or 'skipped'.
    """
    try:
        source_patch_path = os.path.join(dirs['patches'], patch_filename)
        source_mask_path = os.path.join(dirs['masks'], patch_filename)

        if not os.path.exists(source_mask_path):
            logging.getLogger().warning(f"Mask not found for patch '{patch_filename}'. Skipping.")
            return 'skipped'

        # Each process creates its own segmentator object. This is safe and robust.
        segmentator = _create_graph_segmentator(OPTIMAL_PARAMS)
        
        background_percentage = get_background_percentage(source_patch_path, segmentator)

        if background_percentage is None:
            return 'skipped'

        if background_percentage > OPTIMAL_PARAMS['bg_percentage_thresh']:
            dest_patch_path = os.path.join(dirs['rejected_patches'], patch_filename)
            dest_mask_path = os.path.join(dirs['rejected_masks'], patch_filename)
            shutil.move(source_patch_path, dest_patch_path)
            shutil.move(source_mask_path, dest_mask_path)
            return 'rejected'
        else:
            return 'accepted'
    except Exception as e:
        logging.getLogger().error(f"An unexpected error occurred while processing {patch_filename}: {e}")
        return 'skipped'


# --- 6. MAIN SCRIPT LOGIC (MANAGER) ---

def filter_images_parallel(patches_dir, masks_dir, output_dir, max_workers=None):
    """
    Manages the parallel filtering of images using a process pool.
    """
    logger = logging.getLogger()
    start_time = time.time()
    logger.info("--- Starting Parallel Patch Filtering Process ---")
    
    # Determine the number of workers to use
    if max_workers is None:
        # Default to the number of CPU cores
        max_workers = os.cpu_count()
    logger.info(f"Using up to {max_workers} worker processes.")

    # Setup output directories from the main process
    rejected_patches_path = os.path.join(output_dir, "PATCH_REJECTED")
    rejected_masks_path = os.path.join(output_dir, "MASK_REJECTED")
    os.makedirs(rejected_patches_path, exist_ok=True)
    os.makedirs(rejected_masks_path, exist_ok=True)

    try:
        image_files = [f for f in os.listdir(patches_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not image_files:
            logger.error(f"No image files found in '{patches_dir}'. Please check the path.")
            return
    except FileNotFoundError:
        logger.error(f"The patches directory '{patches_dir}' does not exist.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    # Bundle directory paths to pass to each worker
    dirs = {
        'patches': patches_dir,
        'masks': masks_dir,
        'rejected_patches': rejected_patches_path,
        'rejected_masks': rejected_masks_path
    }
    
    results = []
    # Using ProcessPoolExecutor to manage child processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use functools.partial to create a function that only needs one argument (the filename)
        task = partial(process_image_worker, dirs=dirs)
        
        # Map the task to the list of image files and track progress with tqdm
        # The executor.map function returns results as they are completed
        results_iterator = executor.map(task, image_files)
        pbar = tqdm(results_iterator, total=len(image_files), desc="Filtering Patches", file=sys.stdout)
        
        # The list comprehension pulls results from the progress bar iterator
        results = [result for result in pbar]

    # Tally the results using Counter
    result_counts = Counter(results)

    logger.info("\n--- Filtering Complete ---")
    logger.info(f"Total images analyzed: {len(image_files)}")
    logger.info(f"Images Accepted: {result_counts['accepted']}")
    logger.info(f"Images Rejected: {result_counts['rejected']}")
    logger.info(f"Images Skipped (errors or missing masks): {result_counts['skipped']}")
    logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")
    logger.info(f"A detailed log has been saved to: {os.path.abspath('filtering_parallel.log')}")


# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    # Setup logging as the first step
    setup_logging()
    
    # --- IMPORTANT: Update these paths to match your system ---
    PATCHES_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER_BASE\CANCER"
    MASKS_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER_BASE\CANCER_MASK"
    OUTPUT_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER_BASE\REJECTED"

    # You can specify the number of processes, or leave it as None to use all available CPU cores.
    # For tasks with heavy I/O (like reading from a slow hard drive), using fewer workers
    # than CPU cores might sometimes be faster.
    NUMBER_OF_WORKERS = None 

    filter_images_parallel(PATCHES_DIR, MASKS_DIR, OUTPUT_DIR, max_workers=NUMBER_OF_WORKERS)