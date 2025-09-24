import os
import sys
import shutil
import time
import numpy as np

try:
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError
    from tqdm import tqdm
except ImportError:
    print("Error: Please ensure 'opencv-contrib-python' and 'tqdm' are installed.")
    print("Activate your virtual environment and run 'pip install opencv-contrib-python tqdm'")
    sys.exit(1)

# --- 1. OPTIMAL PARAMETERS (from the tuning experiment) ---
OPTIMAL_PARAMS = {
    'k': 300,
    'min_size': 50,
    'bg_intensity_thresh': 235,
    'bg_percentage_thresh': 0.05  # The final classification threshold
}

# --- 2. CORE SEGMENTATION FUNCTION ---

def get_background_percentage(image_path, params):
    """
    Processes a single image using the optimized Graph-Based method and returns the
    calculated background percentage. Returns None if the image cannot be processed.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError("Image could not be read by OpenCV.")
        
        # Create the graph segmentation object with the optimal parameters
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
        segment_map = segmentator.processImage(image)
        
        num_segments = np.max(segment_map) + 1
        background_pixel_count = 0
        
        # Heuristic: Identify background segments by their high average intensity
        for seg_id in range(num_segments):
            segment_mask = (segment_map == seg_id)
            # Ensure the mask is not empty to avoid errors with cv2.mean
            if np.sum(segment_mask) > 0:
                avg_color = cv2.mean(image, mask=segment_mask.astype(np.uint8))
                # avg_color is (B, G, R, alpha), so we average the BGR channels
                avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
                if avg_intensity > params['bg_intensity_thresh']:
                    background_pixel_count += np.sum(segment_mask)
        
        # Calculate the final percentage
        return background_pixel_count / image.size
        
    except Exception as e:
        print(f"\nWarning: Could not process image {os.path.basename(image_path)}. Error: {e}. Skipping.")
        return None

# --- 3. MAIN SCRIPT LOGIC ---

def filter_images(patches_dir, masks_dir, output_dir):
    """
    Iterates through a folder of patches, applies the filtering logic, and moves
    rejected patches and their corresponding masks to a new directory.
    """
    start_time = time.time()
    print("--- Starting Patch Filtering Process ---")
    
    # --- Setup Output Directories ---
    rejected_patches_path = os.path.join(output_dir, "PATCH_REJECTED")
    rejected_masks_path = os.path.join(output_dir, "MASK_REJECTED")
    
    print(f"Creating output directory for rejected patches: {rejected_patches_path}")
    os.makedirs(rejected_patches_path, exist_ok=True)
    print(f"Creating output directory for rejected masks: {rejected_masks_path}")
    os.makedirs(rejected_masks_path, exist_ok=True)
    
    # --- Discover Image Patches ---
    try:
        image_files = [f for f in os.listdir(patches_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not image_files:
            print(f"Error: No image files found in '{patches_dir}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: The patches directory '{patches_dir}' does not exist.")
        return

    print(f"Found {len(image_files)} images to process.")

    # --- Initialize Counters ---
    accepted_count = 0
    rejected_count = 0
    
    # --- Main Processing Loop ---
    pbar = tqdm(image_files, desc="Filtering Patches")
    for patch_filename in pbar:
        source_patch_path = os.path.join(patches_dir, patch_filename)
        source_mask_path = os.path.join(masks_dir, patch_filename)
        
        # Robustness check: Ensure the corresponding mask exists
        if not os.path.exists(source_mask_path):
            print(f"\nWarning: Mask not found for patch '{patch_filename}'. Skipping.")
            continue
            
        # Analyze the patch
        background_percentage = get_background_percentage(source_patch_path, OPTIMAL_PARAMS)
        
        if background_percentage is None:
            # Error occurred during processing
            continue

        # The core decision logic
        if background_percentage > OPTIMAL_PARAMS['bg_percentage_thresh']:
            # This patch is REJECTED
            rejected_count += 1
            
            # Define destination paths
            dest_patch_path = os.path.join(rejected_patches_path, patch_filename)
            dest_mask_path = os.path.join(rejected_masks_path, patch_filename)
            
            # Move both the patch and the mask
            try:
                shutil.move(source_patch_path, dest_patch_path)
                shutil.move(source_mask_path, dest_mask_path)
            except Exception as e:
                print(f"\nError moving files for patch '{patch_filename}'. Error: {e}")
        else:
            # This patch is ACCEPTED
            accepted_count += 1
            
    print("\n--- Filtering Complete ---")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images Accepted: {accepted_count}")
    print(f"Images Rejected: {rejected_count}")
    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    
    PATCHES_DIR = r"D:\Usuario\Downloads\unzipped_master\CANCER"
    MASKS_DIR = r"D:\Usuario\Downloads\unzipped_master\CANCER_MASK"
    OUTPUT_DIR = r"D:\Usuario\Downloads\unzipped_master\REJECTED"

    filter_images(PATCHES_DIR, MASKS_DIR, OUTPUT_DIR)