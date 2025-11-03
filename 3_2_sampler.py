import os
import random
import shutil
from collections import defaultdict
import re
from tqdm import tqdm
import logging
import sys

# --- CONFIGURATION ---
# The base directory where your 'PATCHES' folder is located.
PROJECT_BASE_PATH = r"D:\Usuario\Desktop\Base_de_dados\CATCH" 

# The name of the source directory with all the patches.
SOURCE_PATCH_DIR_NAME = "PATCHES"

# The desired ratio of images to keep (0.2 = 20%).
SAMPLE_RATIO = 0.10

# Set to True to print what the script would do without actually moving any files.
# Set to False to perform the actual file moving operation.
DRY_RUN = False 

# --- NEW: LOGGING CONFIGURATION ---
LOG_FILE = "downsampling.log"

# ---------------------------

def setup_logging():
    """Configures logging to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'), # Append to the log file
            logging.StreamHandler(sys.stdout)       # Print to the console
        ]
    )

def main():
    """
    Performs stratified random sampling of patch files to create a smaller,
    representative subset for training.
    """
    setup_logging()
    
    base_patch_dir = os.path.join(PROJECT_BASE_PATH, SOURCE_PATCH_DIR_NAME)
    subset_dir_name = f"{SOURCE_PATCH_DIR_NAME}_SUBSET_{int(SAMPLE_RATIO * 100)}"
    subset_patch_dir = os.path.join(PROJECT_BASE_PATH, subset_dir_name)

    logging.info("--- Patch Downsampling Utility ---")
    logging.info(f"Source Directory: {base_patch_dir}")
    logging.info(f"Target Directory: {subset_patch_dir}")
    logging.info(f"Sampling Ratio:   {SAMPLE_RATIO:.2f} ({int(SAMPLE_RATIO * 100)}%)")
    if DRY_RUN:
        logging.info("\n*** DRY RUN MODE IS ACTIVE. NO FILES WILL BE MOVED. ***")
    else:
        logging.info("\n*** LIVE MODE IS ACTIVE. FILES WILL BE MOVED using shutil.move (cut-and-paste). ***")

    if not os.path.isdir(base_patch_dir):
        logging.error(f"Error: Source directory not found at '{base_patch_dir}'. Aborting.")
        return

    # --- Step 1: Group all patches by class and source slide ---
    logging.info("Step 1: Gathering and grouping all patch files...")
    
    grouped_patches = defaultdict(lambda: defaultdict(list))
    patient_id_pattern = re.compile(r"PATIENT_(\d+)_")

    classes = ['CANCER', 'NOT_CANCER']
    for class_name in classes:
        class_path = os.path.join(base_patch_dir, class_name)
        if not os.path.isdir(class_path):
            logging.warning(f"Directory for class '{class_name}' not found. Skipping.")
            continue
            
        for filename in tqdm(os.listdir(class_path), desc=f"Scanning {class_name}", unit="file"):
            match = patient_id_pattern.search(filename)
            if match:
                slide_id = match.group(1)
                full_path = os.path.join(class_path, filename)
                grouped_patches[class_name][slide_id].append(full_path)

    logging.info("Grouping complete.")

    # --- Step 2: Create destination directories and perform stratified sampling ---
    logging.info("Step 2: Performing stratified sampling and moving files...")
    total_files_to_move = 0
    total_files_moved = 0
    
    for class_name, slides in grouped_patches.items():
        dest_img_dir = os.path.join(subset_patch_dir, class_name)
        dest_mask_dir = os.path.join(subset_patch_dir, f"{class_name}_MASK")
        
        if not DRY_RUN:
            os.makedirs(dest_img_dir, exist_ok=True)
            os.makedirs(dest_mask_dir, exist_ok=True)
        
        logging.info(f"Processing class: {class_name}")
        for slide_id, files in slides.items():
            num_files = len(files)
            num_to_sample = int(num_files * SAMPLE_RATIO)
            if num_to_sample == 0 and num_files > 0:
                logging.warning(f"  - Slide '{slide_id}': Found {num_files} patches, but sample count is 0 due to low ratio. No files will be moved for this group.")
                continue
            
            total_files_to_move += num_to_sample
            
            logging.info(f"  - Slide '{slide_id}': Found {num_files} patches. Sampling {num_to_sample}.")
            
            selected_files = random.sample(files, num_to_sample)
            
            for img_path in tqdm(selected_files, desc=f"    Moving '{slide_id}'", unit="file", leave=False):
                filename = os.path.basename(img_path)
                
                source_mask_path = os.path.join(base_patch_dir, f"{class_name}_MASK", filename)
                dest_img_path = os.path.join(dest_img_dir, filename)
                dest_mask_path = os.path.join(dest_mask_dir, filename)
                
                if not DRY_RUN:
                    try:
                        shutil.move(img_path, dest_img_path)
                        if os.path.exists(source_mask_path):
                            shutil.move(source_mask_path, dest_mask_path)
                        else:
                            logging.warning(f"Mask file not found for {filename}")
                        total_files_moved += 1
                    except Exception as e:
                        logging.error(f"Error moving {filename}: {e}")

    logging.info("--- Summary ---")
    if DRY_RUN:
        logging.info(f"Dry run complete. Would have selected and moved {total_files_to_move} images (and their masks).")
        logging.info("To perform the operation, set DRY_RUN = False and run the script again.")
    else:
        logging.info(f"Operation complete. Moved {total_files_moved} images (and their masks) to '{subset_patch_dir}'.")
    logging.info(f"A detailed log of this operation has been saved to '{LOG_FILE}'.")

if __name__ == "__main__":
    main()