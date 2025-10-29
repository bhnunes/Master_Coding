import os
import sys
import logging
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tqdm import tqdm
from typing import Tuple

# -------------------------------
# Logging (Unchanged)
# -------------------------------
def setup_logging():
    """Configures the root logger to output to both a file and the console."""
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Log file will be created in the same directory as the script
    file_handler = logging.FileHandler('data_zip.log', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    logging.info("Logging configured. Output will be saved to data_zip.log")

# -------------------------------
# Core helpers (Slightly modified)
# -------------------------------

def add_dir_flat(ziph: ZipFile, base_dir: Path, pbar: tqdm) -> Tuple[int, int]:
    """
    Add base_dir contents to a zip file, updating a tqdm progress bar.
    The archive names will be relative to the base_dir (e.g., CANCER/image.png).
    Returns (files_added, bytes_total).
    """
    files_added = 0
    bytes_total = 0
    
    # List files first to correctly update the progress bar
    files_to_add = [f for f in base_dir.rglob("*") if f.is_file()]

    for file_path in files_to_add:
        arcname = file_path.relative_to(base_dir.parent)
        try:
            ziph.write(file_path, arcname=str(arcname))
            files_added += 1
            try:
                bytes_total += file_path.stat().st_size
            except Exception:
                pass # Ignore if file is gone, etc.
        except Exception as e:
            logging.warning(f"Failed to add {file_path} -> {arcname}: {e}")
        pbar.update(1)
        
    return files_added, bytes_total

# -------------------------------
# Main (Heavily Refactored and Simplified)
# -------------------------------
def main():
    # --- CONFIGURATION ---
    # This is the only path you need to set.
    # It should point to the output directory from your data preparation script.
    # e.g., r'D:\...\NORMALIZED_SPLITS\REINHARD_seed_42'
    ROOT = r"D:\Usuario\Desktop\Base_de_dados\CHILE\PATCHES\NOT_NORMALIZED\NOT_NORMALIZED_seed_42"
    
    # Optional: Set a different output dir for the zip file. If None, it saves alongside the ROOT folder.
    OUT = None 
    
    # 1 is a good trade-off between speed and size. 0 is no compression. 9 is max.
    COMPRESS_LEVEL = 1 
    # --- END CONFIGURATION ---

    setup_logging()

    root_path = Path(ROOT).expanduser()
    
    # Determine the output directory for the zip file
    if OUT:
        out_dir = Path(OUT).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # By default, save the zip file one level up from the root directory
        out_dir = root_path.parent

    # Define the name of the zip file based on the root folder's name
    zip_name = f"{root_path.name}.zip"
    zip_path = out_dir / zip_name

    # Check if the required TRAIN/VALIDATION/TEST folders exist
    include_dirs = ("TRAIN", "VALIDATION", "TEST")
    source_dirs = [root_path / d for d in include_dirs]
    
    if not any(d.is_dir() for d in source_dirs):
        logging.error(f"FATAL: None of the required directories (TRAIN, VALIDATION, TEST) were found in '{root_path}'.")
        sys.exit(1)

    # --- Pre-calculate total number of files for an accurate progress bar ---
    logging.info("Counting total files to be zipped...")
    total_files_to_zip = sum(1 for d in source_dirs if d.is_dir() for f in d.rglob("*") if f.is_file())
    
    if total_files_to_zip == 0:
        logging.warning("No files found to zip. Exiting.")
        return

    logging.info(f"Found {total_files_to_zip} files to archive.")
    
    # Allow overwriting if the zip file already exists
    if zip_path.exists():
        logging.warning(f"Overwriting existing archive: {zip_path}")
        zip_path.unlink(missing_ok=True)

    logging.info(f"Creating archive: {zip_name} ...")

    files_added_total = 0
    bytes_total = 0
    
    try:
        # ZIP_DEFLATED is standard. allowZip64=True is crucial for large datasets (>4GB)
        with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL, allowZip64=True) as ziph:
            with tqdm(total=total_files_to_zip, desc="Zipping Files", unit="file") as pbar:
                for src_dir in source_dirs:
                    if src_dir.is_dir():
                        f_count, b_count = add_dir_flat(ziph, src_dir, pbar)
                        files_added_total += f_count
                        bytes_total += b_count
                    else:
                        logging.warning(f"Directory not found, skipping: {src_dir}")

        # --- Final Summary ---
        logging.info("===============================================")
        logging.info("Zipping finished successfully!")
        logging.info(f"Archive created at: {zip_path}")
        logging.info(f"Total files archived: {files_added_total}")
        final_zip_size_gb = zip_path.stat().st_size / 1e9
        logging.info(f"Final zip file size: {final_zip_size_gb:.3f} GB")
        logging.info(f"(Approx. raw data size was: {bytes_total / 1e9:.3f} GB)")
        logging.info("===============================================")

    except Exception as e:
        logging.exception(f"An error occurred while creating the zip file: {e}")
        # Clean up partial zip file on error
        if zip_path.exists():
            zip_path.unlink()


if __name__ == "__main__":
    main()