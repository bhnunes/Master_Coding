import os
import sys
import logging
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple

# -------------------------------
# Logging
# -------------------------------
def setup_logging():
    """Configures the root logger to output to both a file and the console."""
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler('data_zip.log', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    logging.info("Logging configured. Output will be saved to data_zip.log")


# -------------------------------
# Core helpers
# -------------------------------
INCLUDE_DIRS = ("TEST", "TRAIN", "VALIDATION")

def list_folds(root: Path):
    """Yield fold directories whose name matches 'fold_<n>' with n as an integer."""
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("fold_"):
            # Ensure suffix is an int
            try:
                int(p.name.split("_", 1)[1])
                yield p
            except Exception:
                continue


def _count_files_in_dirs(base: Path, dirs=INCLUDE_DIRS) -> int:
    total = 0
    for d in dirs:
        target = base / d
        if target.is_dir():
            for _ in target.rglob("*"):
                # Count only files
                pass
            total += sum(1 for f in target.rglob("*") if f.is_file())
    return total


def add_dir_flat(ziph: ZipFile, base_dir: Path, top_name: str) -> Tuple[int, int]:
    """
    Add base_dir contents to zip with arcnames prefixed by 'top_name/'.
    Ensures there is no extra parent directory in the archive.
    Returns (files_added, bytes_total).
    """
    files_added = 0
    bytes_total = 0
    if not base_dir.exists():
        return 0, 0

    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue
        # arcname ensures top-level is exactly TEST/ or TRAIN/ or VALIDATION/
        rel = file_path.relative_to(base_dir)
        arcname = Path(top_name) / rel
        try:
            ziph.write(file_path, arcname=str(arcname))
            files_added += 1
            try:
                bytes_total += file_path.stat().st_size
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"Failed to add {file_path} -> {arcname}: {e}")
    return files_added, bytes_total


def build_zip_for_fold(
    fold_dir: Path,
    out_dir: Path,
    compresslevel: int = 1,
) -> Tuple[str, int, int]:
    """
    Create MASTER_SET_n.zip for a given fold directory in out_dir.
    - Only TEST, TRAIN, VALIDATION are included at archive top-level.
    - Uses ZIP_DEFLATED with fast compresslevel by default.
    Returns (zip_name, files_added, bytes_total).
    """
    fold_name = fold_dir.name  # e.g., 'fold_3'
    try:
        n = int(fold_name.split("_", 1)[1])
    except Exception:
        raise ValueError(f"Invalid fold name: {fold_name}. Expected 'fold_<n>'")

    zip_name = f"MASTER_SET_{n}.zip"
    zip_path = out_dir / zip_name

    # Allow overwriting if exists (safe for idempotent runs)
    if zip_path.exists():
        logging.info(f"Overwriting existing archive: {zip_path}")
        zip_path.unlink(missing_ok=True)

    logging.info(f"[{fold_name}] Creating {zip_name} ...")

    files_added_total = 0
    bytes_total = 0

    # ZIP_DEFLATED with low compresslevel (1) is usually the fastest trade-off
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as ziph:
        for top in INCLUDE_DIRS:
            src = fold_dir / top
            if not src.exists():
                logging.warning(f"[{fold_name}] Missing directory: {src}")
                continue
            fcount, bcount = add_dir_flat(ziph, src, top)
            files_added_total += fcount
            bytes_total += bcount
            logging.info(f"[{fold_name}] Added {fcount} files from {top}/")

    logging.info(f"[{fold_name}] DONE: {zip_name} | Files: {files_added_total} | Size ~ {bytes_total/1e9:.3f} GB (raw)")
    return zip_name, files_added_total, bytes_total


# -------------------------------
# Main
# -------------------------------
def main():
    ROOT=r"D:\Usuario\Desktop\Base_de_dados\MASTER_100_ENSEMBLE\master_split"
    ROOT=os.path.normpath(ROOT)
    OUT=None
    WORKERS=max(os.cpu_count() - 1, 1)
    COMPRESSLEVEL=1 #0 is faster but bigger, 1 is a good trade-off

    setup_logging()

    root = Path(ROOT).expanduser()
    out_dir = Path(OUT).expanduser() if OUT else root
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = sorted(list(list_folds(root)), key=lambda p: int(p.name.split("_", 1)[1]))
    if not folds:
        logging.error(f"No 'fold_n' directories found under: {root}")
        sys.exit(1)

    # Count total files for a global progress indicator (optional)
    logging.info("Counting files across all folds (for progress estimation)...")
    total_files = 0
    for f in folds:
        total_files += _count_files_in_dirs(f, INCLUDE_DIRS)
    logging.info(f"Estimated total files to archive: {total_files}")

    # Run folds in parallel; we show a tqdm over number of folds.
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        future_to_fold = {
            ex.submit(build_zip_for_fold, fold, out_dir, COMPRESSLEVEL): fold for fold in folds
        }
        for future in tqdm(as_completed(future_to_fold), total=len(future_to_fold), desc="Folds zipped", unit="fold"):
            fold = future_to_fold[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logging.exception(f"Failed to build zip for {fold.name}: {e}")

    # Summary
    made = [r for r in results if r]
    made.sort(key=lambda x: int(x[0].split("_")[-1].split(".")[0]))
    total_archived_files = sum(r[1] for r in made)
    total_bytes = sum(r[2] for r in made)

    logging.info("===============================================")
    logging.info("Zipping finished.")
    for zip_name, files_cnt, bytes_cnt in made:
        logging.info(f"{zip_name} -> files: {files_cnt}, ~{bytes_cnt/1e9:.3f} GB (raw)")
    logging.info(f"TOTAL files archived: {total_archived_files}")
    logging.info(f"Approx raw size: {total_bytes/1e9:.3f} GB")
    logging.info("===============================================")


if __name__ == "__main__":
    # For Windows, protect multiprocessing entry
    main()
