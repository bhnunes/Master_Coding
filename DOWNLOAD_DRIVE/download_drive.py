"""
Download N random Google Drive files by ID (from CSV index) with progress.

This script is designed to download a target number of unique image files,
ensuring that for each image, a corresponding annotation and geojson file
exist in specified source directories. Upon successful download of an image,
the companion files are moved to their respective destination folders.

The process is robust and resumable. If stopped, it will recognize existing
files and only download the remaining number needed to reach the target.

Requires:
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib pandas tqdm

Place your OAuth 2.0 client file as credentials.json alongside this script (Desktop app).
On first run, a browser window will prompt consent; token.json will be created for reuse.
"""

import os
import sys
import time
import random
import shutil
import pathlib
import logging
from typing import List,Dict

import pandas as pd
from tqdm import tqdm

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# ===============================
# ====== GLOBAL CONFIGURATION ======
# ===============================

# --- Core Paths ---
# One or more CSVs produced earlier in Colab (columns: name,id)
CSV_PATHS: List[str] = [
    r"D:\Usuario\Desktop\Master_Coding\Master_Coding\DOWNLOAD_DRIVE\IDS_list.csv",
]

# --- Source Paths for Companion Files ---
# The script will verify that for an image 'file.png', corresponding files
# exist in these two folders before attempting to download.
SOURCE_ANNOTATIONS = r"C:\DIAGSET\SOURCE_ANNOTATIONS"
SOURCE_GEOJSON = r"C:\DIAGSET\SOURCE_GEOJSON"

# --- Destination Paths ---
# Where the final files will be stored.
DEST_DIR = r"C:\DIAGSET\DIAGSET\IMAGES"
DEST_ANNOTATIONS = r"C:\DIAGSET\DIAGSET\ANNOTATIONS"
DEST_GEOJSON = r"C:\DIAGSET\DIAGSET\GEOJSON"

# --- Script Behavior ---
# The script will stop once the DEST_DIR has at least this many images.
N_TARGET = 104
# Set to a number for reproducible random sampling, or None for non-deterministic.
RANDOM_SEED = 42
# If any file IDs are from Shared drives, this must be True.
INCLUDE_SHARED_DRIVES = False

# --- Authentication & API ---
# Path to your OAuth 2.0 credentials file.
OAUTH_SECRET_FILE = r"D:\Usuario\Desktop\Master_Coding\Master_Coding\DOWNLOAD_DRIVE\credentials.json"
# The script will create this token file after the first successful login.
# OAuth scope—read-only is enough for downloading.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
# Chunk size for download (bytes). Big chunks = fewer API calls.
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MiB

# =========================
# ====== LOGGING SETUP ====
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =========================
# ===== AUTH / SERVICE ====
# =========================
def get_drive_service():
    """Authenticate and return Drive v3 service (installed app flow)."""
    creds = None
    if not os.path.exists(OAUTH_SECRET_FILE):
        raise FileNotFoundError(
            f"'{OAUTH_SECRET_FILE}' not found. Download an OAuth client "
            "(Desktop App) from Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(OAUTH_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


# =========================
# ===== CSV HANDLING ======
# =========================
def load_index(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concat CSVs, drop duplicates by id, keep columns name,id."""
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p, dtype=str)
        if not {"name", "id"}.issubset(df.columns):
            raise ValueError(f"CSV must have columns 'name' and 'id': {p}")
        frames.append(df[["name", "id"]])
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    return all_df


# =========================
# === FILE SYSTEM UTILS ===
# =========================
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def list_existing_files(dir_path: str) -> Dict[str, int]:
    """Return mapping filename -> size for existing files in dest dir."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for entry in os.scandir(dir_path):
        if entry.is_file():
            try:
                out[entry.name] = entry.stat().st_size
            except FileNotFoundError:
                pass
    return out


def index_companion_files(annotations_path: str, geojson_path: str) -> Dict[str, Dict[str, str]]:
    """
    Scans source directories to find files that have both an annotation and a
    geojson companion. Returns a map of basename -> {annotation_file, geojson_file}.
    """
    logger.info("Scanning for companion files to build index...")
    companion_map = {}
    try:
        ann_files = [e.name for e in os.scandir(annotations_path) if e.is_file()]
        geojson_files = [e.name for e in os.scandir(geojson_path) if e.is_file()]

        logger.info(f"Indexing {len(ann_files)} annotation files...")
        ann_map = {pathlib.Path(f).stem: f for f in tqdm(ann_files, desc="Annotations")}

        logger.info(f"Indexing {len(geojson_files)} geojson files...")
        geojson_map = {pathlib.Path(f).stem: f for f in tqdm(geojson_files, desc="GeoJSONs")}

    except FileNotFoundError as e:
        logger.error(f"A source directory was not found: {e}. Please check your paths.")
        sys.exit(1)

    # Find the intersection of basenames
    common_basenames = set(ann_map.keys()).intersection(set(geojson_map.keys()))

    companion_map = {
        base: {
            'annotation': ann_map[base],
            'geojson': geojson_map[base]
        }
        for base in common_basenames
    }
    logger.info(f"Found {len(companion_map)} complete sets of companion files.")
    return companion_map


# =========================
# === DRIVE METADATA ======
# =========================
def get_file_metadata(service, file_id: str) -> Dict:
    """Fetches metadata (name, size) for a given file ID."""
    fields = "id, name, size, mimeType"
    return service.files().get(
        fileId=file_id,
        fields=fields,
        supportsAllDrives=INCLUDE_SHARED_DRIVES
    ).execute()


# =========================
# === DOWNLOAD HANDLER ====
# =========================
def download_with_progress(service, file_id: str, dest_path: str, file_size: int) -> None:
    """
    Download a Drive file by ID to dest_path with a tqdm progress bar.
    Writes to dest_path + '.part' and renames atomically when complete.
    """
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=INCLUDE_SHARED_DRIVES
    )

    tmp_path = dest_path + ".part"
    ensure_dir(os.path.dirname(dest_path))

    with open(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
        done = False
        desc = os.path.basename(dest_path)
        total = file_size if file_size is not None else None

        with tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
            last_reported = 0
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    if total is not None:
                        current = int(status.progress() * total)
                        delta = current - last_reported
                        if delta > 0:
                            pbar.update(delta)
                            last_reported = current
                    else:
                        pbar.set_postfix_str(f"{status.progress()*100:.1f}%")
            if total is not None and last_reported < total:
                pbar.update(total - last_reported)

    os.replace(tmp_path, dest_path)


# =========================
# ======= MAIN LOOP =======
# =========================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 1) Ensure all destination directories exist
    ensure_dir(DEST_DIR)
    ensure_dir(DEST_ANNOTATIONS)
    ensure_dir(DEST_GEOJSON)
    logger.info(f"Image destination: '{DEST_DIR}'")
    logger.info(f"Annotation destination: '{DEST_ANNOTATIONS}'")
    logger.info(f"GeoJSON destination: '{DEST_GEOJSON}'")

    # 2) Check how many images are already downloaded
    existing_images = list_existing_files(DEST_DIR)
    existing_count = len(existing_images)
    if existing_count >= N_TARGET:
        logger.info(
            f"Destination folder already has {existing_count} images (target is {N_TARGET}). "
            "No download needed."
        )
        return
    logger.info(f"Found {existing_count} existing images. Need to download {N_TARGET - existing_count} more.")

    # 3) Index available companion files from source folders
    companion_map = index_companion_files(SOURCE_ANNOTATIONS, SOURCE_GEOJSON)
    if not companion_map:
        logger.warning("No complete companion file sets found. Cannot download anything.")
        return

    # 4) Load CSV index and filter it
    df = load_index(CSV_PATHS)
    logger.info(f"Loaded {len(df)} unique file IDs from CSV index.")

    # Filter the index to only include files that have companions
    df['basename'] = df['name'].apply(lambda x: pathlib.Path(x).stem if pd.notna(x) else None)
    initial_candidates = len(df)
    df = df[df['basename'].isin(companion_map.keys())]
    logger.info(
        f"Filtered candidates: {initial_candidates} -> {len(df)} "
        "(kept only entries with available companion files)."
    )

    # 5) Prepare candidates for download
    population = df.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle
    candidates = population.to_dict(orient="records")

    # 6) Authenticate and get Drive service
    service = get_drive_service()

    downloaded_now = 0
    candidate_idx = 0

    # 7) Main download loop
    while (existing_count + downloaded_now) < N_TARGET and candidate_idx < len(candidates):
        rec = candidates[candidate_idx]
        candidate_idx += 1
        file_id = rec["id"]
        basename = rec["basename"]

        # --- Check if image is already in destination ---
        # Fetch metadata to get the true filename from Drive
        try:
            meta = get_file_metadata(service, file_id)
        except HttpError as e:
            logger.warning(f"Skipping id={file_id}: Metadata error ({e})")
            continue

        name = meta.get("name")
        if not name:
            logger.warning(f"Skipping id={file_id}: File has no name in Drive.")
            continue

        # If a file with the same name already exists, skip it
        if name in existing_images:
            logger.info(f"Already have '{name}'. Skipping.")
            continue

        size_str = meta.get("size")
        file_size = int(size_str) if size_str is not None else None
        dest_path = os.path.join(DEST_DIR, name)

        logger.info(f"Attempting download: {name} (id={file_id})")

        # --- Download with retry logic ---
        max_retries = 5
        download_success = False
        for attempt in range(1, max_retries + 1):
            try:
                download_with_progress(service, file_id, dest_path, file_size)
                download_success = True
                break
            except HttpError as e:
                status = getattr(e, "status_code", None)
                if status in (403, 429, 500, 503) or "userRateLimitExceeded" in str(e):
                    sleep_s = min(60, 2 ** attempt)
                    logger.warning(f"HTTP error on '{name}' (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s}s...")
                    time.sleep(sleep_s)
                else:
                    logger.error(f"Unrecoverable HTTP error for '{name}': {e}")
                    break
            except Exception as ex:
                logger.error(f"Unexpected error downloading '{name}': {ex}")
                break
        
        # --- If download was successful, move companion files ---
        if download_success:
            logger.info(f"Successfully downloaded '{name}'. Moving companion files.")
            try:
                # Get companion filenames from our index
                companions = companion_map[basename]
                ann_file = companions['annotation']
                geojson_file = companions['geojson']

                # Move annotation
                shutil.move(os.path.join(SOURCE_ANNOTATIONS, ann_file),
                            os.path.join(DEST_ANNOTATIONS, ann_file))
                # Move geojson
                shutil.move(os.path.join(SOURCE_GEOJSON, geojson_file),
                            os.path.join(DEST_GEOJSON, geojson_file))

                logger.info(f"Moved companions for '{basename}'.")
                downloaded_now += 1

            except (FileNotFoundError, KeyError) as e:
                logger.error(f"CRITICAL: Failed to move companion files for '{basename}': {e}.")
                logger.warning(f"Deleting downloaded image '{dest_path}' to maintain consistency.")
                try:
                    os.remove(dest_path)
                except OSError as del_e:
                    logger.error(f"Failed to delete inconsistent image file '{dest_path}': {del_e}")
            except Exception as ex:
                logger.error(f"An unexpected error occurred moving companions for '{basename}': {ex}")
                logger.warning(f"Deleting downloaded image '{dest_path}' to maintain consistency.")
                try:
                    os.remove(dest_path)
                except OSError as del_e:
                    logger.error(f"Failed to delete inconsistent image file '{dest_path}': {del_e}")


    # --- Final Status Report ---
    final_count = existing_count + downloaded_now
    logger.info("=" * 30)
    if final_count >= N_TARGET:
        logger.info(f"SUCCESS: Reached target. Total images in '{DEST_DIR}': {final_count}.")
    else:
        logger.warning(
            f"FINISHED: Process ended before reaching target. "
            f"Total images in '{DEST_DIR}': {final_count} (Target was {N_TARGET}). "
            f"This may be due to running out of valid candidates or encountering persistent errors."
        )


if __name__ == "__main__":
    main()