"""
Download N random Google Drive files by ID (from CSV index) with progress.

Requires:
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib pandas tqdm

Place your OAuth 2.0 client file as credentials.json alongside this script (Desktop app).
On first run, a browser window will prompt consent; token.json will be created for reuse.
"""

import os
import sys
import time
import math
import random
import shutil
import pathlib
import logging
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# =========================
# ====== GLOBAL VARS ======
# =========================

# One or more CSVs produced earlier in Colab (columns: name,id)
CSV_PATHS: List[str] = [
    r"D:\Usuario\Desktop\Master_Coding\Master_Coding\DOWNLOAD_DRIVE\IDS_list.csv",   # <- change/add your CSVs here
    # r"./another_index.csv",
]

DEST_DIR = r"D:\Usuario\Desktop\Base_de_dados\DIAGSET\IMAGES"         # Where files will be saved
N_TARGET = 50                     # Stop when folder has at least this many files
RANDOM_SEED = 42                  # Set None for non-deterministic
INCLUDE_SHARED_DRIVES = True      # If any file IDs are from Shared drives

# OAuth scope—read-only is enough for downloading
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Chunk size for download (bytes). Big chunks = fewer API calls; adjust if needed.
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
def get_drive_service() -> "googleapiclient.discovery.Resource":
    """Authenticate and return Drive v3 service (installed app flow)."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Download an OAuth client (Desktop App) "
                    "from Google Cloud Console and place it next to this script."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

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


# =========================
# === DRIVE METADATA ======
# =========================
def get_file_metadata(service, file_id: str) -> Dict:
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
    # Ensure parent dir exists
    ensure_dir(os.path.dirname(dest_path))

    # Open in binary write
    with open(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
        done = False
        desc = os.path.basename(dest_path)
        total = file_size if file_size is not None else None

        # tqdm config
        if total is None:
            # Unknown size—still show progress bar in bytes
            pbar = tqdm(unit="B", unit_scale=True, desc=desc)
        else:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)

        last_reported = 0
        try:
            while not done:
                status, done = downloader.next_chunk()
                if status is not None:
                    # status.progress() is a float [0,1]
                    if total is not None:
                        current = int(status.progress() * total)
                        delta = current - last_reported
                        if delta > 0:
                            pbar.update(delta)
                            last_reported = current
                    else:
                        # If size unknown, still tick based on fraction
                        pbar.set_postfix(progress=f"{status.progress()*100:.1f}%")
            # Ensure bar completes
            if total is not None and last_reported < total:
                pbar.update(total - last_reported)
        finally:
            pbar.close()

    # Atomic rename on success
    os.replace(tmp_path, dest_path)


# =========================
# ======= MAIN LOOP =======
# =========================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    ensure_dir(DEST_DIR)

    # 1) Load CSV index
    df = load_index(CSV_PATHS)
    logger.info(f"Loaded {len(df)} unique IDs from CSV(s).")

    # 2) Short-circuit if folder already has N files
    existing = list_existing_files(DEST_DIR)
    existing_count = len(existing)
    if existing_count >= N_TARGET:
        logger.info(
            f"Destination already has {existing_count} files (>= N_TARGET={N_TARGET}). "
            "No download needed."
        )
        return

    # 3) Random selection (sample more than N to account for skips)
    population = df.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle
    candidates = population.to_dict(orient="records")

    # 4) Drive service
    service = get_drive_service()

    downloaded_now = 0
    i = 0

    # Loop while we still need more files and still have candidates
    while (existing_count + downloaded_now) < N_TARGET and i < len(candidates):
        rec = candidates[i]
        i += 1
        file_id = rec["id"]
        fallback_name = rec["name"] or f"{file_id}"

        # Fetch metadata (get true name + size)
        try:
            meta = get_file_metadata(service, file_id)
        except HttpError as e:
            logger.warning(f"Skipping id={file_id} due to metadata error: {e}")
            continue

        name = meta.get("name") or fallback_name
        size_str = meta.get("size")
        file_size = int(size_str) if size_str is not None else None

        # Resolve destination (avoid name collisions by appending _<id> when needed)
        dest_name = name
        if dest_name in existing:
            # If same size -> consider already downloaded; else append id to avoid clobbering
            if file_size is not None and existing[dest_name] == file_size:
                logger.info(f"Already have '{dest_name}' (size match). Skipping.")
                continue
            else:
                stem, ext = os.path.splitext(dest_name)
                dest_name = f"{stem}_{file_id}{ext}"

        dest_path = os.path.join(DEST_DIR, dest_name)

        # If a partial or final file exists with matching size, skip
        if os.path.exists(dest_path):
            if file_size is None or os.path.getsize(dest_path) == file_size:
                logger.info(f"Already have '{dest_name}'. Skipping.")
                continue

        logger.info(f"Downloading: {name} (id={file_id}, size={file_size if file_size else 'unknown'})")

        # Retry logic for robustness
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                download_with_progress(service, file_id, dest_path, file_size)
                downloaded_now += 1
                break
            except HttpError as e:
                # Exponential backoff on 5xx and 403 rate limits
                status = getattr(e, "status_code", None)
                if status in (403, 429, 500, 503) or "userRateLimitExceeded" in str(e):
                    sleep_s = min(60, 2 ** attempt)
                    logger.warning(f"HTTP error (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s}s...")
                    time.sleep(sleep_s)
                else:
                    logger.error(f"Unrecoverable error for id={file_id}: {e}")
                    break
            except Exception as ex:
                logger.error(f"Error downloading id={file_id}: {ex}")
                break

    # Final status
    final_count = existing_count + downloaded_now
    if final_count >= N_TARGET:
        logger.info(f"Reached target: {final_count} files in '{DEST_DIR}'.")
    else:
        logger.info(
            f"Finished. {final_count} files in '{DEST_DIR}' (< N_TARGET={N_TARGET}). "
            f"Likely ran out of candidates or encountered errors."
        )


if __name__ == "__main__":
    main()