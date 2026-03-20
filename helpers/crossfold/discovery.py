from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PATIENT_RE = re.compile(r"PATIENT_(\d+)_")


def extract_patient_id(filename: str) -> int | None:
    """Extract the numeric patient identifier from a patch filename."""

    match = PATIENT_RE.search(filename)
    if match is None:
        return None
    return int(match.group(1))


def load_patch_dataset(data_dir: Path) -> pd.DataFrame:
    """Load valid image and mask pairs from Stage 2 patch output folders."""

    logging.info("Loading Stage 5 data from %s", data_dir)
    rows: list[dict[str, object]] = []
    for label_name in ("CANCER", "NOT_CANCER"):
        image_dir = data_dir / label_name
        mask_dir = data_dir / f"{label_name}_MASK"
        label = 1 if label_name == "CANCER" else 0
        if not image_dir.is_dir() or not mask_dir.is_dir():
            logging.warning("Missing dirs for %s. Skipping.", label_name)
            continue

        image_files = sorted(
            path.name for path in image_dir.iterdir() if path.suffix.lower() == ".png"
        )
        mask_files = {path.name for path in mask_dir.iterdir() if path.suffix.lower() == ".png"}
        logging.info("Scanning %s: %s images", label_name, len(image_files))

        for filename in tqdm(image_files, desc=f"Indexing {label_name}", leave=False):
            if filename not in mask_files:
                logging.warning("Mask not found for %s. Skipping.", filename)
                continue
            patient_id = extract_patient_id(filename)
            if patient_id is None:
                logging.warning("Could not extract patient id from %s. Skipping.", filename)
                continue

            image_path = image_dir / filename
            mask_path = mask_dir / filename
            if not image_path.is_file() or not mask_path.is_file():
                logging.warning("Invalid paths for %s. Skipping.", filename)
                continue

            rows.append(
                {
                    "patient_id": patient_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "label": label,
                    "filename": filename,
                }
            )

    if not rows:
        raise ValueError("No valid, readable image/mask pairs were found.")
    dataset = pd.DataFrame(rows)
    logging.info(
        "Loaded %s patch pairs from %s patients.",
        len(dataset),
        dataset["patient_id"].nunique(),
    )
    return dataset
