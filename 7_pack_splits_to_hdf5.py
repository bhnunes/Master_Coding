# -*- coding: utf-8 -*-
"""
7_pack_splits_to_hdf5.py

Purpose
-------
Convert an on-disk split folder structure into three HDF5 files:
    TRAIN.h5, VALIDATION.h5, TEST.h5

Why
---
- Avoids slow "zip -> upload -> download -> unzip" loops in Colab.
- HDF5 provides fast random access with no PNG decode cost at training time
  (you can still do on-the-fly augmentation in the training Dataset).

Expected folder structure under ROOT
------------------------------------
ROOT/
  TRAIN/
    CANCER/
    CANCER_MASK/
    NOT_CANCER/
    NOT_CANCER_MASK/
  VALIDATION/
    CANCER/
    CANCER_MASK/
    NOT_CANCER/
    NOT_CANCER_MASK/
  TEST/   (optional)
    CANCER/
    CANCER_MASK/
    NOT_CANCER/
    NOT_CANCER_MASK/

Notes (Scientific / Reproducibility)
------------------------------------
- This script NEVER changes splits. It only packs what is already on disk.
- Patient leakage is handled upstream (data split script + sanity checks).
- We store metadata in the HDF5 file: build time, root path, and the
  exact mapping filename -> patient_id/label.
"""

import os
import sys
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import h5py
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(str(output_dir / "hdf5_packing.log"), encoding="utf-8")
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_format)
    logger.addHandler(sh)

    logging.info("Logging configured. Output will be saved to hdf5_packing.log")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _list_pngs(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted([p.name for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(".png")])


def _default_sources_for_split(split: str) -> List[Dict]:
    """Matches the training script's 'sources' structure."""
    split = split.upper()
    return [
        {"img": f"{split}/CANCER",     "mask": f"{split}/CANCER_MASK",     "label": 1},
        {"img": f"{split}/NOT_CANCER", "mask": f"{split}/NOT_CANCER_MASK", "label": 0},
    ]


# -----------------------------------------------------------------------------
# Core packer (ported from training script, with reproducibility additions)
# -----------------------------------------------------------------------------
def convert_dataset_to_hdf5(
    dataset_dir: Path,
    output_path: Path,
    sources: List[Dict],
    img_size: int = 224,
    patient_id_regex: str = r"PATIENT_(\d+)_",
    delete_source_folders: bool = False,
) -> Path:
    """
    Robust HDF5 packer (PNG -> HDF5).
    - sources: [{'img': dir, 'mask': dir, 'label': int}, ...]
    - Uses a write pointer to prevent empty rows on read failure.
    - Resizes dataset at the end to match valid sample count.
    - Adds reproducibility metadata (attrs + an index table).
    """
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    if output_path.exists():
        logging.info(f"HDF5 already exists, skipping: {output_path}")
        return output_path

    logging.info(f"--- Packing HDF5: {output_path} ---")

    # 1) Scan pairs
    valid_pairs: List[Dict] = []
    patient_pat = re.compile(patient_id_regex)

    logging.info("Step 1: Scanning file paths...")
    for src in sources:
        img_dir = dataset_dir / src["img"]
        msk_dir = dataset_dir / src["mask"]

        if not img_dir.is_dir():
            logging.warning(f"Missing image dir (skipping): {img_dir}")
            continue
        if not msk_dir.is_dir():
            logging.warning(f"Missing mask dir (skipping): {msk_dir}")
            continue

        fnames = _list_pngs(img_dir)
        if not fnames:
            logging.warning(f"No PNGs found in: {img_dir}")

        for fname in fnames:
            m = patient_pat.search(fname)
            mask_path = msk_dir / fname
            if m and mask_path.exists():
                valid_pairs.append(
                    {
                        "img_path": str(img_dir / fname),
                        "mask_path": str(mask_path),
                        "filename": fname,
                        "label": int(src["label"]),
                        "pid": int(m.group(1)),
                    }
                )

    estimated_total = len(valid_pairs)
    if estimated_total == 0:
        raise RuntimeError(
            f"No valid (image, mask) pairs found for output: {output_path}\n"
            f"Check folder structure and naming (patient id regex: {patient_id_regex})."
        )

    logging.info(f"Found {estimated_total} candidate pairs. Starting pack...")

    # 2) Write
    cv2.setNumThreads(0)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(str(output_path), "w") as h5f:
        # Root attrs (reproducibility)
        h5f.attrs["created_utc"] = str(np.datetime64("now", "s"))
        h5f.attrs["dataset_root"] = str(dataset_dir.resolve())
        h5f.attrs["img_size"] = int(img_size)
        h5f.attrs["patient_id_regex"] = patient_id_regex
        h5f.attrs["sources_json"] = json.dumps(sources, ensure_ascii=False)

        # Resizable datasets
        dset_img = h5f.create_dataset(
            "images",
            shape=(estimated_total, img_size, img_size, 3),
            maxshape=(None, img_size, img_size, 3),
            dtype="uint8",
            chunks=(1, img_size, img_size, 3),
        )
        dset_mask = h5f.create_dataset(
            "masks",
            shape=(estimated_total, img_size, img_size),
            maxshape=(None, img_size, img_size),
            dtype="uint8",
            chunks=(1, img_size, img_size),
        )
        dset_labels = h5f.create_dataset(
            "labels", shape=(estimated_total,), maxshape=(None,), dtype="uint8"
        )
        dset_pids = h5f.create_dataset(
            "patient_ids", shape=(estimated_total,), maxshape=(None,), dtype="int32"
        )
        dset_fnames = h5f.create_dataset(
            "filenames", shape=(estimated_total,), maxshape=(None,), dtype=str_dtype
        )

        write_ptr = 0
        for item in tqdm(valid_pairs, desc=f"Packing {output_path.name}", unit="pair"):
            try:
                img = cv2.imread(item["img_path"], cv2.IMREAD_COLOR)
                if img is None:
                    continue

                msk = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
                if msk is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

                msk = cv2.resize(msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                msk = np.where(msk > 0, 1, 0).astype(np.uint8)

                dset_img[write_ptr] = img
                dset_mask[write_ptr] = msk
                dset_labels[write_ptr] = item["label"]
                dset_pids[write_ptr] = item["pid"]
                dset_fnames[write_ptr] = item["filename"]
                write_ptr += 1

            except Exception as e:
                logging.warning(f"Failed to process {item['img_path']}: {e}")
                continue

        # Final resize
        if write_ptr < estimated_total:
            logging.info(
                f"Resizing HDF5 from {estimated_total} -> {write_ptr} "
                f"(removed {estimated_total - write_ptr} bad files)"
            )
            dset_img.resize((write_ptr, img_size, img_size, 3))
            dset_mask.resize((write_ptr, img_size, img_size))
            dset_labels.resize((write_ptr,))
            dset_pids.resize((write_ptr,))
            dset_fnames.resize((write_ptr,))

    size_gb = output_path.stat().st_size / (1024**3)
    logging.info(f"HDF5 conversion complete: {output_path} ({size_gb:.2f} GB)")

    if delete_source_folders:
        import shutil
        for src in sources:
            for rel in [src["img"], src["mask"]]:
                d = (dataset_dir / rel)
                try:
                    if d.is_dir():
                        d_res = d.resolve()
                        if str(d_res).startswith(str(dataset_dir.resolve())):
                            shutil.rmtree(d_res)
                            logging.info(f"Deleted source folder: {d_res}")
                        else:
                            logging.warning(f"Refused to delete (outside root): {d_res}")
                except Exception as e:
                    logging.warning(f"Could not delete {d}: {e}")

    return output_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # --- CONFIGURATION ---
    ROOT = r"D:\Usuario\Desktop\Base_de_dados\CAMELYON16\PATCHES\NOT_NORMALIZED\NOT_NORMALIZED_seed_42"
    OUT = None
    SPLITS = ("TRAIN", "VALIDATION", "TEST")
    IMG_SIZE = 224
    PATIENT_ID_REGEX = r"PATIENT_(\d+)_"
    DELETE_SOURCE_FOLDERS = False
    # --- END CONFIGURATION ---

    root_path = Path(ROOT).expanduser()

    if OUT:
        out_dir = Path(OUT).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = root_path.parent

    setup_logging(out_dir)

    if not root_path.is_dir():
        logging.critical(f"FATAL: ROOT does not exist or is not a directory: {root_path}")
        sys.exit(1)

    manifest_path = root_path / "manifest.csv"
    manifest_sha = None
    if manifest_path.exists():
        try:
            manifest_sha = sha256_file(manifest_path)
            logging.info(f"Found manifest.csv (sha256={manifest_sha[:16]}...)")
        except Exception as e:
            logging.warning(f"Could not hash manifest.csv: {e}")

    outputs: Dict[str, str] = {}
    for split in SPLITS:
        split_dir = root_path / split
        if not split_dir.is_dir():
            logging.warning(f"Split folder not found, skipping: {split_dir}")
            continue

        h5_name = f"{split}.h5"
        h5_path = out_dir / h5_name

        sources = _default_sources_for_split(split)

        out_h5 = convert_dataset_to_hdf5(
            dataset_dir=root_path,
            output_path=h5_path,
            sources=sources,
            img_size=IMG_SIZE,
            patient_id_regex=PATIENT_ID_REGEX,
            delete_source_folders=DELETE_SOURCE_FOLDERS,
        )

        if manifest_sha is not None:
            try:
                with h5py.File(str(out_h5), "a") as f:
                    f.attrs["manifest_csv_sha256"] = manifest_sha
            except Exception as e:
                logging.warning(f"Could not store manifest hash into {out_h5}: {e}")

        outputs[split] = str(out_h5)

    if not outputs:
        logging.critical("FATAL: No splits were packed. Check that ROOT contains TRAIN/VALIDATION/TEST.")
        sys.exit(1)

    logging.info("===============================================")
    logging.info("HDF5 packing finished successfully!")
    for k, v in outputs.items():
        try:
            gb = Path(v).stat().st_size / (1024**3)
            logging.info(f"{k}: {v} ({gb:.2f} GB)")
        except Exception:
            logging.info(f"{k}: {v}")
    logging.info("===============================================")


if __name__ == "__main__":
    main()
