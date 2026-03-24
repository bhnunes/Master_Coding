from __future__ import annotations

import logging
from pathlib import Path

from helpers.packaging.config import PackagingConfig
from helpers.packaging.discovery import discover_patch_pool_samples
from helpers.packaging.writer import write_patch_dataset_hdf5


def run_packaging_pipeline(config: PackagingConfig) -> Path:
    logging.info("Starting Stage 7 HDF5 packaging from %s", config.base_dir)
    samples = discover_patch_pool_samples(config.base_dir, config.patient_id_regex)
    if not samples:
        raise ValueError(f"No PNG pairs found in cleaned patch pool: {config.base_dir}")
    output_path = write_patch_dataset_hdf5(
        config.output_path,
        samples,
        img_size=config.img_size,
        overwrite=config.overwrite_outputs,
    )
    logging.info("Packed %s samples into %s", len(samples), output_path)
    return output_path
