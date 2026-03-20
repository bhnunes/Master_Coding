from __future__ import annotations

import logging
from pathlib import Path

from helpers.packaging.config import PackagingConfig
from helpers.packaging.discovery import discover_split_samples
from helpers.packaging.writer import write_split_hdf5


def run_packaging_pipeline(config: PackagingConfig) -> dict[str, Path]:
    logging.info("Starting Stage 7 HDF5 packaging from %s", config.base_dir)
    outputs: dict[str, Path] = {}
    for split_name in config.splits:
        split_dir = config.base_dir / split_name
        if not split_dir.is_dir():
            if split_name.upper() == "TEST":
                logging.info("Skipping optional split without folder: %s", split_dir)
                continue
            raise ValueError(f"Required split folder does not exist: {split_dir}")

        samples = discover_split_samples(config.base_dir, split_name, config.patient_id_regex)
        if not samples:
            raise ValueError(f"No PNG pairs found for required split: {split_name}")
        output_path = config.output_dir / f"{split_name.upper()}.h5"
        outputs[split_name.upper()] = write_split_hdf5(
            output_path,
            samples,
            img_size=config.img_size,
            overwrite=config.overwrite_outputs,
        )
        logging.info("Packed %s samples into %s", len(samples), output_path)

    if not outputs:
        raise ValueError("No HDF5 files were created.")
    return outputs
