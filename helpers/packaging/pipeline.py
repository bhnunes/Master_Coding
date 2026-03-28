from __future__ import annotations

import logging
from pathlib import Path

from helpers.packaging.config import PackagingConfig
from helpers.packaging.writer import (
    copy_source_hdf5_dataset,
    filter_source_hdf5_by_manifest,
    merge_source_hdf5_shards,
)


def run_packaging_pipeline(config: PackagingConfig) -> Path:
    if config.accepted_manifest_path is not None:
        logging.info(
            "Starting Stage 7 HDF5 finalization from %s using accepted manifest %s",
            config.source_hdf5_path,
            config.accepted_manifest_path,
        )
        output_path = filter_source_hdf5_by_manifest(
            config.source_hdf5_path,
            config.accepted_manifest_path,
            config.output_path,
            overwrite=config.overwrite_outputs,
        )
        logging.info("Filtered canonical HDF5 input into %s", output_path)
        return output_path

    if config.source_hdf5_path.is_dir():
        logging.info("Starting Stage 7 HDF5 merge from shard directory %s", config.source_hdf5_path)
        output_path = merge_source_hdf5_shards(
            config.source_hdf5_path,
            config.output_path,
            overwrite=config.overwrite_outputs,
        )
        logging.info("Merged Stage 2 HDF5 shards into %s", output_path)
        return output_path

    logging.info("Starting Stage 7 HDF5 finalization from %s", config.source_hdf5_path)
    output_path = copy_source_hdf5_dataset(
        config.source_hdf5_path,
        config.output_path,
        overwrite=config.overwrite_outputs,
    )
    logging.info("Validated and copied canonical HDF5 input into %s", output_path)
    return output_path
