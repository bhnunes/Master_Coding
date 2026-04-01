from __future__ import annotations

import gc
import logging
import os

import torch
from dotenv import load_dotenv

from helpers.logging_utils import configure_root_logger
from helpers.smart_sampling.config import load_smart_sampler_config
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.training.runtime import seed_everything


def main() -> None:
    """Filter TRAIN.h5 into TRAIN_FILTERED.h5 using patient-wise smart sampling."""

    load_dotenv(override=True)

    try:
        config = load_smart_sampler_config(os.environ)
        configure_root_logger(
            config.log_path,
            logger_level=logging.INFO,
            file_level=logging.INFO,
            console_level=logging.INFO,
            file_mode="a",
            file_pattern="%(asctime)s - %(levelname)s - %(message)s",
            console_pattern="%(message)s",
        )
        seed_everything(config.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        outputs = run_smart_sampling_pipeline(config)
    except Exception as error:
        print(f"Stage 8 smart sampling failed: {error}")
        raise SystemExit(2) from error

    logging.info("Stage 8 smart sampling completed successfully")
    print("Stage 8 smart sampling completed successfully:")
    logging.info("Filtered HDF5: %s", outputs.filtered_h5_path)
    print(f"- Filtered HDF5: {outputs.filtered_h5_path}")
    if outputs.selection_csv_path is not None:
        logging.info("Selection CSV: %s", outputs.selection_csv_path)
        print(f"- Selection CSV: {outputs.selection_csv_path}")
    if outputs.stats_csv_path is not None:
        logging.info("Stats CSV: %s", outputs.stats_csv_path)
        print(f"- Stats CSV: {outputs.stats_csv_path}")
    if outputs.run_config_path is not None:
        logging.info("Run config JSON: %s", outputs.run_config_path)
        print(f"- Run config JSON: {outputs.run_config_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
