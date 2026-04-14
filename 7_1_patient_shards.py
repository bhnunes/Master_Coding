from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from helpers.logging_utils import configure_root_logger
from helpers.patient_shards.config import load_patient_shards_config
from helpers.patient_shards.pipeline import run_patient_shards_pipeline


def main() -> None:
    """Run Stage 6.5 patient-isolated shard generation from `.env` configuration."""

    load_dotenv(override=True)
    config = load_patient_shards_config(os.environ)
    logger = configure_root_logger(
        config.log_path,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.INFO,
        file_mode="a",
        file_pattern="%(asctime)s - %(levelname)s - %(message)s",
        console_pattern="%(message)s",
    )
    summary = run_patient_shards_pipeline(config)
    logger.info(
        "Stage 6.5 summary: total_shards=%s | split_shard_counts=%s | output=%s",
        summary.total_shards,
        summary.split_shard_counts,
        summary.output_base_dir,
    )


if __name__ == "__main__":
    main()
