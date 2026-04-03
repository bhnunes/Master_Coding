from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from helpers.logging_utils import configure_root_logger
from helpers.packaging.config import load_packaging_config
from helpers.packaging.pipeline import run_packaging_pipeline


def main() -> None:
    """Pack the cleaned post-Stage-4.3 patch pool into one source HDF5 dataset."""

    load_dotenv(override=True)

    try:
        config = load_packaging_config(os.environ)
        configure_root_logger(
            config.log_path,
            logger_level=logging.INFO,
            file_level=logging.INFO,
            console_level=logging.INFO,
            file_mode="a",
            file_pattern="%(asctime)s - %(levelname)s - %(message)s",
            console_pattern="%(message)s",
        )
        output_path = run_packaging_pipeline(config)
    except Exception as error:
        print(f"Stage 3 packaging failed: {error}")
        raise SystemExit(2) from error

    logging.info("Stage 3 packaging completed successfully")
    logging.info("SOURCE_DATASET: %s", output_path)
    print("Stage 3 packaging completed successfully:")
    print(f"- SOURCE_DATASET: {output_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
