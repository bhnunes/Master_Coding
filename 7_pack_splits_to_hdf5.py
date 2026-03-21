from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from helpers.logging_utils import configure_root_logger
from helpers.packaging.config import load_packaging_config
from helpers.packaging.pipeline import run_packaging_pipeline


def main() -> None:
    """Pack Stage 5 split folders into the minimal HDF5 contract used by training."""

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
        outputs = run_packaging_pipeline(config)
    except Exception as error:
        print(f"Stage 7 packaging failed: {error}")
        raise SystemExit(2) from error

    logging.info("Stage 7 packaging completed successfully")
    print("Stage 7 packaging completed successfully:")
    for split_name, path in outputs.items():
        logging.info("%s: %s", split_name, path)
        print(f"- {split_name}: {path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
