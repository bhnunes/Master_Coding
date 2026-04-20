from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from helpers.ensemble_optimizer.config import load_ensemble_optimizer_config
from helpers.ensemble_optimizer.pipeline import run_ensemble_optimizer_pipeline
from helpers.logging_utils import LoggerWriter, configure_root_logger


def main() -> None:
    load_dotenv(override=True)
    config = load_ensemble_optimizer_config()
    logger = configure_root_logger(
        config.log_path,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.INFO,
        file_mode="a",
        file_pattern="%(asctime)s - %(levelname)s - %(message)s",
        console_pattern="%(message)s",
    )
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    outputs = run_ensemble_optimizer_pipeline(config)
    print(f"Ensemble recipe saved to: {outputs.recipe_path}")
    print(f"Run config saved to: {outputs.run_config_path}")


if __name__ == "__main__":
    main()
