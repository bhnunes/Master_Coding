from __future__ import annotations

import logging
from pathlib import Path

from helpers.logging_utils import configure_root_logger


def configure_crossfold_logging(log_path: Path) -> logging.Logger:
    """Configure Stage 5 file and console logging."""

    logger = configure_root_logger(
        log_path,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.INFO,
        file_mode="a",
        file_pattern="%(asctime)s - %(levelname)s - %(message)s",
        console_pattern="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Logging configured: %s", log_path)
    return logger
