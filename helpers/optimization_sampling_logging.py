from __future__ import annotations

import logging
from pathlib import Path


def configure_optimization_sampling_logger(log_path: Path) -> logging.Logger:
    """Configure and return the Stage 4 logger."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("optimization_sampling")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(process)d - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
