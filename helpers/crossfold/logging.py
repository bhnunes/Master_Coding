from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_crossfold_logging(output_dir: Path) -> None:
    """Configure Stage 5 file and console logging for one run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "data_preparation.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info("Logging configured: %s", log_file)
