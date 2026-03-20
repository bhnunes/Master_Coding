from __future__ import annotations

import logging
from pathlib import Path


def configure_stage_logger(
    logger_name: str,
    log_path: Path,
    *,
    logger_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_level: int = logging.WARNING,
    file_mode: str = "a",
    file_pattern: str = "%(asctime)s - %(process)d - %(levelname)s - %(message)s",
    console_pattern: str = "%(levelname)s: %(message)s",
) -> logging.Logger:
    """Configure a named stage logger with file and console handlers."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode=file_mode)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(file_pattern))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(console_pattern))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def configure_optimization_sampling_logger(log_path: Path) -> logging.Logger:
    """Configure and return the Stage 4 logger."""

    return configure_stage_logger(
        "optimization_sampling",
        log_path,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.WARNING,
        file_mode="a",
    )
