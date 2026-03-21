from __future__ import annotations

import logging
from pathlib import Path

from helpers.logging_utils import configure_logger


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

    return configure_logger(
        log_path,
        logger_name=logger_name,
        logger_level=logger_level,
        file_level=file_level,
        console_level=console_level,
        file_mode=file_mode,
        file_pattern=file_pattern,
        console_pattern=console_pattern,
        propagate=False,
    )


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
