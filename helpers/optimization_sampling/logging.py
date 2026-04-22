from __future__ import annotations

import logging
from pathlib import Path

from helpers.logging_utils import LoggerSettings, configure_stage_logger

__all__ = ["configure_stage_logger", "configure_optimization_sampling_logger"]


def configure_optimization_sampling_logger(log_path: Path) -> logging.Logger:
    """Configure and return the Stage 3.1 logger."""

    return configure_stage_logger(
        "optimization_sampling",
        log_path,
        settings=LoggerSettings(
            logger_level=logging.INFO,
            file_level=logging.INFO,
            console_level=logging.WARNING,
            file_mode="a",
        ),
    )
