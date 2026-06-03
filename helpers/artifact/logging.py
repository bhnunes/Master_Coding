from __future__ import annotations

import logging
from pathlib import Path

from helpers.logging_utils import configure_logger


class Style:
    """ANSI styling and status icons for friendly terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    ROCKET = "🚀"
    DB = "📦"
    FOLDER = "📁"
    IMAGE = "🖼️"
    ZIP = "🗜️"


def configure_artifact_logger(log_path: Path) -> logging.Logger:
    """Create the artifact detection logger with console and file handlers."""

    return configure_logger(
        log_path,
        logger_name="artifact_detection",
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.INFO,
        file_mode="a",
        file_pattern="%(asctime)s | %(levelname)s | %(message)s",
        console_pattern="%(message)s",
        propagate=False,
    )


def banner(title: str) -> str:
    """Return a styled banner title."""

    return f"\n{Style.BLUE}{Style.BOLD}--- {title} ---{Style.RESET}"
