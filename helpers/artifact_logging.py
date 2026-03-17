from __future__ import annotations

import logging
from pathlib import Path


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


def configure_artifact_logger(log_folder: Path) -> logging.Logger:
    """Create the artifact detection logger with console and file handlers."""

    log_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("artifact_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_folder / "artifact_detection.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def banner(title: str) -> str:
    """Return a styled banner title."""

    return f"\n{Style.BLUE}{Style.BOLD}--- {title} ---{Style.RESET}"
