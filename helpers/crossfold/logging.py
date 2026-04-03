from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from helpers.logging_utils import configure_root_logger

_PROGRESS_LOG_INTERVAL_SECONDS = 1.5


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


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ProgressReporter:
    phase_name: str
    total_units: int
    unit_label: str
    context: str = ""
    start_time: float = 0.0
    last_logged_at: float = 0.0

    def __post_init__(self) -> None:
        self.start_time = perf_counter()
        self.last_logged_at = self.start_time

    def log_start(self, summary: str) -> None:
        logging.info("%s: %s", self.phase_name, summary)

    def log(
        self,
        *,
        completed_units: int,
        force: bool = False,
        extra_parts: list[str] | None = None,
    ) -> None:
        now = perf_counter()
        if not force and completed_units < self.total_units:
            if now - self.last_logged_at < _PROGRESS_LOG_INTERVAL_SECONDS:
                return

        elapsed = max(now - self.start_time, 1e-9)
        rate = completed_units / elapsed if completed_units > 0 else 0.0
        remaining_units = max(self.total_units - completed_units, 0)
        eta_seconds = remaining_units / rate if rate > 0 else 0.0
        percent = (completed_units / self.total_units) * 100.0 if self.total_units else 100.0
        parts = [
            f"{completed_units}/{self.total_units} {self.unit_label}",
            f"{percent:.1f}%",
            f"{rate:.1f} {self.unit_label}/s",
            f"ETA {format_duration(eta_seconds)}",
        ]
        if self.context:
            parts.insert(0, self.context)
        if extra_parts:
            parts.extend(extra_parts)
        logging.info("%s: %s", self.phase_name, " | ".join(parts))
        self.last_logged_at = now
