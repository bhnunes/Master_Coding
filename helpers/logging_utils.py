from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from helpers.runtime_platform import resolve_env_path

DEFAULT_LOG_FOLDER = "./logs"


def resolve_log_folder(
    environment: Mapping[str, str | None],
    *,
    system_name: str | None = None,
    fallback_names: Sequence[str] = (),
) -> Path:
    """Resolve the shared runtime log folder, with compatibility fallbacks."""

    for variable_name in ("LOG_FOLDER", *fallback_names):
        value = environment.get(variable_name)
        if value not in {None, ""}:
            path = resolve_env_path(
                value,
                variable_name,
                system_name=system_name,
                required=True,
            )
            assert path is not None
            return path

    default_path = resolve_env_path(
        DEFAULT_LOG_FOLDER,
        "LOG_FOLDER",
        system_name=system_name,
        required=True,
    )
    assert default_path is not None
    return default_path


def configure_logger(
    log_path: Path,
    *,
    logger_name: str | None = None,
    logger_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_mode: str = "a",
    file_pattern: str = "%(asctime)s - %(process)d - %(levelname)s - %(message)s",
    console_pattern: str = "%(message)s",
    console_stream: TextIO | None = None,
    propagate: bool | None = None,
) -> logging.Logger:
    """Configure a file+console logger for one stage."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.handlers.clear()
    if propagate is None:
        logger.propagate = logger_name is not None
    else:
        logger.propagate = propagate

    file_handler = logging.FileHandler(log_path, mode=file_mode)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(file_pattern))

    stream: TextIO = sys.stdout if console_stream is None else console_stream
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(console_pattern))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def configure_root_logger(
    log_path: Path,
    *,
    logger_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_mode: str = "a",
    file_pattern: str = "%(asctime)s - %(process)d - %(levelname)s - %(message)s",
    console_pattern: str = "%(message)s",
    console_stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the root logger with file and console handlers."""

    return configure_logger(
        log_path,
        logger_name=None,
        logger_level=logger_level,
        file_level=file_level,
        console_level=console_level,
        file_mode=file_mode,
        file_pattern=file_pattern,
        console_pattern=console_pattern,
        console_stream=console_stream,
        propagate=False,
    )


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


class LoggerWriter:
    """File-like adapter that forwards writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._logger.log(self._level, stripped)
        return len(message)

    def flush(self) -> None:
        stripped = self._buffer.rstrip()
        if stripped:
            self._logger.log(self._level, stripped)
        self._buffer = ""

    def isatty(self) -> bool:
        return False
