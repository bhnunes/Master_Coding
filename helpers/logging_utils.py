from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from helpers.runtime_platform import resolve_env_path

DEFAULT_LOG_FOLDER = "./logs"


@dataclass(frozen=True)
class LoggerSettings:
    logger_level: int = logging.INFO
    file_level: int = logging.INFO
    console_level: int = logging.INFO
    file_mode: str = "a"
    file_pattern: str = "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
    console_pattern: str = "%(message)s"
    console_stream: TextIO | None = None


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
    settings: LoggerSettings | None = None,
    propagate: bool | None = None,
    **legacy_kwargs: object,
) -> logging.Logger:
    """Configure a file+console logger for one stage."""

    if settings is None:
        active_settings = LoggerSettings(
            logger_level=cast(int | None, legacy_kwargs.get("logger_level")) or logging.INFO,
            file_level=cast(int | None, legacy_kwargs.get("file_level")) or logging.INFO,
            console_level=(cast(int | None, legacy_kwargs.get("console_level")) or logging.INFO),
            file_mode=cast(str | None, legacy_kwargs.get("file_mode")) or "a",
            file_pattern=(
                "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
                if legacy_kwargs.get("file_pattern") is None
                else cast(str, legacy_kwargs["file_pattern"])
            ),
            console_pattern=(
                "%(message)s"
                if legacy_kwargs.get("console_pattern") is None
                else cast(str, legacy_kwargs["console_pattern"])
            ),
            console_stream=cast(TextIO | None, legacy_kwargs.get("console_stream")),
        )
    else:
        active_settings = settings
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(active_settings.logger_level)
    logger.handlers.clear()
    if propagate is None:
        logger.propagate = logger_name is not None
    else:
        logger.propagate = propagate

    file_handler = logging.FileHandler(log_path, mode=active_settings.file_mode)
    file_handler.setLevel(active_settings.file_level)
    file_handler.setFormatter(logging.Formatter(active_settings.file_pattern))

    stream: TextIO = (
        sys.stdout if active_settings.console_stream is None else active_settings.console_stream
    )
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(active_settings.console_level)
    console_handler.setFormatter(logging.Formatter(active_settings.console_pattern))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def configure_root_logger(
    log_path: Path,
    *,
    settings: LoggerSettings | None = None,
) -> logging.Logger:
    """Configure the root logger with file and console handlers."""

    return configure_logger(
        log_path,
        logger_name=None,
        settings=settings,
        propagate=False,
    )


def configure_stage_logger(
    logger_name: str,
    log_path: Path,
    *,
    settings: LoggerSettings | None = None,
) -> logging.Logger:
    """Configure a named stage logger with file and console handlers."""

    default_settings = LoggerSettings(
        console_level=logging.WARNING,
        console_pattern="%(levelname)s: %(message)s",
    )
    return configure_logger(
        log_path,
        logger_name=logger_name,
        settings=settings or default_settings,
        propagate=False,
    )


class LoggerWriter:
    """File-like adapter that forwards writes to a logger."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        fallback_stream: TextIO | None = None,
    ) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""
        if fallback_stream is not None:
            self._fallback_stream = fallback_stream
        elif level >= logging.ERROR:
            self._fallback_stream = cast(TextIO, sys.__stderr__)
        else:
            self._fallback_stream = cast(TextIO, sys.__stdout__)
        self._writing = False

    def _write_fallback(self, message: str) -> None:
        try:
            self._fallback_stream.write(message)
            self._fallback_stream.flush()
        except Exception:
            pass

    def _log_line(self, line: str) -> None:
        if self._writing:
            self._write_fallback(f"{line}\n")
            return

        try:
            self._writing = True
            self._logger.log(self._level, line)
        except Exception:
            self._write_fallback(f"{line}\n")
        finally:
            self._writing = False

    def write(self, message: str) -> int:
        if self._writing:
            self._write_fallback(message)
            return len(message)

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._log_line(stripped)
        return len(message)

    def flush(self) -> None:
        if self._writing:
            return

        stripped = self._buffer.rstrip()
        if stripped:
            self._log_line(stripped)
        self._buffer = ""

    def isatty(self) -> bool:
        return False
