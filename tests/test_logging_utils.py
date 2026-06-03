from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from helpers import logging_utils
from helpers.logging_utils import (
    LoggerSettings,
    LoggerWriter,
    configure_logger,
    configure_root_logger,
    configure_stage_logger,
    resolve_log_folder,
)


def test_resolve_log_folder_prefers_global_log_folder(tmp_path: Path) -> None:
    shared_logs = tmp_path / "shared_logs"
    legacy_logs = tmp_path / "legacy_logs"

    resolved = resolve_log_folder(
        {
            "LOG_FOLDER": str(shared_logs),
            "ARTIFACT_LOG_FOLDER": str(legacy_logs),
        },
        fallback_names=("ARTIFACT_LOG_FOLDER",),
    )

    assert resolved == shared_logs


def test_resolve_log_folder_ignores_empty_global_value_and_uses_fallback(tmp_path: Path) -> None:
    legacy_logs = tmp_path / "legacy_logs"

    resolved = resolve_log_folder(
        {
            "LOG_FOLDER": "",
            "ARTIFACT_LOG_FOLDER": str(legacy_logs),
        },
        fallback_names=("ARTIFACT_LOG_FOLDER",),
    )

    assert resolved == legacy_logs


def test_resolve_log_folder_forwards_expected_args_for_global_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = Mock(return_value=Path("/tmp/shared_logs"))
    monkeypatch.setattr(logging_utils, "resolve_env_path", delegated)

    resolved = resolve_log_folder({"LOG_FOLDER": "/tmp/shared_logs"}, system_name="Linux")

    assert resolved == Path("/tmp/shared_logs")
    delegated.assert_called_once_with(
        "/tmp/shared_logs",
        "LOG_FOLDER",
        system_name="Linux",
        required=True,
    )


def test_resolve_log_folder_forwards_expected_args_for_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = Mock(return_value=Path("logs"))
    monkeypatch.setattr(logging_utils, "resolve_env_path", delegated)

    resolved = resolve_log_folder({}, system_name="Windows")

    assert resolved == Path("logs")
    delegated.assert_called_once_with(
        logging_utils.DEFAULT_LOG_FOLDER,
        "LOG_FOLDER",
        system_name="Windows",
        required=True,
    )


def test_configure_root_logger_writes_to_configured_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "runtime.log"
    logger = configure_root_logger(
        log_path,
        settings=LoggerSettings(console_level=logging.CRITICAL),
    )

    logger.info("hello log")

    assert log_path.is_file()
    assert "hello log" in log_path.read_text(encoding="utf-8")


def test_configure_root_logger_passes_settings_and_disables_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = Mock()
    settings = LoggerSettings(console_level=logging.CRITICAL)

    monkeypatch.setattr(logging_utils, "configure_logger", delegated)
    log_path = tmp_path / "root.log"

    configure_root_logger(log_path, settings=settings)

    delegated.assert_called_once_with(
        log_path,
        logger_name=None,
        settings=settings,
        propagate=False,
    )


def test_configure_root_logger_applies_actual_root_defaults_and_settings(tmp_path: Path) -> None:
    log_path = tmp_path / "root" / "root.log"
    settings = LoggerSettings(
        logger_level=logging.ERROR,
        file_level=logging.WARNING,
        console_level=logging.CRITICAL,
        file_pattern="ROOT:%(levelname)s:%(message)s",
        console_pattern="ROOT_CONSOLE:%(levelname)s:%(message)s",
        console_stream=io.StringIO(),
    )

    logger = configure_root_logger(log_path, settings=settings)
    file_handler, console_handler = logger.handlers

    assert logger is logging.getLogger()
    assert logger.propagate is False
    assert logger.level == logging.ERROR
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.WARNING
    assert file_handler.formatter is not None
    assert file_handler.formatter._fmt == settings.file_pattern
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.CRITICAL
    assert console_handler.formatter is not None
    assert console_handler.formatter._fmt == settings.console_pattern


def test_configure_logger_applies_requested_settings(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "custom.log"
    console_stream = io.StringIO()
    settings = LoggerSettings(
        logger_level=logging.DEBUG,
        file_level=logging.ERROR,
        console_level=logging.WARNING,
        file_mode="w",
        file_pattern="FILE:%(levelname)s:%(message)s",
        console_pattern="CONSOLE:%(levelname)s:%(message)s",
        console_stream=console_stream,
    )

    logger = configure_logger(
        log_path,
        logger_name="custom_logger",
        settings=settings,
        propagate=False,
    )

    assert log_path.parent.is_dir()
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == len(("file", "console"))

    file_handler, console_handler = logger.handlers

    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.ERROR
    assert file_handler.formatter is not None
    assert file_handler.formatter._fmt == settings.file_pattern

    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.WARNING
    assert console_handler.stream is console_stream
    assert console_handler.formatter is not None
    assert console_handler.formatter._fmt == settings.console_pattern


def test_configure_logger_creates_parent_directory_with_expected_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mkdir_calls: list[tuple[Path, bool, bool]] = []
    original_mkdir = Path.mkdir

    def spy_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        mkdir_calls.append((self, parents, exist_ok))
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", spy_mkdir)
    log_path = tmp_path / "deep" / "logger" / "runtime.log"

    configure_logger(log_path, logger_name="mkdir_spy")

    assert mkdir_calls
    assert mkdir_calls[0] == (log_path.parent, True, True)


def test_configure_logger_builds_settings_from_legacy_kwargs(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy" / "legacy.log"
    console_stream = io.StringIO()

    logger = configure_logger(
        log_path,
        logger_name="legacy_logger",
        logger_level=logging.DEBUG,
        file_level=logging.ERROR,
        console_level=logging.WARNING,
        file_mode="w",
        file_pattern="LEGACY_FILE:%(levelname)s:%(message)s",
        console_pattern="LEGACY_CONSOLE:%(levelname)s:%(message)s",
        console_stream=console_stream,
    )

    file_handler, console_handler = logger.handlers

    assert logger.level == logging.DEBUG
    assert logger.propagate is True
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.ERROR
    assert file_handler.formatter is not None
    assert file_handler.formatter._fmt == "LEGACY_FILE:%(levelname)s:%(message)s"
    assert file_handler.mode == "w"
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.WARNING
    assert console_handler.stream is console_stream
    assert console_handler.formatter is not None
    assert console_handler.formatter._fmt == "LEGACY_CONSOLE:%(levelname)s:%(message)s"


def test_configure_logger_defaults_to_stdout_and_root_propagation(tmp_path: Path) -> None:
    log_path = tmp_path / "default" / "default.log"

    logger = configure_logger(log_path)

    file_handler, console_handler = logger.handlers
    assert logger.propagate is False
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.mode == "a"
    assert file_handler.formatter is not None
    assert file_handler.formatter._fmt == "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.stream is sys.stdout
    assert console_handler.formatter is not None
    assert console_handler.formatter._fmt == "%(message)s"


def test_configure_stage_logger_uses_warning_console_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_logger = Mock()

    monkeypatch.setattr(logging_utils, "configure_logger", configure_logger)
    log_path = tmp_path / "stage.log"

    configure_stage_logger("stage_name", log_path)

    configure_logger.assert_called_once_with(
        log_path,
        logger_name="stage_name",
        settings=LoggerSettings(
            console_level=logging.WARNING,
            console_pattern="%(levelname)s: %(message)s",
        ),
        propagate=False,
    )


def test_configure_stage_logger_applies_actual_stage_defaults(tmp_path: Path) -> None:
    log_path = tmp_path / "stage" / "stage.log"

    logger = configure_stage_logger("stage_defaults", log_path)
    file_handler, console_handler = logger.handlers

    assert logger.name == "stage_defaults"
    assert logger.propagate is False
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.INFO
    assert file_handler.formatter is not None
    assert file_handler.formatter._fmt == "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.WARNING
    assert console_handler.formatter is not None
    assert console_handler.formatter._fmt == "%(levelname)s: %(message)s"


def test_logger_writer_forwards_complete_lines_to_logger() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("test_logger_writer_forwards_complete_lines")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    writer = LoggerWriter(logger, logging.INFO)

    assert writer.write("first") == len("first")
    writer.write(" line\nsecond line\n")

    assert stream.getvalue().splitlines() == ["first line", "second line"]


def test_logger_writer_uses_fallback_for_reentrant_stderr_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = io.StringIO()
    logger = logging.getLogger("test_logger_writer_uses_fallback_for_reentrant_stderr_write")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    class RecursiveStderrHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            sys.stderr.write(f"recursive handler saw: {record.getMessage()}\n")

    logger.addHandler(RecursiveStderrHandler())
    writer = LoggerWriter(logger, logging.ERROR, fallback_stream=fallback)
    monkeypatch.setattr(sys, "stderr", writer)

    writer.write("outer error\n")

    assert fallback.getvalue() == "recursive handler saw: outer error\n"


def test_logging_utils_module_namespace_smoke_exercises_primary_functions(
    tmp_path: Path,
) -> None:
    log_folder = logging_utils.resolve_log_folder({})
    assert log_folder == Path("logs")

    log_path = tmp_path / "namespace" / "runtime.log"
    logger = logging_utils.configure_logger(log_path, logger_name="namespace_logger")
    logger.info("namespace logger")

    root_log_path = tmp_path / "namespace" / "root.log"
    root_logger = logging_utils.configure_root_logger(
        root_log_path,
        settings=LoggerSettings(console_level=logging.CRITICAL),
    )
    root_logger.info("root logger")

    stage_log_path = tmp_path / "namespace" / "stage.log"
    stage_logger = logging_utils.configure_stage_logger(
        "namespace_stage",
        stage_log_path,
    )
    stage_logger.warning("stage logger")

    assert "namespace logger" in log_path.read_text(encoding="utf-8")
    assert "root logger" in root_log_path.read_text(encoding="utf-8")
    assert "stage logger" in stage_log_path.read_text(encoding="utf-8")
