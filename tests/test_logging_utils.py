from __future__ import annotations

import logging
from pathlib import Path

from helpers.logging_utils import LoggerSettings, configure_root_logger, resolve_log_folder


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


def test_configure_root_logger_writes_to_configured_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "runtime.log"
    logger = configure_root_logger(
        log_path,
        settings=LoggerSettings(console_level=logging.CRITICAL),
    )

    logger.info("hello log")

    assert log_path.is_file()
    assert "hello log" in log_path.read_text(encoding="utf-8")
