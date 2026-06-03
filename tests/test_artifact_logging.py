from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from helpers.artifact.logging import Style, banner, configure_artifact_logger


def test_configure_artifact_logger_passes_expected_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_logger = Mock()

    import helpers.artifact.logging as artifact_logging

    monkeypatch.setattr(artifact_logging, "configure_logger", configure_logger)
    log_path = tmp_path / "artifact.log"

    configure_artifact_logger(log_path)

    configure_logger.assert_called_once_with(
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


def test_banner_wraps_title_with_expected_style_codes() -> None:
    assert banner("Stage 1") == f"\n{Style.BLUE}{Style.BOLD}--- Stage 1 ---{Style.RESET}"
