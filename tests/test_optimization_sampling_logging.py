from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from helpers.optimization_sampling.logging import (
    configure_optimization_sampling_logger,
    configure_stage_logger,
)


def test_configure_stage_logger_passes_through_custom_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_logger = Mock()

    import helpers.logging_utils as logging_utils

    monkeypatch.setattr(logging_utils, "configure_logger", configure_logger)
    log_path = tmp_path / "stage.log"

    configure_stage_logger(
        "stage_name",
        log_path,
        logger_level=logging.DEBUG,
        file_level=logging.WARNING,
        console_level=logging.ERROR,
        file_mode="w",
        file_pattern="file-pattern",
        console_pattern="console-pattern",
    )

    configure_logger.assert_called_once_with(
        log_path,
        logger_name="stage_name",
        logger_level=logging.DEBUG,
        file_level=logging.WARNING,
        console_level=logging.ERROR,
        file_mode="w",
        file_pattern="file-pattern",
        console_pattern="console-pattern",
        propagate=False,
    )


def test_configure_optimization_sampling_logger_uses_stage_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_stage = Mock()

    import helpers.optimization_sampling.logging as sampling_logging

    monkeypatch.setattr(sampling_logging, "configure_stage_logger", configure_stage)
    log_path = tmp_path / "optimization.log"

    configure_optimization_sampling_logger(log_path)

    configure_stage.assert_called_once_with(
        "optimization_sampling",
        log_path,
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.WARNING,
        file_mode="a",
    )
