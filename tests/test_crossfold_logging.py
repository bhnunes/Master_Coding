import logging
from pathlib import Path

from helpers.crossfold.logging import configure_crossfold_logging


def test_configure_crossfold_logging_creates_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "data_preparation.log"
    configure_crossfold_logging(log_path)
    logging.info("hello")

    assert log_path.is_file()
    assert "hello" in log_path.read_text(encoding="utf-8")
