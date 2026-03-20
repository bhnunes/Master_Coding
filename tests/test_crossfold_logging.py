import logging
from pathlib import Path

from helpers.crossfold.logging import configure_crossfold_logging


def test_configure_crossfold_logging_creates_log_file(tmp_path: Path) -> None:
    configure_crossfold_logging(tmp_path)
    logging.info("hello")

    log_path = tmp_path / "data_preparation.log"
    assert log_path.is_file()
    assert "hello" in log_path.read_text(encoding="utf-8")
