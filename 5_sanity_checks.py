from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from helpers.logging_utils import LoggerSettings, configure_root_logger
from helpers.sanity.config import load_sanity_config
from helpers.sanity.pipeline import run_sanity_pipeline
from helpers.sanity.reporting import build_fatal_error_report, render_sanity_report


def main() -> None:
    """Run Stage 6 scientific sanity checks from `.env` configuration."""

    load_dotenv(override=True)
    try:
        config = load_sanity_config(os.environ)
        logger = configure_root_logger(
            config.log_path,
            settings=LoggerSettings(
                logger_level=logging.INFO,
                file_level=logging.INFO,
                console_level=logging.INFO,
                file_mode="a",
                file_pattern="%(asctime)s - %(levelname)s - %(message)s",
                console_pattern="%(message)s",
            ),
        )
        report = run_sanity_pipeline(config)
        rendered_report = render_sanity_report(report)
        logger.info("%s", rendered_report)
        print(rendered_report)
    except Exception as error:
        rendered_error = build_fatal_error_report(error)
        logging.getLogger().error("%s", rendered_error)
        print(rendered_error)
        raise SystemExit(2) from error
    raise SystemExit(0 if report.verdict == "SPLITS PASSED" else 2)


if __name__ == "__main__":
    main()
