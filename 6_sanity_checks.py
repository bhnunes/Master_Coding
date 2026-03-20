from __future__ import annotations

import os

from dotenv import load_dotenv

from helpers.sanity.config import load_sanity_config
from helpers.sanity.pipeline import run_sanity_pipeline
from helpers.sanity.reporting import build_fatal_error_report, render_sanity_report


def main() -> None:
    """Run Stage 6 scientific sanity checks from `.env` configuration."""

    load_dotenv(override=True)
    try:
        config = load_sanity_config(os.environ)
        report = run_sanity_pipeline(config)
        print(render_sanity_report(report))
    except Exception as error:
        print(build_fatal_error_report(error))
        raise SystemExit(2) from error
    raise SystemExit(0 if report.verdict == "SPLITS PASSED" else 2)


if __name__ == "__main__":
    main()
