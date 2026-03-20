from __future__ import annotations

import os

from dotenv import load_dotenv

from helpers.crossfold.config import load_crossfold_config
from helpers.crossfold.pipeline import run_crossfold_pipeline


def main() -> None:
    """Run Stage 5 patient-level dataset splitting from `.env` configuration."""

    load_dotenv(override=True)
    config = load_crossfold_config(os.environ)
    run_crossfold_pipeline(config)


if __name__ == "__main__":
    main()
