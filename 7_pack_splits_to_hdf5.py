from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from helpers.packaging.config import load_packaging_config
from helpers.packaging.pipeline import run_packaging_pipeline


def main() -> None:
    """Pack Stage 5 split folders into the minimal HDF5 contract used by training."""

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        config = load_packaging_config(os.environ)
        outputs = run_packaging_pipeline(config)
    except Exception as error:
        print(f"Stage 7 packaging failed: {error}")
        raise SystemExit(2) from error

    print("Stage 7 packaging completed successfully:")
    for split_name, path in outputs.items():
        print(f"- {split_name}: {path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
