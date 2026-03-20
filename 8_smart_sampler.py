from __future__ import annotations

import gc
import logging
import os

import torch
from dotenv import load_dotenv

from helpers.smart_sampling.config import load_smart_sampler_config
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.training.runtime import seed_everything


def main() -> None:
    """Filter TRAIN.h5 into TRAIN_FILTERED.h5 using patient-wise smart sampling."""

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        config = load_smart_sampler_config(os.environ)
        seed_everything(config.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        outputs = run_smart_sampling_pipeline(config)
    except Exception as error:
        print(f"Stage 8 smart sampling failed: {error}")
        raise SystemExit(2) from error

    print("Stage 8 smart sampling completed successfully:")
    print(f"- Filtered HDF5: {outputs.filtered_h5_path}")
    if outputs.selection_csv_path is not None:
        print(f"- Selection CSV: {outputs.selection_csv_path}")
    if outputs.stats_csv_path is not None:
        print(f"- Stats CSV: {outputs.stats_csv_path}")
    if outputs.run_config_path is not None:
        print(f"- Run config JSON: {outputs.run_config_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
