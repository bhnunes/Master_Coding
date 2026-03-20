from __future__ import annotations

import gc
import logging
import os

import torch
from dotenv import load_dotenv

from helpers.lr_finder.config import load_lr_finder_config
from helpers.lr_finder.pipeline import run_lr_finder_pipeline
from helpers.training.runtime import seed_everything


def main() -> None:
    """Run Stage 9 LR Finder screening and build the LaTeX PDF report."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        config = load_lr_finder_config(os.environ)
        seed_everything(config.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        outputs = run_lr_finder_pipeline(config)
    except Exception as error:
        print(f"Stage 9 LR Finder failed: {error}")
        raise SystemExit(2) from error

    print("Stage 9 LR Finder completed successfully:")
    print(f"- PDF report: {outputs.pdf_path}")
    print(f"- LaTeX source: {outputs.tex_path}")
    print(f"- Run config JSON: {outputs.run_config_path}")
    print(f"- Summary CSV: {outputs.summary_all_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
