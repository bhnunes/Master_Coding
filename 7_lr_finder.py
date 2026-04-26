from __future__ import annotations

import gc
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dotenv import load_dotenv

from helpers.logging_utils import LoggerSettings, configure_root_logger
from helpers.lr_finder.config import load_lr_finder_config
from helpers.lr_finder.pipeline import LRFinderOutputs, run_lr_finder_pipeline
from helpers.training.runtime import seed_everything


def _apply_huggingface_token(hf_token: str | None) -> None:
    if hf_token is None:
        logging.info("Stage 8 Hugging Face token not configured; relying on ambient auth state")
        return
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    logging.info("Stage 8 Hugging Face token detected and applied to environment")


def _print_startup_summary(total_trials: int, output_dir: str) -> None:
    print(f"Stage 7 LR Finder starting: expected_trials={total_trials} output_dir={output_dir}")


def _print_completion_summary(outputs: LRFinderOutputs) -> None:
    print("Stage 7 LR Finder completed successfully")
    print(f"- Valid records: {outputs.valid_records}")
    print(f"- Completed trials: {outputs.completed_trials}")
    print(f"- Failed trials: {outputs.failed_trials}")
    print("- Per-architecture:")
    for architecture, stats in outputs.architecture_trial_stats.items():
        print(
            f"  {architecture}: valid={stats['valid_records']} "
            f"completed={stats['completed_trials']} failed={stats['failed_trials']}"
        )
    print(f"- PDF report: {outputs.pdf_path}")
    print(f"- LaTeX source: {outputs.tex_path}")
    print(f"- Run config JSON: {outputs.run_config_path}")
    print(f"- Summary CSV: {outputs.summary_all_path}")


def main() -> None:
    """Run Stage 7 LR Finder screening and build the LaTeX PDF report."""

    load_dotenv(override=True)
    try:
        config = load_lr_finder_config(os.environ)
        configure_root_logger(
            config.log_path,
            settings=LoggerSettings(
                logger_level=logging.INFO,
                file_level=logging.INFO,
                console_level=logging.ERROR,
                file_mode="a",
                file_pattern="%(asctime)s - %(levelname)s - %(message)s",
                console_pattern="%(message)s",
            ),
        )
        _apply_huggingface_token(config.hf_token)
        total_trials = len(config.model_plans) * config.num_lhs_samples * config.num_repeats
        _print_startup_summary(total_trials, str(config.output_dir))
        logging.info(
            (
                "Stage 7 LR Finder starting: architectures=%s lhs_samples=%s repeats=%s "
                "expected_trials=%s output_dir=%s"
            ),
            len(config.model_plans),
            config.num_lhs_samples,
            config.num_repeats,
            total_trials,
            config.output_dir,
        )
        seed_everything(config.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        outputs = run_lr_finder_pipeline(config)
    except Exception as error:
        print(f"Stage 7 LR Finder failed: {error}")
        raise SystemExit(2) from error

    logging.info("Stage 7 LR Finder completed successfully")
    logging.info("PDF report: %s", outputs.pdf_path)
    logging.info("LaTeX source: %s", outputs.tex_path)
    logging.info("Run config JSON: %s", outputs.run_config_path)
    logging.info("Summary CSV: %s", outputs.summary_all_path)
    _print_completion_summary(outputs)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
