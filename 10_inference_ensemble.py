from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from helpers.ensemble_inference.config import load_ensemble_inference_config
from helpers.ensemble_inference.pipeline import run_ensemble_inference_pipeline
from helpers.logging_utils import LoggerSettings, LoggerWriter, configure_root_logger
from helpers.runtime_normalization import format_runtime_normalization_status


def main() -> None:
    load_dotenv(override=True)
    config = load_ensemble_inference_config()
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
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    print(
        format_runtime_normalization_status(
            runtime_normalization_method=config.runtime_normalization_method,
            runtime_vahadane_backend=config.runtime_vahadane_backend,
        )
    )
    outputs = run_ensemble_inference_pipeline(config)
    print(f"Inference output directory: {outputs.output_dir}")
    print(f"Run config saved to: {outputs.run_config_path}")
    print(f"Metrics JSON saved to: {outputs.metrics_json_path}")
    print(f"Confusion matrix saved to: {outputs.confusion_matrix_path}")
    print(f"Markdown report saved to: {outputs.markdown_report_path}")
    if outputs.csv_report_path is not None:
        print(f"CSV report saved to: {outputs.csv_report_path}")
    if outputs.pdf_report_path is not None:
        print(f"PDF report saved to: {outputs.pdf_report_path}")


if __name__ == "__main__":
    main()
