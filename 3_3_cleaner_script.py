from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from tqdm import tqdm

from helpers.graph.cleaning_config import load_graph_cleaning_config
from helpers.graph.cleaning_pipeline import (
    GraphCleaningPipelineConfig,
    build_cleaning_message,
    run_graph_cleaning_pipeline,
)
from helpers.logging_utils import LoggerSettings, configure_stage_logger


def main() -> None:
    """Run Stage 3.3 graph-based cleaning from `.env` configuration."""

    load_dotenv(override=True)
    config = load_graph_cleaning_config(os.environ)
    logger = configure_stage_logger(
        "graph_cleaning",
        config.log_path,
        settings=LoggerSettings(
            logger_level=logging.DEBUG,
            file_level=logging.DEBUG,
            console_level=logging.INFO,
            file_mode="w",
            file_pattern="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        ),
    )
    summary = run_graph_cleaning_pipeline(
        GraphCleaningPipelineConfig(
            master_manifest_path=config.master_manifest_path,
            output_base_dir=config.output_base_dir,
            graph_params=config.graph_params,
            tau=config.tau,
            num_workers=config.num_workers,
            logger=logger,
            progress_factory=lambda iterable: tqdm(
                iterable,
                total=_resolve_progress_total(iterable),
                desc="Filtering Images",
            ),
        )
    )
    logger.info("Loaded graph cleaning parameters from: %s", config.params_path)
    print(build_cleaning_message(summary))


def _resolve_progress_total(iterable: object) -> int | None:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


if __name__ == "__main__":
    main()
