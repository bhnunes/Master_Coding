from __future__ import annotations

import os

from dotenv import load_dotenv

from helpers.graph_tuning_config import load_graph_tuning_config
from helpers.graph_tuning_pipeline import build_recommendation_message, run_graph_tuning_pipeline
from helpers.optimization_sampling_logging import configure_stage_logger


def main() -> None:
    """Run Stage 4.2 graph tuning from `.env` configuration."""

    load_dotenv(override=True)
    config = load_graph_tuning_config(os.environ)
    logger = configure_stage_logger(
        "graph_tuning",
        config.log_path,
        logger_level=20,
        file_level=10,
        console_level=20,
        file_mode="w",
        file_pattern="%(asctime)s - %(levelname)s - %(message)s",
    )
    summary = run_graph_tuning_pipeline(
        source_image_folder=config.source_image_folder,
        source_mask_folder=config.source_mask_folder,
        review_base_dir=config.review_base_dir,
        test_set_size=config.test_set_size,
        n_splits_inner_cv=config.n_splits_inner_cv,
        n_bayesian_calls=config.n_bayesian_calls,
        n_initial_points=config.n_initial_points,
        random_state=config.random_state,
        bg_intensity_range=config.bg_intensity_range,
        k_range=config.k_range,
        min_size_range=config.min_size_range,
        erosion_range=config.erosion_range,
        logger=logger,
    )
    print(build_recommendation_message(summary))


if __name__ == "__main__":
    main()
