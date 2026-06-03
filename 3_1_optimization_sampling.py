from __future__ import annotations

import os

from dotenv import load_dotenv

from helpers.optimization_sampling.config import load_optimization_sampling_config
from helpers.optimization_sampling.logging import configure_optimization_sampling_logger
from helpers.optimization_sampling.pipeline import (
    OptimizationSamplingConfig,
    build_next_steps_message,
    run_optimization_sampling,
)


def main() -> None:
    """Run Stage 3.1 optimization sampling from `.env` configuration."""

    load_dotenv(override=True)
    config = load_optimization_sampling_config(os.environ)
    logger = configure_optimization_sampling_logger(config.log_path)
    summary = run_optimization_sampling(
        OptimizationSamplingConfig(
            master_manifest_path=config.master_manifest_path,
            output_base=config.output_base,
            confidence_level=config.confidence_level,
            margin_of_error=config.margin_of_error,
            proportion=config.proportion,
            pilot_sample_size=config.pilot_sample_size,
            master_pool_fraction=config.master_pool_fraction,
            overlay_color=config.overlay_color,
            overlay_thickness=config.overlay_thickness,
            overlay_alpha=config.overlay_alpha,
            seed=config.seed,
            num_processes=config.num_processes,
            logger=logger,
        )
    )
    print(build_next_steps_message(summary))


if __name__ == "__main__":
    main()
