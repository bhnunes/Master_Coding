from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from helpers.optimization_sampling.overlay import generate_overlay_images
from helpers.optimization_sampling.sampling import (
    OverlayTask,
    build_overlay_tasks,
    discover_manifest_image_mask_pairs,
    ensure_manual_labeling_directories,
    select_sample_stems,
)

OverlayRunner = Callable[[Sequence[OverlayTask], int], list[bool]]


@dataclass(frozen=True)
class OptimizationSamplingSummary:
    """Execution summary for Stage 3.1 optimization sampling."""

    total_population: int
    required_sample_size: int
    pilot_sample_size: int
    master_pool_size: int
    requested_overlay_count: int
    generated_overlay_count: int
    failed_overlay_count: int
    pilot_folder: Path
    master_pool_folder: Path


@dataclass(frozen=True)
class OptimizationSamplingConfig:
    master_manifest_path: Path
    output_base: Path
    confidence_level: float
    margin_of_error: float
    proportion: float
    pilot_sample_size: int
    master_pool_fraction: float
    overlay_color: tuple[int, int, int]
    overlay_thickness: int
    overlay_alpha: float
    num_processes: int
    seed: int
    logger: logging.Logger | None = None
    rng: random.Random | None = None
    overlay_runner: OverlayRunner = generate_overlay_images


def run_optimization_sampling(config: OptimizationSamplingConfig) -> OptimizationSamplingSummary:
    """Run the current Stage 3.1 optimization sampling workflow."""

    active_logger = config.logger or logging.getLogger("optimization_sampling")
    active_logger.info("--- Experiment Setup Initiated: On-the-Fly Generation ---")
    active_logger.info(
        "Resolving Stage 3.1 candidates from Stage 2 master manifest: %s",
        config.master_manifest_path,
    )
    active_rng = config.rng or random.Random(config.seed)
    pairs = discover_manifest_image_mask_pairs(config.master_manifest_path)
    selection = select_sample_stems(
        list(pairs),
        pilot_sample_size=config.pilot_sample_size,
        master_pool_fraction=config.master_pool_fraction,
        confidence_level=config.confidence_level,
        margin_of_error=config.margin_of_error,
        proportion=config.proportion,
        rng=active_rng,
    )
    active_logger.info(
        "Identified %s canonical cancer image-mask pairs for image-level sampling.",
        selection.total_population,
    )
    active_logger.info("--- Experiment Parameters ---")
    active_logger.info(
        "Statistically Required Sample Size (%.0f%% confidence, %.0f%% error): %s",
        config.confidence_level * 100,
        config.margin_of_error * 100,
        selection.required_sample_size,
    )
    active_logger.info(
        "Pilot Sample Size (for estimating proportions): %s", selection.pilot_sample_size
    )
    active_logger.info(
        "Master Candidate Pool Size (%.0f%% of total): %s",
        config.master_pool_fraction * 100,
        selection.master_pool_size,
    )
    active_logger.info(
        "Randomly selected %s non-overlapping images for the master pool.",
        len(selection.master_pool_stems),
    )
    active_logger.info(
        "Randomly selected %s non-overlapping images for the pilot sample.",
        len(selection.pilot_sample_stems),
    )

    tasks = build_overlay_tasks(
        pairs=pairs,
        output_base=config.output_base,
        master_pool_stems=selection.master_pool_stems,
        pilot_sample_stems=selection.pilot_sample_stems,
        color=config.overlay_color,
        thickness=config.overlay_thickness,
        alpha=config.overlay_alpha,
    )
    active_logger.info(
        "Starting on-the-fly overlay generation for %s selected images using %s processes.",
        len(tasks),
        config.num_processes,
    )
    results = config.overlay_runner(tasks, config.num_processes)
    success_count = sum(results)
    active_logger.info(
        "Successfully generated %s out of %s required overlay images.",
        success_count,
        len(tasks),
    )
    if success_count < len(tasks):
        active_logger.warning(
            "%s images failed to process. Check the Stage 3.1 log for details.",
            len(tasks) - success_count,
        )

    pilot_folder, master_pool_folder = ensure_manual_labeling_directories(config.output_base)
    active_logger.info("Created subdirectories for manual labeling inside '%s'", pilot_folder)
    active_logger.info("Created subdirectories for manual labeling inside '%s'", master_pool_folder)

    return OptimizationSamplingSummary(
        total_population=selection.total_population,
        required_sample_size=selection.required_sample_size,
        pilot_sample_size=selection.pilot_sample_size,
        master_pool_size=selection.master_pool_size,
        requested_overlay_count=len(tasks),
        generated_overlay_count=success_count,
        failed_overlay_count=len(tasks) - success_count,
        pilot_folder=pilot_folder,
        master_pool_folder=master_pool_folder,
    )


def build_next_steps_message(summary: OptimizationSamplingSummary) -> str:
    """Build the operator instructions shown after Stage 3.1 setup completes."""

    approved_target = math.ceil(0.7 * summary.required_sample_size)
    rejected_target = math.ceil(0.3 * summary.required_sample_size)
    return (
        "\n\n--- Setup Complete! (Efficient Method) ---\n"
        "Overlays for the pilot and master pools were generated on-the-fly.\n"
        "NO intermediate 'overlays' folder was created, saving significant disk space.\n"
        "\nA detailed log has been saved to the Stage 3.1 log file.\n"
        "\nYour next steps are:\n"
        f"1. Go to the '{summary.pilot_folder.name}' folder. Move the "
        f"{summary.pilot_sample_size} generated overlays into the 'APPROVED' "
        "or 'REJECTED' subfolders based on your criteria.\n"
        "2. Once sorted, calculate the proportion of each class (e.g., 70 Approved -> 70%).\n"
        f"3. Use this proportion to calculate your final target numbers from the "
        f"{summary.required_sample_size} total samples needed.\n"
        f"   - Example: 0.70 * {summary.required_sample_size} = {approved_target} "
        "'Approved' images.\n"
        f"   - Example: 0.30 * {summary.required_sample_size} = {rejected_target} "
        "'Rejected' images.\n"
        f"4. Go to the '{summary.master_pool_folder.name}' folder and sample images "
        "until you reach your targets.\n"
        "5. This final collection will be your statistically representative "
        "dataset for the experiment."
    )
