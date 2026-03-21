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
    discover_image_mask_pairs,
    ensure_manual_labeling_directories,
    select_sample_stems,
)

OverlayRunner = Callable[[Sequence[OverlayTask], int], list[bool]]


@dataclass(frozen=True)
class OptimizationSamplingSummary:
    """Execution summary for Stage 4 optimization sampling."""

    total_population: int
    required_sample_size: int
    pilot_sample_size: int
    master_pool_size: int
    requested_overlay_count: int
    generated_overlay_count: int
    failed_overlay_count: int
    pilot_folder: Path
    master_pool_folder: Path


def run_optimization_sampling(
    *,
    image_folder: Path,
    mask_folder: Path,
    output_base: Path,
    confidence_level: float,
    margin_of_error: float,
    proportion: float,
    pilot_sample_size: int,
    master_pool_fraction: float,
    overlay_color: tuple[int, int, int],
    overlay_thickness: int,
    overlay_alpha: float,
    num_processes: int,
    logger: logging.Logger | None = None,
    rng: random.Random | None = None,
    overlay_runner: OverlayRunner = generate_overlay_images,
) -> OptimizationSamplingSummary:
    """Run the current Stage 4 optimization sampling workflow."""

    active_logger = logger or logging.getLogger("optimization_sampling")
    active_logger.info("--- Experiment Setup Initiated: On-the-Fly Generation ---")
    active_logger.info("Scanning for images in: %s", image_folder)
    active_logger.info("Scanning for masks in: %s", mask_folder)

    pairs = discover_image_mask_pairs(image_folder, mask_folder)
    selection = select_sample_stems(
        list(pairs),
        pilot_sample_size=pilot_sample_size,
        master_pool_fraction=master_pool_fraction,
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
        proportion=proportion,
        rng=rng,
    )
    active_logger.info(
        "Identified %s patient-aware sampling units from valid image-mask pairs.",
        selection.total_population,
    )
    active_logger.info("--- Experiment Parameters ---")
    active_logger.info(
        "Statistically Required Sample Size (%.0f%% confidence, %.0f%% error): %s",
        confidence_level * 100,
        margin_of_error * 100,
        selection.required_sample_size,
    )
    active_logger.info(
        "Pilot Sample Size (for estimating proportions): %s", selection.pilot_sample_size
    )
    active_logger.info(
        "Master Candidate Pool Size (%.0f%% of total): %s",
        master_pool_fraction * 100,
        selection.master_pool_size,
    )
    active_logger.info(
        "Randomly selected %s patient-disjoint representatives for the master pool.",
        len(selection.master_pool_stems),
    )
    active_logger.info(
        "Randomly selected %s patient-disjoint representatives for the pilot sample.",
        len(selection.pilot_sample_stems),
    )

    tasks = build_overlay_tasks(
        pairs=pairs,
        output_base=output_base,
        master_pool_stems=selection.master_pool_stems,
        pilot_sample_stems=selection.pilot_sample_stems,
        color=overlay_color,
        thickness=overlay_thickness,
        alpha=overlay_alpha,
    )
    active_logger.info(
        "Starting on-the-fly overlay generation for %s selected images using %s processes.",
        len(tasks),
        num_processes,
    )
    results = overlay_runner(tasks, num_processes)
    success_count = sum(results)
    active_logger.info(
        "Successfully generated %s out of %s required overlay images.",
        success_count,
        len(tasks),
    )
    if success_count < len(tasks):
        active_logger.warning(
            "%s images failed to process. Check the Stage 4 log for details.",
            len(tasks) - success_count,
        )

    pilot_folder, master_pool_folder = ensure_manual_labeling_directories(output_base)
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
    """Build the operator instructions shown after Stage 4 setup completes."""

    approved_target = math.ceil(0.7 * summary.required_sample_size)
    rejected_target = math.ceil(0.3 * summary.required_sample_size)
    return (
        "\n\n--- Setup Complete! (Efficient Method) ---\n"
        "Overlays for the pilot and master pools were generated on-the-fly.\n"
        "NO intermediate 'overlays' folder was created, saving significant disk space.\n"
        "\nA detailed log has been saved to the Stage 4 log file.\n"
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
