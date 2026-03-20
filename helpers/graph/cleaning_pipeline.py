from __future__ import annotations

import logging
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np

from helpers.graph.contamination import GraphContaminationParameters, calculate_roi_contamination

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif"}
SKIPPED_NO_MASK = "skipped_no_mask"
SKIPPED_PROCESSING_ERROR = "skipped_processing_error"
SKIPPED_UNEXPECTED_ERROR = "skipped_unexpected_error"
ACCEPTED = "accepted"
REJECTED = "rejected"

Scorer = Callable[[Path, Path, GraphContaminationParameters], float | None]
type ProgressFactory = Callable[[Iterable[str]], Iterable[str]]


@dataclass(frozen=True)
class GraphCleaningSummary:
    """Execution summary for Stage 4.3 graph cleaning."""

    total_images: int
    accepted: int
    rejected: int
    skipped: int
    rejected_images_dir: Path
    rejected_masks_dir: Path


@dataclass(frozen=True)
class _CleaningDirectories:
    images: Path
    masks: Path
    rejected_images: Path
    rejected_masks: Path


def run_graph_cleaning_pipeline(
    *,
    source_image_dir: Path,
    source_mask_dir: Path,
    output_base_dir: Path,
    graph_params: GraphContaminationParameters,
    tau: float,
    num_workers: int,
    logger: logging.Logger,
    scorer: Scorer = calculate_roi_contamination,
    progress_factory: ProgressFactory | None = None,
) -> GraphCleaningSummary:
    """Run Stage 4.3 filtering and move rejected image/mask pairs."""

    started_at = time.time()
    logger.info("--- Starting Production Image Filtering Process ---")
    logger.info("Using optimal parameters: %s | tau=%.2f", graph_params, tau)
    logger.info("Distributing work across %s CPU cores.", num_workers)

    rejected_images_dir = output_base_dir / "REJECTED_IMAGES"
    rejected_masks_dir = output_base_dir / "REJECTED_MASKS"
    rejected_images_dir.mkdir(parents=True, exist_ok=True)
    rejected_masks_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rejected files will be moved to: %s", output_base_dir)

    image_files = _list_image_files(source_image_dir, logger)
    if not image_files:
        return GraphCleaningSummary(
            total_images=0,
            accepted=0,
            rejected=0,
            skipped=0,
            rejected_images_dir=rejected_images_dir,
            rejected_masks_dir=rejected_masks_dir,
        )

    logger.info("Found %s images to process.", len(image_files))
    directories = _CleaningDirectories(
        images=source_image_dir,
        masks=source_mask_dir,
        rejected_images=rejected_images_dir,
        rejected_masks=rejected_masks_dir,
    )

    results = _process_images(
        image_files=image_files,
        directories=directories,
        graph_params=graph_params,
        tau=tau,
        num_workers=num_workers,
        scorer=scorer,
        progress_factory=progress_factory,
    )
    result_counts = Counter(results)
    skipped_total = (
        result_counts[SKIPPED_NO_MASK]
        + result_counts[SKIPPED_PROCESSING_ERROR]
        + result_counts[SKIPPED_UNEXPECTED_ERROR]
    )

    logger.info("\n--- Filtering Complete ---")
    logger.info("Total images analyzed: %s", len(image_files))
    logger.info("Images Accepted (Kept in source folder): %s", result_counts[ACCEPTED])
    logger.info("Images Rejected (Moved to output folder): %s", result_counts[REJECTED])
    logger.info("Images Skipped (Errors or missing masks): %s", skipped_total)
    logger.info("Total execution time: %.2f minutes.", (time.time() - started_at) / 60)
    logger.info("A detailed log has been saved to: %s", _resolve_log_destination(logger))
    return GraphCleaningSummary(
        total_images=len(image_files),
        accepted=result_counts[ACCEPTED],
        rejected=result_counts[REJECTED],
        skipped=skipped_total,
        rejected_images_dir=rejected_images_dir,
        rejected_masks_dir=rejected_masks_dir,
    )


def build_cleaning_message(summary: GraphCleaningSummary) -> str:
    """Build the final operator-facing result message."""

    return (
        "\n\n--- Cleaning Complete ---\n"
        f"Total images analyzed: {summary.total_images}\n"
        f"Accepted: {summary.accepted}\n"
        f"Rejected: {summary.rejected}\n"
        f"Skipped: {summary.skipped}\n"
        f"Rejected images: {summary.rejected_images_dir}\n"
        f"Rejected masks: {summary.rejected_masks_dir}"
    )


def process_image_worker(
    image_filename: str,
    *,
    directories: _CleaningDirectories,
    graph_params: GraphContaminationParameters,
    tau: float,
    scorer: Scorer = calculate_roi_contamination,
) -> str:
    """Process one image and move it when the contamination rate exceeds tau."""

    logger = logging.getLogger("graph_cleaning")
    try:
        source_image_path = directories.images / image_filename
        source_mask_path = directories.masks / image_filename
        if not source_mask_path.exists():
            logger.debug("Mask not found for image '%s'. Skipping.", image_filename)
            return SKIPPED_NO_MASK

        contamination_rate = scorer(source_image_path, source_mask_path, graph_params)
        if contamination_rate is None or np.isnan(contamination_rate):
            return SKIPPED_PROCESSING_ERROR

        if contamination_rate > tau:
            shutil.move(str(source_image_path), str(directories.rejected_images / image_filename))
            shutil.move(str(source_mask_path), str(directories.rejected_masks / image_filename))
            logger.debug(
                "Rejected '%s' with contamination rate: %.3f", image_filename, contamination_rate
            )
            return REJECTED

        logger.debug(
            "Accepted '%s' with contamination rate: %.3f", image_filename, contamination_rate
        )
        return ACCEPTED
    except OSError as error:
        logger.error("An unexpected error occurred while processing %s: %s", image_filename, error)
        return SKIPPED_UNEXPECTED_ERROR


def _list_image_files(source_image_dir: Path, logger: logging.Logger) -> list[str]:
    if not source_image_dir.is_dir():
        logger.error("The source image directory '%s' does not exist.", source_image_dir)
        return []
    image_files = [
        path.name
        for path in sorted(source_image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not image_files:
        logger.error("No image files found in the source directory: '%s'.", source_image_dir)
    return image_files


def _process_images(
    *,
    image_files: Sequence[str],
    directories: _CleaningDirectories,
    graph_params: GraphContaminationParameters,
    tau: float,
    num_workers: int,
    scorer: Scorer,
    progress_factory: ProgressFactory | None,
) -> list[str]:
    if num_workers <= 1:
        results_iterable: Iterable[str] = (
            process_image_worker(
                image_filename,
                directories=directories,
                graph_params=graph_params,
                tau=tau,
                scorer=scorer,
            )
            for image_filename in image_files
        )
        return list(
            progress_factory(results_iterable) if progress_factory is not None else results_iterable
        )

    worker = partial(
        process_image_worker,
        directories=directories,
        graph_params=graph_params,
        tau=tau,
        scorer=scorer,
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iterable = executor.map(worker, image_files)
        if progress_factory is not None:
            return list(progress_factory(cast(Iterable[str], results_iterable)))
        return list(results_iterable)


def _resolve_log_destination(logger: logging.Logger) -> str:
    for handler in logger.handlers:
        filename = getattr(handler, "baseFilename", None)
        if isinstance(filename, str):
            return filename
    return "log file"
