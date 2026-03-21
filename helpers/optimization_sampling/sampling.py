from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_CONFIDENCE_LEVELS: dict[float, float] = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


@dataclass(frozen=True)
class ImageMaskPair:
    """A matched image and mask pair keyed by shared stem."""

    stem: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class OverlayTask:
    """A single overlay generation task."""

    image_path: Path
    mask_path: Path
    output_path: Path
    color: tuple[int, int, int]
    thickness: int
    alpha: float


@dataclass(frozen=True)
class SampleSelection:
    """Selected Stage 4 sample groups and derived statistics."""

    total_population: int
    required_sample_size: int
    pilot_sample_size: int
    master_pool_size: int
    master_pool_stems: list[str]
    pilot_sample_stems: list[str]


def infer_sampling_group_id(stem: str) -> str:
    """Infer a patient-aware grouping key from a patch stem."""

    patient_match = re.search(r"PATIENT_([^_]+)", stem)
    if patient_match is not None:
        candidate = patient_match.group(1).strip("-_")
        if candidate:
            return candidate

    slide_match = re.search(r"SLIDE_([^_]+)", stem)
    if slide_match is not None:
        candidate = slide_match.group(1).strip("-_")
        if candidate:
            return candidate

    return stem


def build_group_representatives(
    stems: Sequence[str],
    *,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Choose one representative stem per inferred patient/slide group."""

    active_rng = rng or random.Random()
    stems_by_group: dict[str, list[str]] = {}
    for stem in stems:
        group_id = infer_sampling_group_id(stem)
        stems_by_group.setdefault(group_id, []).append(stem)

    representatives: dict[str, str] = {}
    for group_id, group_stems in stems_by_group.items():
        ordered = sorted(group_stems)
        representatives[group_id] = active_rng.choice(ordered)
    return representatives


def calculate_cochran_sample_size(
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    proportion: float = 0.5,
) -> int:
    """Calculate sample size using Cochran's formula."""

    try:
        z_score = SUPPORTED_CONFIDENCE_LEVELS[confidence_level]
    except KeyError as error:
        raise ValueError("Confidence level must be one of 0.90, 0.95, or 0.99.") from error

    return math.ceil((z_score**2 * proportion * (1 - proportion)) / (margin_of_error**2))


def discover_image_mask_pairs(image_folder: Path, mask_folder: Path) -> dict[str, ImageMaskPair]:
    """Discover image/mask pairs that share the same file stem."""

    if not image_folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
    if not mask_folder.is_dir():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_folder}")

    image_files = {path.stem: path for path in sorted(image_folder.iterdir()) if path.is_file()}
    mask_files = {path.stem: path for path in sorted(mask_folder.iterdir()) if path.is_file()}
    common_stems = sorted(set(image_files) & set(mask_files))

    if not common_stems:
        raise ValueError(
            "No matching image and mask files found. Please ensure filenames are identical."
        )

    return {
        stem: ImageMaskPair(stem=stem, image_path=image_files[stem], mask_path=mask_files[stem])
        for stem in common_stems
    }


def select_sample_stems(
    stems: Sequence[str],
    *,
    pilot_sample_size: int,
    master_pool_fraction: float,
    confidence_level: float,
    margin_of_error: float,
    proportion: float,
    rng: random.Random | None = None,
) -> SampleSelection:
    """Select non-overlapping master and pilot stems with patient-aware grouping."""

    if not stems:
        raise ValueError("No image/mask pairs are available for sampling.")

    representatives = build_group_representatives(stems, rng=rng)
    grouped_stems = list(representatives.values())
    total_population = len(grouped_stems)
    required_sample_size = calculate_cochran_sample_size(
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
        proportion=proportion,
    )
    master_pool_size = math.ceil(total_population * master_pool_fraction)
    if total_population < pilot_sample_size + master_pool_size:
        raise ValueError(
            "Not enough images to create non-overlapping pilot and master pool samples."
        )

    selected_stems = list(grouped_stems)
    active_rng = rng or random.Random()
    active_rng.shuffle(selected_stems)
    master_pool_stems = selected_stems[:master_pool_size]
    pilot_sample_stems = selected_stems[master_pool_size : master_pool_size + pilot_sample_size]

    return SampleSelection(
        total_population=total_population,
        required_sample_size=required_sample_size,
        pilot_sample_size=pilot_sample_size,
        master_pool_size=master_pool_size,
        master_pool_stems=master_pool_stems,
        pilot_sample_stems=pilot_sample_stems,
    )


def build_overlay_tasks(
    *,
    pairs: Mapping[str, ImageMaskPair],
    output_base: Path,
    master_pool_stems: Sequence[str],
    pilot_sample_stems: Sequence[str],
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> list[OverlayTask]:
    """Build overlay tasks for the selected master and pilot pools."""

    tasks: list[OverlayTask] = []
    master_output = output_base / "master_candidate_pool"
    pilot_output = output_base / "pilot_sample"

    for stem in master_pool_stems:
        pair = pairs[stem]
        tasks.append(
            OverlayTask(
                image_path=pair.image_path,
                mask_path=pair.mask_path,
                output_path=master_output / pair.image_path.name,
                color=color,
                thickness=thickness,
                alpha=alpha,
            )
        )

    for stem in pilot_sample_stems:
        pair = pairs[stem]
        tasks.append(
            OverlayTask(
                image_path=pair.image_path,
                mask_path=pair.mask_path,
                output_path=pilot_output / pair.image_path.name,
                color=color,
                thickness=thickness,
                alpha=alpha,
            )
        )

    return tasks


def ensure_manual_labeling_directories(output_base: Path) -> tuple[Path, Path]:
    """Create the manual-review folders required by the current workflow."""

    pilot_folder = output_base / "pilot_sample"
    master_folder = output_base / "master_candidate_pool"
    for folder in (pilot_folder, master_folder):
        (folder / "APPROVED").mkdir(parents=True, exist_ok=True)
        (folder / "REJECTED").mkdir(parents=True, exist_ok=True)
    return pilot_folder, master_folder
