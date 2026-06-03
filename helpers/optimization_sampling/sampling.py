from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from helpers.extraction.master_manifest import MasterManifest

SUPPORTED_CONFIDENCE_LEVELS: dict[float, float] = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


@dataclass(frozen=True)
class ImageMaskPair:
    """A matched image and mask pair keyed by shared stem."""

    stem: str
    label: int
    image_path: Path | str
    mask_path: Path | str
    output_name: str


@dataclass(frozen=True)
class OverlayTask:
    """A single overlay generation task."""

    image_path: Path | str
    mask_path: Path | str
    output_path: Path
    color: tuple[int, int, int]
    thickness: int
    alpha: float


@dataclass(frozen=True)
class SampleSelection:
    """Selected Stage 3.1 image-level samples and derived statistics."""

    total_population: int
    required_sample_size: int
    pilot_sample_size: int
    master_pool_size: int
    master_pool_stems: list[str]
    pilot_sample_stems: list[str]


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


def discover_manifest_image_mask_pairs(master_manifest_path: Path) -> dict[str, ImageMaskPair]:
    """Discover canonical cancer image/mask pairs from the Stage 2 master manifest."""

    records = MasterManifest(master_manifest_path).list_stage2_patch_records()
    if not records:
        raise ValueError(f"No Stage 2 rows found in master manifest: {master_manifest_path}")

    pairs: dict[str, ImageMaskPair] = {}
    for record in records:
        if record.label != 1:
            continue
        stem = Path(record.filename).stem
        if stem in pairs:
            raise ValueError(
                "Duplicate filename stem "
                f"'{stem}' found at row {record.source_row_index} in {record.source_hdf5_path}. "
                "Stems must be unique across canonical Stage 2 manifest rows for Stage 3.1 "
                "sampling."
            )
        pairs[stem] = ImageMaskPair(
            stem=stem,
            label=record.label,
            image_path=record.source_image_path,
            mask_path=record.source_mask_path,
            output_name=record.filename,
        )

    if not pairs:
        raise ValueError(f"No cancer Stage 2 rows found in master manifest: {master_manifest_path}")

    return pairs


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
    """Select non-overlapping master and pilot stems at the image level."""

    if not stems:
        raise ValueError("No image/mask pairs are available for sampling.")

    total_population = len(stems)
    required_sample_size = calculate_cochran_sample_size(
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
        proportion=proportion,
    )
    master_pool_size = math.ceil(total_population * master_pool_fraction)
    if total_population < pilot_sample_size + master_pool_size:
        raise ValueError(
            "Not enough images to create non-overlapping pilot and master pool samples: "
            f"population={total_population}, pilot={pilot_sample_size}, "
            f"master_pool={master_pool_size}."
        )

    selected_stems = list(stems)
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
                output_path=master_output / pair.output_name,
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
                output_path=pilot_output / pair.output_name,
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
