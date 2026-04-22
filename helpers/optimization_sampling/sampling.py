from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py

SUPPORTED_CONFIDENCE_LEVELS: dict[float, float] = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


@dataclass(frozen=True)
class ImageMaskPair:
    """A matched image and mask pair keyed by shared stem."""

    stem: str
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


def discover_hdf5_image_mask_pairs(source_hdf5_path: Path) -> dict[str, ImageMaskPair]:
    """Discover image/mask pairs from a canonical HDF5 dataset."""

    if not source_hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 source does not exist: {source_hdf5_path}")

    pairs: dict[str, ImageMaskPair] = {}
    seen_stems: set[str] = set()
    with h5py.File(source_hdf5_path, "r") as handle:
        filenames = cast(h5py.Dataset, handle["filenames"])
        filename_values = filenames[:]
        for index, filename_value in enumerate(filename_values):
            filename = (
                filename_value.decode("utf-8")
                if isinstance(filename_value, bytes)
                else str(filename_value)
            )
            stem = Path(filename).stem
            if stem in seen_stems:
                raise ValueError(
                    f"Duplicate filename stem '{stem}' found at row {index} in {source_hdf5_path}. "
                    "Stems must be unique for Stage 3.1 sampling."
                )
            pairs[stem] = ImageMaskPair(
                stem=stem,
                image_path=f"{source_hdf5_path}::images[{index}]",
                mask_path=f"{source_hdf5_path}::masks[{index}]",
                output_name=filename,
            )
            seen_stems.add(stem)
    if not pairs:
        raise ValueError(f"No rows found in HDF5 source: {source_hdf5_path}")
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
