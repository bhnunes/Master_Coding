from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    filename: str
    label: int
    patient_id: int


def _list_png_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(
        path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )


def _build_source_specs(split_name: str) -> tuple[tuple[str, str, int], ...]:
    return (
        (f"{split_name}/CANCER", f"{split_name}/CANCER_MASK", 1),
        (f"{split_name}/NOT_CANCER", f"{split_name}/NOT_CANCER_MASK", 0),
    )


def discover_split_samples(
    base_dir: Path, split_name: str, patient_id_regex: str
) -> list[SampleRecord]:
    split = split_name.upper()
    patient_pattern = re.compile(patient_id_regex)
    samples: list[SampleRecord] = []

    for image_dir_name, mask_dir_name, label in _build_source_specs(split):
        image_dir = base_dir / image_dir_name
        mask_dir = base_dir / mask_dir_name
        image_names = _list_png_names(image_dir)
        mask_names = _list_png_names(mask_dir)
        if not image_names and not mask_names:
            continue
        if image_names != mask_names:
            raise ValueError(f"Image/mask filename mismatch for {split} source '{image_dir_name}'.")
        for filename in image_names:
            patient_match = patient_pattern.search(filename)
            if patient_match is None:
                raise ValueError(f"Could not extract patient id from filename: {filename}")
            samples.append(
                SampleRecord(
                    image_path=image_dir / filename,
                    mask_path=mask_dir / filename,
                    filename=filename,
                    label=label,
                    patient_id=int(patient_match.group(1)),
                )
            )

    return sorted(samples, key=lambda sample: sample.filename)
