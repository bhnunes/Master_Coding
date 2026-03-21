from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from helpers.optimization_sampling.pipeline import run_optimization_sampling
from helpers.optimization_sampling.sampling import (
    OverlayTask,
    build_group_representatives,
    build_overlay_tasks,
    calculate_cochran_sample_size,
    discover_image_mask_pairs,
    infer_sampling_group_id,
    select_sample_stems,
)


def test_calculate_cochran_sample_size_preserves_existing_defaults() -> None:
    assert calculate_cochran_sample_size() == 385


def test_discover_image_mask_pairs_returns_only_matching_stems(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "case_a.png").write_bytes(b"image")
    (image_dir / "case_b.png").write_bytes(b"image")
    (mask_dir / "case_a.png").write_bytes(b"mask")
    (mask_dir / "case_c.png").write_bytes(b"mask")

    pairs = discover_image_mask_pairs(image_dir, mask_dir)

    assert list(pairs) == ["case_a"]
    assert pairs["case_a"].image_path == image_dir / "case_a.png"
    assert pairs["case_a"].mask_path == mask_dir / "case_a.png"


def test_select_sample_stems_returns_non_overlapping_groups() -> None:
    selection = select_sample_stems(
        stems=[f"case_{index}" for index in range(1000)],
        pilot_sample_size=100,
        master_pool_fraction=0.10,
        confidence_level=0.95,
        margin_of_error=0.05,
        proportion=0.5,
        rng=random.Random(7),
    )

    assert selection.required_sample_size == 385
    assert selection.master_pool_size == 100
    assert len(selection.master_pool_stems) == 100
    assert len(selection.pilot_sample_stems) == 100
    assert set(selection.master_pool_stems).isdisjoint(selection.pilot_sample_stems)


def test_infer_sampling_group_id_prefers_patient_ids() -> None:
    assert infer_sampling_group_id("CANCER_PATIENT_42_SLIDE_a_X_1_Y_2") == "42"
    assert infer_sampling_group_id("NOT_CANCER_SLIDE_slide7_PATCH_3") == "slide7"
    assert infer_sampling_group_id("case_1") == "case_1"


def test_build_group_representatives_selects_one_stem_per_patient() -> None:
    representatives = build_group_representatives(
        [
            "CANCER_PATIENT_1_PATCH_A",
            "CANCER_PATIENT_1_PATCH_B",
            "CANCER_PATIENT_2_PATCH_A",
        ],
        rng=random.Random(4),
    )

    assert set(representatives) == {"1", "2"}
    assert representatives["1"] in {"CANCER_PATIENT_1_PATCH_A", "CANCER_PATIENT_1_PATCH_B"}
    assert representatives["2"] == "CANCER_PATIENT_2_PATCH_A"


def test_select_sample_stems_uses_patient_aware_population_and_disjoint_groups() -> None:
    stems = [
        f"CANCER_PATIENT_{patient}_PATCH_{patch}" for patient in range(1, 251) for patch in (1, 2)
    ]

    selection = select_sample_stems(
        stems=stems,
        pilot_sample_size=50,
        master_pool_fraction=0.20,
        confidence_level=0.95,
        margin_of_error=0.05,
        proportion=0.5,
        rng=random.Random(9),
    )

    assert selection.total_population == 250
    master_groups = {infer_sampling_group_id(stem) for stem in selection.master_pool_stems}
    pilot_groups = {infer_sampling_group_id(stem) for stem in selection.pilot_sample_stems}
    assert len(selection.master_pool_stems) == len(master_groups)
    assert len(selection.pilot_sample_stems) == len(pilot_groups)
    assert master_groups.isdisjoint(pilot_groups)


def test_select_sample_stems_rejects_small_population() -> None:
    with pytest.raises(ValueError, match="Not enough images"):
        select_sample_stems(
            stems=[f"case_{index}" for index in range(150)],
            pilot_sample_size=100,
            master_pool_fraction=0.5,
            confidence_level=0.95,
            margin_of_error=0.05,
            proportion=0.5,
            rng=random.Random(1),
        )


def test_build_overlay_tasks_preserves_output_folder_contract(tmp_path: Path) -> None:
    pairs = discover_image_mask_pairs(_build_dataset(tmp_path, 4), tmp_path / "masks")

    tasks = build_overlay_tasks(
        pairs=pairs,
        output_base=tmp_path / "output",
        master_pool_stems=["case_0", "case_1"],
        pilot_sample_stems=["case_2", "case_3"],
        color=(0, 0, 255),
        thickness=2,
        alpha=1.0,
    )

    output_paths = {task.output_path for task in tasks}
    assert tmp_path / "output" / "master_candidate_pool" / "case_0.png" in output_paths
    assert tmp_path / "output" / "pilot_sample" / "case_2.png" in output_paths


def test_run_optimization_sampling_creates_manual_review_folders(tmp_path: Path) -> None:
    image_dir = _build_dataset(tmp_path, 1100)
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    created_outputs: list[Path] = []

    def fake_overlay_runner(tasks: Sequence[OverlayTask], num_processes: int) -> list[bool]:
        del num_processes
        for task in tasks:
            output_path = task.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"overlay")
            created_outputs.append(output_path)
        return [True] * len(tasks)

    summary = run_optimization_sampling(
        image_folder=image_dir,
        mask_folder=mask_dir,
        output_base=output_dir,
        confidence_level=0.95,
        margin_of_error=0.05,
        proportion=0.5,
        pilot_sample_size=100,
        master_pool_fraction=0.10,
        overlay_color=(0, 0, 255),
        overlay_thickness=2,
        overlay_alpha=1.0,
        num_processes=2,
        rng=random.Random(3),
        overlay_runner=fake_overlay_runner,
    )

    assert summary.total_population == 1100
    assert summary.generated_overlay_count == len(created_outputs)
    assert (output_dir / "pilot_sample" / "APPROVED").is_dir()
    assert (output_dir / "pilot_sample" / "REJECTED").is_dir()
    assert (output_dir / "master_candidate_pool" / "APPROVED").is_dir()
    assert (output_dir / "master_candidate_pool" / "REJECTED").is_dir()


def _build_dataset(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    for index in range(count):
        filename = f"case_{index}.png"
        (image_dir / filename).write_bytes(b"image")
        (mask_dir / filename).write_bytes(b"mask")
    return image_dir
