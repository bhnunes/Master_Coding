from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers.optimization_sampling.pipeline import run_optimization_sampling
from helpers.optimization_sampling.sampling import (
    ImageMaskPair,
    OverlayTask,
    build_group_representatives,
    build_overlay_tasks,
    calculate_cochran_sample_size,
    discover_hdf5_image_mask_pairs,
    infer_sampling_group_id,
    select_sample_stems,
)


def test_calculate_cochran_sample_size_preserves_existing_defaults() -> None:
    assert calculate_cochran_sample_size() == 385


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
    pairs = {
        f"case_{index}": ImageMaskPair(
            stem=f"case_{index}",
            image_path=f"SOURCE_DATASET.h5::images[{index}]",
            mask_path=f"SOURCE_DATASET.h5::masks[{index}]",
            output_name=f"case_{index}.png",
        )
        for index in range(4)
    }

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


def test_discover_hdf5_image_mask_pairs_reads_canonical_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_7_PATCH_001.png"]))

    pairs = discover_hdf5_image_mask_pairs(source_path)

    assert list(pairs) == ["PATIENT_7_PATCH_001"]
    assert pairs["PATIENT_7_PATCH_001"].output_name == "PATIENT_7_PATCH_001.png"
    assert str(pairs["PATIENT_7_PATCH_001"].image_path).endswith("::images[0]")


def test_run_optimization_sampling_creates_manual_review_folders(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    output_dir = tmp_path / "output"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1100, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1100, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.ones((1100,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.arange(1, 1101, dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([f"case_{index}.png".encode() for index in range(1100)]),
        )
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
        source_hdf5_path=source_path,
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


def test_run_optimization_sampling_accepts_hdf5_source(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    output_dir = tmp_path / "output"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1200, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1200, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.ones((1200,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.arange(1, 1201, dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array(
                [f"CANCER_PATIENT_{index}_PATCH_001.png".encode() for index in range(1, 1201)]
            ),
        )
    created_outputs: list[Path] = []

    def fake_overlay_runner(tasks: Sequence[OverlayTask], num_processes: int) -> list[bool]:
        del num_processes
        for task in tasks:
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            task.output_path.write_bytes(b"overlay")
            created_outputs.append(task.output_path)
        return [True] * len(tasks)

    summary = run_optimization_sampling(
        source_hdf5_path=source_path,
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

    assert summary.total_population == 1200
    assert summary.generated_overlay_count == len(created_outputs)
    assert created_outputs[0].suffix == ".png"
