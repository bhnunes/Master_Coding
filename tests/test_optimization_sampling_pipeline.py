from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from helpers.extraction.master_manifest import MasterManifest, Stage2SlideRows
from helpers.optimization_sampling.pipeline import (
    OptimizationSamplingConfig,
    run_optimization_sampling,
)
from helpers.optimization_sampling.sampling import (
    ImageMaskPair,
    OverlayTask,
    build_overlay_tasks,
    calculate_cochran_sample_size,
    discover_manifest_image_mask_pairs,
    select_sample_stems,
)

DEFAULT_COHCRAN_SAMPLE_SIZE = 385
MASTER_POOL_SAMPLE_SIZE = 100
PILOT_SAMPLE_SIZE = 100
REPEATED_PATIENT_POPULATION = 500
REPEATED_PATIENT_PILOT_SAMPLE_SIZE = 50
SHARED_SLIDE_POPULATION = 1200
SHARED_SLIDE_MASTER_POOL_SIZE = 120
HDF5_SOURCE_POPULATION = 1100
RGBA_CHANNELS = 3
PATCH_SIDE = 4
SEED = 42


def _pipeline_config(
    *,
    master_manifest_path: Path,
    output_base: Path,
    overlay_runner: Any,
    rng: random.Random,
) -> OptimizationSamplingConfig:
    return OptimizationSamplingConfig(
        master_manifest_path=master_manifest_path,
        output_base=output_base,
        confidence_level=0.95,
        margin_of_error=0.05,
        proportion=0.5,
        pilot_sample_size=100,
        master_pool_fraction=0.10,
        overlay_color=(0, 0, 255),
        overlay_thickness=2,
        overlay_alpha=1.0,
        seed=SEED,
        num_processes=2,
        rng=rng,
        overlay_runner=overlay_runner,
    )


def _write_manifest_for_shard(
    *,
    master_manifest_path: Path,
    shard_path: Path,
    filenames: Sequence[str],
    labels: Sequence[int],
    patient_ids: Sequence[int],
) -> None:
    MasterManifest(master_manifest_path).replace_stage2_slide_rows(
        Stage2SlideRows(
            source_hdf5_path=shard_path,
            records=[
                {
                    "filename": filename,
                    "patient_id": patient_id,
                    "label": label,
                }
                for filename, label, patient_id in zip(filenames, labels, patient_ids, strict=True)
            ],
            source_slide_path=shard_path.with_suffix(".svs"),
            annotation_path=None,
            artifacts_geojson_path=None,
            stage2_case_record_id=1,
            stage2_processing_signature="stage2-test",
            stage2_status="completed",
        )
    )


def test_calculate_cochran_sample_size_preserves_existing_defaults() -> None:
    assert calculate_cochran_sample_size() == DEFAULT_COHCRAN_SAMPLE_SIZE


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

    assert selection.required_sample_size == DEFAULT_COHCRAN_SAMPLE_SIZE
    assert selection.master_pool_size == MASTER_POOL_SAMPLE_SIZE
    assert len(selection.master_pool_stems) == MASTER_POOL_SAMPLE_SIZE
    assert len(selection.pilot_sample_stems) == PILOT_SAMPLE_SIZE
    assert set(selection.master_pool_stems).isdisjoint(selection.pilot_sample_stems)


def test_select_sample_stems_uses_image_level_population_even_with_repeated_patients() -> None:
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

    assert selection.total_population == REPEATED_PATIENT_POPULATION
    assert len(selection.master_pool_stems) == MASTER_POOL_SAMPLE_SIZE
    assert len(selection.pilot_sample_stems) == REPEATED_PATIENT_PILOT_SAMPLE_SIZE
    assert set(selection.master_pool_stems).isdisjoint(selection.pilot_sample_stems)


def test_select_sample_stems_does_not_collapse_large_image_population_by_filename_pattern() -> None:
    stems = [f"CANCER_PATIENT_shared_SLIDE_shared_X_{index}_Y_{index}" for index in range(1200)]

    selection = select_sample_stems(
        stems=stems,
        pilot_sample_size=100,
        master_pool_fraction=0.10,
        confidence_level=0.95,
        margin_of_error=0.05,
        proportion=0.5,
        rng=random.Random(5),
    )

    assert selection.total_population == SHARED_SLIDE_POPULATION
    assert len(selection.master_pool_stems) == SHARED_SLIDE_MASTER_POOL_SIZE
    assert len(selection.pilot_sample_stems) == PILOT_SAMPLE_SIZE
    assert set(selection.master_pool_stems).isdisjoint(selection.pilot_sample_stems)


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
            label=1,
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


def test_discover_manifest_image_mask_pairs_reads_canonical_rows_from_master_manifest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_a = source_path / "patient_7.h5"
    shard_b = source_path / "patient_8.h5"
    with h5py.File(shard_a, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_7_PATCH_001.png"]))
        handle.attrs["source_signature"] = "signature-a"
    with h5py.File(shard_b, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([8], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_8_PATCH_001.png"]))
        handle.attrs["source_signature"] = "signature-b"

    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_a,
        filenames=["PATIENT_7_PATCH_001.png"],
        labels=[1],
        patient_ids=[7],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_b,
        filenames=["PATIENT_8_PATCH_001.png"],
        labels=[0],
        patient_ids=[8],
    )

    pairs = discover_manifest_image_mask_pairs(master_manifest_path)

    assert list(pairs) == ["PATIENT_7_PATCH_001"]
    assert pairs["PATIENT_7_PATCH_001"].label == 1
    assert pairs["PATIENT_7_PATCH_001"].output_name == "PATIENT_7_PATCH_001.png"
    assert str(pairs["PATIENT_7_PATCH_001"].image_path).endswith("::images[0]")
    assert "PATIENT_8_PATCH_001" not in pairs


def test_discover_manifest_image_mask_pairs_rejects_duplicate_stems_across_manifest_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    with h5py.File(source_path / "patient_7.h5", "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([7], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"case_1.png"], dtype="S10"),
        )
        handle.attrs["source_signature"] = "signature-a"
    with h5py.File(source_path / "patient_8.h5", "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.array([0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([8], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"case_1.tiff"], dtype="S11"),
        )
        handle.attrs["source_signature"] = "signature-b"

    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "patient_7.h5",
        filenames=["case_1.png"],
        labels=[1],
        patient_ids=[7],
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=source_path / "patient_8.h5",
        filenames=["case_1.tiff"],
        labels=[1],
        patient_ids=[8],
    )

    with pytest.raises(ValueError, match="Duplicate filename stem 'case_1'"):
        discover_manifest_image_mask_pairs(master_manifest_path)


def test_discover_manifest_image_mask_pairs_rejects_missing_master_manifest(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"

    with pytest.raises(FileNotFoundError, match="Stage 3 requires an existing master manifest"):
        discover_manifest_image_mask_pairs(master_manifest_path)


def test_discover_manifest_image_mask_pairs_rejects_empty_master_manifest(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    MasterManifest(master_manifest_path).initialize()

    with pytest.raises(ValueError, match="No Stage 2 rows found in master manifest"):
        discover_manifest_image_mask_pairs(master_manifest_path)


def test_discover_manifest_image_mask_pairs_rejects_manifest_without_cancer_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "patient_8.h5"
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((1, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.array([0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([8], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_8_PATCH_001.png"]))
        handle.attrs["source_signature"] = "signature-b"

    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=["PATIENT_8_PATCH_001.png"],
        labels=[0],
        patient_ids=[8],
    )

    with pytest.raises(ValueError, match="No cancer Stage 2 rows found in master manifest"):
        discover_manifest_image_mask_pairs(master_manifest_path)


def test_run_optimization_sampling_creates_manual_review_folders(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "patient_shard.h5"
    output_dir = tmp_path / "output"
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros(
                (HDF5_SOURCE_POPULATION, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS),
                dtype=np.uint8,
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((HDF5_SOURCE_POPULATION, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset("labels", data=np.ones((HDF5_SOURCE_POPULATION,), dtype=np.uint8))
        handle.create_dataset(
            "patient_ids",
            data=np.arange(1, HDF5_SOURCE_POPULATION + 1, dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.array(
                [f"case_{index}.png".encode() for index in range(HDF5_SOURCE_POPULATION)]
            ),
        )
        handle.attrs["source_signature"] = "signature-a"
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=[f"case_{index}.png" for index in range(HDF5_SOURCE_POPULATION)],
        labels=[1] * HDF5_SOURCE_POPULATION,
        patient_ids=list(range(1, HDF5_SOURCE_POPULATION + 1)),
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
        _pipeline_config(
            output_base=output_dir,
            overlay_runner=fake_overlay_runner,
            rng=random.Random(3),
            master_manifest_path=master_manifest_path,
        )
    )

    assert summary.total_population == HDF5_SOURCE_POPULATION
    assert summary.generated_overlay_count == len(created_outputs)
    assert (output_dir / "pilot_sample" / "APPROVED").is_dir()
    assert (output_dir / "pilot_sample" / "REJECTED").is_dir()
    assert (output_dir / "master_candidate_pool" / "APPROVED").is_dir()
    assert (output_dir / "master_candidate_pool" / "REJECTED").is_dir()


def test_run_optimization_sampling_uses_only_cancer_manifest_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_path = source_path / "patient_shard.h5"
    output_dir = tmp_path / "output"
    cancer_count = 1100
    non_cancer_count = 50
    total_rows = cancer_count + non_cancer_count
    with h5py.File(shard_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros((total_rows, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS), dtype=np.uint8),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((total_rows, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset(
            "labels",
            data=np.array(([1] * cancer_count) + ([0] * non_cancer_count), dtype=np.uint8),
        )
        handle.create_dataset(
            "patient_ids",
            data=np.arange(1, total_rows + 1, dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.array([f"case_{index}.png".encode() for index in range(total_rows)]),
        )
        handle.attrs["source_signature"] = "signature-a"

    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_path,
        filenames=[f"case_{index}.png" for index in range(total_rows)],
        labels=([1] * cancer_count) + ([0] * non_cancer_count),
        patient_ids=list(range(1, total_rows + 1)),
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
        _pipeline_config(
            output_base=output_dir,
            overlay_runner=fake_overlay_runner,
            rng=random.Random(3),
            master_manifest_path=master_manifest_path,
        )
    )

    assert summary.total_population == cancer_count
    assert summary.generated_overlay_count == len(created_outputs)
    assert all(path.name.startswith("case_") for path in created_outputs)
    assert all(int(path.stem.split("_")[1]) < cancer_count for path in created_outputs)


def test_run_optimization_sampling_accepts_master_manifest_file(tmp_path: Path) -> None:
    source_path = tmp_path / "PATCHES" / "HDF5_SHARDS"
    source_path.mkdir(parents=True)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    shard_a = source_path / "patient_1.h5"
    shard_b = source_path / "patient_2.h5"
    output_dir = tmp_path / "output"
    with h5py.File(shard_a, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros(
                (SHARED_SLIDE_POPULATION // 2, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS),
                dtype=np.uint8,
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((SHARED_SLIDE_POPULATION // 2, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset(
            "labels",
            data=np.ones((SHARED_SLIDE_POPULATION // 2,), dtype=np.uint8),
        )
        handle.create_dataset(
            "patient_ids",
            data=np.arange(1, (SHARED_SLIDE_POPULATION // 2) + 1, dtype=np.int32),
        )
        handle.create_dataset(
            "filenames",
            data=np.array(
                [
                    f"CANCER_PATIENT_{index}_PATCH_001.png".encode()
                    for index in range(1, (SHARED_SLIDE_POPULATION // 2) + 1)
                ]
            ),
        )
        handle.attrs["source_signature"] = "signature-a"
    with h5py.File(shard_b, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.zeros(
                (SHARED_SLIDE_POPULATION // 2, PATCH_SIDE, PATCH_SIDE, RGBA_CHANNELS),
                dtype=np.uint8,
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.zeros((SHARED_SLIDE_POPULATION // 2, PATCH_SIDE, PATCH_SIDE), dtype=np.uint8),
        )
        handle.create_dataset(
            "labels",
            data=np.ones((SHARED_SLIDE_POPULATION // 2,), dtype=np.uint8),
        )
        handle.create_dataset(
            "patient_ids",
            data=np.arange(
                (SHARED_SLIDE_POPULATION // 2) + 1,
                SHARED_SLIDE_POPULATION + 1,
                dtype=np.int32,
            ),
        )
        handle.create_dataset(
            "filenames",
            data=np.array(
                [
                    f"CANCER_PATIENT_{index}_PATCH_001.png".encode()
                    for index in range(
                        (SHARED_SLIDE_POPULATION // 2) + 1,
                        SHARED_SLIDE_POPULATION + 1,
                    )
                ]
            ),
        )
        handle.attrs["source_signature"] = "signature-b"
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_a,
        filenames=[
            f"CANCER_PATIENT_{index}_PATCH_001.png"
            for index in range(1, (SHARED_SLIDE_POPULATION // 2) + 1)
        ],
        labels=[1] * (SHARED_SLIDE_POPULATION // 2),
        patient_ids=list(range(1, (SHARED_SLIDE_POPULATION // 2) + 1)),
    )
    _write_manifest_for_shard(
        master_manifest_path=master_manifest_path,
        shard_path=shard_b,
        filenames=[
            f"CANCER_PATIENT_{index}_PATCH_001.png"
            for index in range((SHARED_SLIDE_POPULATION // 2) + 1, SHARED_SLIDE_POPULATION + 1)
        ],
        labels=[1] * (SHARED_SLIDE_POPULATION // 2),
        patient_ids=list(range((SHARED_SLIDE_POPULATION // 2) + 1, SHARED_SLIDE_POPULATION + 1)),
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
        _pipeline_config(
            output_base=output_dir,
            overlay_runner=fake_overlay_runner,
            rng=random.Random(3),
            master_manifest_path=master_manifest_path,
        )
    )

    assert summary.total_population == SHARED_SLIDE_POPULATION
    assert summary.generated_overlay_count == len(created_outputs)
    assert created_outputs[0].suffix == ".png"
