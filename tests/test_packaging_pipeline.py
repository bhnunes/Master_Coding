from pathlib import Path

import h5py
import numpy as np

from helpers.packaging.config import PackagingConfig, load_packaging_config
from helpers.packaging.pipeline import run_packaging_pipeline


def test_run_packaging_pipeline_accepts_canonical_hdf5_input(tmp_path: Path) -> None:
    source_path = tmp_path / "stage2_source.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_11_PATCH_001.png", b"PATIENT_22_PATCH_001.png"]),
        )
        handle.create_dataset(
            "source_image_paths",
            data=np.array([b"/src/p11.png", b"/src/p22.png"]),
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array([b"/src/p11_mask.png", b"/src/p22_mask.png"]),
        )

    output_dir = tmp_path / "packaged"
    config = PackagingConfig(
        source_hdf5_path=source_path,
        output_dir=output_dir,
        output_filename="SOURCE_DATASET.h5",
        overwrite_outputs=True,
    )

    result = run_packaging_pipeline(config)

    assert result == output_dir / "SOURCE_DATASET.h5"
    with h5py.File(result, "r") as handle:
        assert handle["labels"][:].tolist() == [1, 0]
        assert handle["patient_ids"][:].tolist() == [11, 22]
        assert handle["filenames"][:].tolist() == [
            b"PATIENT_11_PATCH_001.png",
            b"PATIENT_22_PATCH_001.png",
        ]
        assert handle["source_image_paths"][:].tolist() == [b"/src/p11.png", b"/src/p22.png"]
        assert handle["source_mask_paths"][:].tolist() == [
            b"/src/p11_mask.png",
            b"/src/p22_mask.png",
        ]


def test_run_packaging_pipeline_merges_stage2_hdf5_shards(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_dir.mkdir()
    first_shard = shard_dir / "slide_a.h5"
    second_shard = shard_dir / "slide_b.h5"

    with h5py.File(first_shard, "w") as handle:
        handle.create_dataset("images", data=np.full((1, 4, 4, 3), 10, dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_11_PATCH_001.png"]))
        handle.create_dataset("source_image_paths", data=np.array([b"/src/a.png"]))
        handle.create_dataset("source_mask_paths", data=np.array([b"/src/a_mask.png"]))
        handle.create_dataset("slide_ids", data=np.array([b"slide_a"]))

    with h5py.File(second_shard, "w") as handle:
        handle.create_dataset("images", data=np.full((1, 4, 4, 3), 20, dtype=np.uint8))
        handle.create_dataset("masks", data=np.ones((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([22], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_22_PATCH_001.png"]))
        handle.create_dataset("source_image_paths", data=np.array([b"/src/b.png"]))
        handle.create_dataset("source_mask_paths", data=np.array([b"/src/b_mask.png"]))
        handle.create_dataset("slide_ids", data=np.array([b"slide_b"]))

    output_dir = tmp_path / "packaged"
    config = PackagingConfig(
        source_hdf5_path=shard_dir,
        output_dir=output_dir,
        output_filename="SOURCE_DATASET.h5",
        overwrite_outputs=True,
    )

    result = run_packaging_pipeline(config)

    assert result == output_dir / "SOURCE_DATASET.h5"
    with h5py.File(result, "r") as handle:
        assert handle["labels"][:].tolist() == [0, 1]
        assert handle["patient_ids"][:].tolist() == [11, 22]
        assert handle["filenames"][:].tolist() == [
            b"PATIENT_11_PATCH_001.png",
            b"PATIENT_22_PATCH_001.png",
        ]
        assert handle["source_image_paths"][:].tolist() == [b"/src/a.png", b"/src/b.png"]
        assert handle["source_mask_paths"][:].tolist() == [b"/src/a_mask.png", b"/src/b_mask.png"]


def test_run_packaging_pipeline_filters_source_hdf5_by_accepted_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_11_PATCH_001.png", b"PATIENT_22_PATCH_001.png"]),
        )
        handle.create_dataset(
            "source_image_paths", data=np.array([b"/src/p11.png", b"/src/p22.png"])
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array([b"/src/p11_mask.png", b"/src/p22_mask.png"]),
        )
        handle.create_dataset("slide_ids", data=np.array([b"slide_a", b"slide_b"]))
        handle.attrs["source_signature"] = "stage2-signature"

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        f"PATIENT_22_PATCH_001.png,accepted,0.1,22,slide_b,{source_path},1\n",
        encoding="utf-8",
    )

    config = PackagingConfig(
        source_hdf5_path=source_path,
        accepted_manifest_path=manifest_path,
        output_dir=tmp_path / "packaged",
        output_filename="SOURCE_DATASET.h5",
        overwrite_outputs=True,
    )

    result = run_packaging_pipeline(config)

    with h5py.File(result, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [22]
        assert handle["filenames"][:].tolist() == [b"PATIENT_22_PATCH_001.png"]
        assert handle.attrs["upstream_source_signature"] == "stage2-signature"
        assert handle.attrs["stage4_cleaning_manifest_path"] == str(manifest_path)


def test_run_packaging_pipeline_auto_uses_adjacent_accepted_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    manifest_path = tmp_path / "accepted_manifest.csv"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11, 22], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_11_PATCH_001.png", b"PATIENT_22_PATCH_001.png"]),
        )
        handle.create_dataset(
            "source_image_paths", data=np.array([b"/src/p11.png", b"/src/p22.png"])
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array([b"/src/p11_mask.png", b"/src/p22_mask.png"]),
        )
        handle.attrs["source_signature"] = "stage2-signature"

    manifest_path.write_text(
        "filename,decision,contamination_rate,patient_id,slide_id,source_hdf5_path,source_row_index\n"
        f"PATIENT_11_PATCH_001.png,accepted,0.1,11,slide_a,{source_path},0\n",
        encoding="utf-8",
    )

    config = load_packaging_config(
        {
            "PACKAGING_SOURCE_HDF5_PATH": str(source_path),
            "PACKAGING_OUTPUT_DIR": str(tmp_path / "packaged"),
        }
    )
    result = run_packaging_pipeline(config)

    with h5py.File(result, "r") as handle:
        assert handle["patient_ids"][:].tolist() == [11]
