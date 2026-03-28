from pathlib import Path

import h5py
import numpy as np

from helpers.extraction.hdf5_manifest import update_stage2_shard_manifest


def _write_shard(path: Path, *, row_count: int, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((row_count, 2, 2, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((row_count, 2, 2), dtype=np.uint8))
        handle.create_dataset("labels", data=np.zeros((row_count,), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.ones((row_count,), dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([f"patch_{index}.png".encode() for index in range(row_count)]),
        )
        handle.create_dataset(
            "source_image_paths",
            data=np.array([f"/src/{index}.png".encode() for index in range(row_count)]),
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array([f"/src/{index}_mask.png".encode() for index in range(row_count)]),
        )
        handle.attrs["source_signature"] = signature


def test_update_stage2_shard_manifest_upserts_and_sorts_entries(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard_a = shard_dir / "slide_a.h5"
    shard_b = shard_dir / "slide_b.h5"
    _write_shard(shard_b, row_count=1, signature="sig-b")
    _write_shard(shard_a, row_count=2, signature="sig-a")

    update_stage2_shard_manifest(shard_dir / "manifest.json", shard_b)
    update_stage2_shard_manifest(shard_dir / "manifest.json", shard_a)

    manifest = (shard_dir / "manifest.json").read_text(encoding="utf-8")

    assert '"relative_path":"slide_a.h5"' in manifest
    assert manifest.index('"relative_path":"slide_a.h5"') < manifest.index(
        '"relative_path":"slide_b.h5"'
    )
    assert '"row_count":2' in manifest
    assert '"source_signature":"sig-a"' in manifest


def test_update_stage2_shard_manifest_removes_missing_shard_entry(tmp_path: Path) -> None:
    shard_dir = tmp_path / "HDF5_SHARDS"
    shard = shard_dir / "slide_a.h5"
    _write_shard(shard, row_count=1, signature="sig-a")

    manifest_path = shard_dir / "manifest.json"
    update_stage2_shard_manifest(manifest_path, shard)
    shard.unlink()
    update_stage2_shard_manifest(manifest_path, shard)

    assert manifest_path.read_text(encoding="utf-8") == '{"shards":[],"version":1}\n'
