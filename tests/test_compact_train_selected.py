from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers.training.compact_train_selected import (
    build_compact_train_selected_from_decisions,
    remap_records_to_compact_train_selected,
)
from helpers.training.master_manifest_queries import CanonicalRowRecord

PATCH_SIDE = 4
RGB_CHANNELS = 3
SELECTED_ROW_COUNT = 2
FIRST_SELECTED_PIXEL = 10
SECOND_SELECTED_PIXEL = 30


def _write_source_shard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.stack(
        [
            np.full((PATCH_SIDE, PATCH_SIDE, RGB_CHANNELS), row_value, dtype=np.uint8)
            for row_value in (10, 20, 30)
        ]
    )
    masks = np.stack(
        [np.full((PATCH_SIDE, PATCH_SIDE), row_value, dtype=np.uint8) for row_value in (0, 1, 0)]
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=np.asarray([0, 1, 0], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.asarray([7, 7, 7], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.asarray([b"p7_0.png", b"p7_1.png", b"p7_2.png"]),
        )


def _selected_decisions(source_path: Path) -> list[dict[str, object]]:
    return [
        {
            "source_hdf5_path": str(source_path),
            "source_row_index": 0,
            "filename": "p7_0.png",
            "patient_id": 7,
            "label": 0,
            "sampling_decision": "sampled_kept",
            "is_stage7_selected": True,
        },
        {
            "source_hdf5_path": str(source_path),
            "source_row_index": 1,
            "filename": "p7_1.png",
            "patient_id": 7,
            "label": 1,
            "sampling_decision": "rejected_reducible",
            "is_stage7_selected": False,
        },
        {
            "source_hdf5_path": str(source_path),
            "source_row_index": 2,
            "filename": "p7_2.png",
            "patient_id": 7,
            "label": 0,
            "sampling_decision": "protected_kept",
            "is_stage7_selected": True,
        },
    ]


def test_compact_train_selected_copies_only_selected_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "patient_7.h5"
    compact_dir = tmp_path / "compact"
    _write_source_shard(source_path)

    result = build_compact_train_selected_from_decisions(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        decisions=_selected_decisions(source_path),
        compact_dir=compact_dir,
        compression="none",
    )

    assert result.row_count == SELECTED_ROW_COUNT
    assert result.shard_count == 1
    compact_shard = next((compact_dir / "shards").glob("*.h5"))
    with h5py.File(compact_shard, "r") as handle:
        assert handle["images"].shape[0] == SELECTED_ROW_COUNT
        assert handle["labels"][:].tolist() == [0, 0]
        assert handle["filenames"][:].tolist() == [b"p7_0.png", b"p7_2.png"]
        assert int(handle["images"][0, 0, 0, 0]) == FIRST_SELECTED_PIXEL
        assert int(handle["images"][1, 0, 0, 0]) == SECOND_SELECTED_PIXEL


def test_compact_train_selected_builds_locally_then_publishes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source" / "patient_7.h5"
    compact_dir = tmp_path / "mounted_drive" / "compact"
    local_work_dir = tmp_path / "fast_disk" / "compact_build"
    _write_source_shard(source_path)

    result = build_compact_train_selected_from_decisions(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        decisions=_selected_decisions(source_path),
        compact_dir=compact_dir,
        local_work_dir=local_work_dir,
        compression="none",
    )

    compact_shard = next((compact_dir / "shards").glob("*.h5"))
    assert result.compact_dir == compact_dir
    assert result.index_path == compact_dir / "index.sqlite"
    assert result.summary_path == compact_dir / "summary.json"
    assert compact_shard.exists()
    assert not list(local_work_dir.glob("*.tmp"))
    assert not (compact_dir / ".source_staging").exists()
    with h5py.File(compact_shard, "r") as handle:
        assert handle.attrs["source_hdf5_path"] == str(source_path)


def test_compact_train_selected_remaps_records_and_fails_when_missing(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source" / "patient_7.h5"
    compact_dir = tmp_path / "compact"
    _write_source_shard(source_path)
    build_compact_train_selected_from_decisions(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        decisions=_selected_decisions(source_path),
        compact_dir=compact_dir,
        compression="none",
    )
    selected_record = CanonicalRowRecord(
        source_hdf5_path=source_path,
        source_row_index=2,
        patient_id="7",
        label=0,
        filename="p7_2.png",
        split="TRAIN",
        normalization_method=None,
        normalization_artifact_id=None,
        sampling_decision="protected_kept",
        is_stage7_selected=True,
    )

    remapped = remap_records_to_compact_train_selected(
        [selected_record],
        compact_dir=compact_dir,
    )

    assert remapped.provenance["mapped_row_count"] == 1
    assert remapped.records[0].source_hdf5_path.parent == compact_dir / "shards"
    assert remapped.records[0].source_row_index == 1
    assert remapped.records[0].filename == selected_record.filename

    missing_record = CanonicalRowRecord(
        source_hdf5_path=source_path,
        source_row_index=1,
        patient_id="7",
        label=1,
        filename="p7_1.png",
        split="TRAIN",
        normalization_method=None,
        normalization_artifact_id=None,
        sampling_decision="rejected_reducible",
        is_stage7_selected=False,
    )
    with pytest.raises(FileNotFoundError, match="Rerun Stage 6"):
        remap_records_to_compact_train_selected([missing_record], compact_dir=compact_dir)
