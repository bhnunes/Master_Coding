import sqlite3
from pathlib import Path

import h5py
import numpy as np

from helpers.crossfold.discovery import collect_source_dataset_provenance, load_patch_dataset
from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref


def test_load_patch_dataset_reads_hdf5_source_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
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
            data=np.array(
                [f"{source_path}::images[0]".encode(), f"{source_path}::images[1]".encode()]
            ),
        )
        handle.create_dataset(
            "source_mask_paths",
            data=np.array(
                [f"{source_path}::masks[0]".encode(), f"{source_path}::masks[1]".encode()]
            ),
        )

    dataset = load_patch_dataset(source_path)

    assert dataset[["patient_id", "label", "filename", "source_row_index"]].to_dict("records") == [
        {
            "patient_id": 11,
            "label": 1,
            "filename": "PATIENT_11_PATCH_001.png",
            "source_row_index": 0,
        },
        {
            "patient_id": 22,
            "label": 0,
            "filename": "PATIENT_22_PATCH_001.png",
            "source_row_index": 1,
        },
    ]
    assert dataset["image_path"].tolist() == [
        f"{source_path}::images[0]",
        f"{source_path}::images[1]",
    ]
    assert dataset["mask_path"].tolist() == [
        f"{source_path}::masks[0]",
        f"{source_path}::masks[1]",
    ]


def test_load_patch_dataset_synthesizes_logical_source_refs_when_paths_are_absent(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([11], dtype=np.int32))
        handle.create_dataset("filenames", data=np.array([b"PATIENT_11_PATCH_001.png"]))

    dataset = load_patch_dataset(source_path)

    assert dataset.to_dict("records") == [
        {
            "patient_id": 11,
            "image_path": f"{source_path}::images[0]",
            "mask_path": f"{source_path}::masks[0]",
            "label": 1,
            "filename": "PATIENT_11_PATCH_001.png",
            "source_row_index": 0,
            "source_hdf5_path": str(source_path),
        }
    ]


def test_load_patch_dataset_reads_sqlite_accepted_rows(tmp_path: Path) -> None:
    source_hdf5_path = tmp_path / "PATCHES" / "HDF5_SHARDS" / "slide_a.h5"
    source_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    source_hdf5_path.write_bytes(b"hdf5-placeholder")
    sqlite_path = tmp_path / "master_manifest.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL
            );
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                is_stage4_accepted INTEGER
            );
            """
        )
        cursor = connection.execute(
            "INSERT INTO patches ("
            "source_hdf5_path, source_row_index, filename, patient_id, label, "
            "source_image_path, source_mask_path"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                to_manifest_path_ref(source_hdf5_path, manifest_path=sqlite_path),
                4,
                "accepted.png",
                11,
                1,
                build_hdf5_dataset_ref("images", 4),
                build_hdf5_dataset_ref("masks", 4),
            ),
        )
        connection.execute(
            "INSERT INTO patch_stage_state (patch_id, is_stage4_accepted) VALUES (?, 1)",
            ((cursor.lastrowid or 0),),
        )
        connection.commit()

    dataset = load_patch_dataset(sqlite_path)

    assert dataset.to_dict("records") == [
        {
            "patient_id": 11,
            "image_path": f"{source_hdf5_path}::images[4]",
            "mask_path": f"{source_hdf5_path}::masks[4]",
            "label": 1,
            "filename": "accepted.png",
            "source_row_index": 4,
            "source_hdf5_path": str(source_hdf5_path),
        }
    ]


def test_collect_source_dataset_provenance_reads_sqlite_state(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "master_manifest.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE patch_stage_state ("
            "patch_id INTEGER PRIMARY KEY, is_stage4_accepted INTEGER"
            ")"
        )
        connection.execute(
            "CREATE TABLE patches (patch_id INTEGER PRIMARY KEY, source_signature TEXT)"
        )
        connection.execute(
            "INSERT INTO patch_stage_state (patch_id, is_stage4_accepted) VALUES (1, 1)"
        )
        connection.execute("INSERT INTO patches (patch_id, source_signature) VALUES (1, 'sig-a')")
        connection.commit()

    provenance = collect_source_dataset_provenance(sqlite_path)

    assert provenance["path"] == str(sqlite_path)
    assert provenance["sha256"]
    attrs = provenance["attrs"]
    assert isinstance(attrs, dict)
    assert attrs["stage4_cleaning_manifest_path"] == str(sqlite_path)
    assert attrs["stage4_cleaning_selected_rows"] == 1
