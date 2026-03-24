from pathlib import Path

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.io import (
    move_file,
    process_and_write_image,
    process_and_write_split_files,
    verify_split_hdf5_integrity,
    verify_split_integrity,
    write_split_hdf5,
)


def test_process_and_write_split_files_preserves_sources_by_default_in_not_normalized_mode(
    tmp_path: Path,
) -> None:
    image_src = tmp_path / "src_image.png"
    mask_src = tmp_path / "src_mask.png"
    cv2.imwrite(str(image_src), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_src), np.zeros((4, 4), dtype=np.uint8))
    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "image_path": str(image_src),
                "mask_path": str(mask_src),
            }
        ]
    )

    process_and_write_split_files(
        split_df=split_df,
        output_dir=tmp_path / "output",
        split_name="TRAIN",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
    )

    assert image_src.exists()
    assert mask_src.exists()
    assert (tmp_path / "output" / "TRAIN" / "CANCER" / "PATIENT_1_PATCH_001.png").is_file()
    assert (tmp_path / "output" / "TRAIN" / "CANCER_MASK" / "PATIENT_1_PATCH_001.png").is_file()


def test_process_and_write_split_files_moves_files_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    image_src = tmp_path / "src_image.png"
    mask_src = tmp_path / "src_mask.png"
    image_src.write_bytes(b"image")
    mask_src.write_bytes(b"mask")
    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
                "image_path": str(image_src),
                "mask_path": str(mask_src),
            }
        ]
    )

    process_and_write_split_files(
        split_df=split_df,
        output_dir=tmp_path / "output",
        split_name="TRAIN",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        allow_destructive_move=True,
    )

    assert not image_src.exists()
    assert not mask_src.exists()
    assert (tmp_path / "output" / "TRAIN" / "CANCER" / "PATIENT_1_PATCH_001.png").is_file()
    assert (tmp_path / "output" / "TRAIN" / "CANCER_MASK" / "PATIENT_1_PATCH_001.png").is_file()


def test_verify_split_integrity_raises_when_mask_filenames_do_not_match(tmp_path: Path) -> None:
    image_dir = tmp_path / "TRAIN" / "CANCER"
    mask_dir = tmp_path / "TRAIN" / "CANCER_MASK"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "a.png").write_bytes(b"image")
    (mask_dir / "b.png").write_bytes(b"mask")

    with pytest.raises(ValueError, match="Integrity FAILED"):
        verify_split_integrity(tmp_path, "TRAIN")


def test_process_and_write_image_applies_normalizer_when_provided(tmp_path: Path) -> None:
    image_src = tmp_path / "src.png"
    cv2.imwrite(str(image_src), np.zeros((4, 4, 3), dtype=np.uint8))
    image_dest = tmp_path / "dest.png"

    class FakeNormalizer:
        def transform(self, image_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
            return (image_rgb + 10).astype(np.uint8)

    success, error = process_and_write_image(str(image_src), str(image_dest), FakeNormalizer())

    assert success is True
    assert error is None
    assert image_dest.is_file()


def test_process_and_write_image_copies_file_without_decoding_when_normalizer_is_absent(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    image_src = tmp_path / "src.png"
    image_src.write_bytes(b"raw-png-bytes")
    image_dest = tmp_path / "dest.png"

    monkeypatch.setattr(
        cv2, "imread", lambda *_args, **_kwargs: pytest.fail("imread should not be used")
    )

    success, error = process_and_write_image(str(image_src), str(image_dest), None)

    assert success is True
    assert error is None
    assert image_dest.read_bytes() == b"raw-png-bytes"


def test_move_file_uses_shutil_fallback_when_replace_fails(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "nested" / "dest.txt"
    source.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(Path, "replace", lambda self, target: (_ for _ in ()).throw(OSError("x")))

    success, error, used_fallback = move_file(str(source), str(destination))

    assert success is True
    assert error is None
    assert used_fallback is True
    assert destination.read_text(encoding="utf-8") == "hello"


def test_verify_split_integrity_skips_missing_split_dir(tmp_path: Path) -> None:
    verify_split_integrity(tmp_path, "TRAIN")


def test_write_split_hdf5_writes_split_contract_from_source_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.array(
                [np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4, 3), 20, dtype=np.uint8)]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.array([np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]),
        )
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([10, 20], dtype=np.int32))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_10_PATCH_001.png", b"PATIENT_20_PATCH_001.png"]),
        )

    split_df = pd.DataFrame(
        [
            {
                "label": 1,
                "patient_id": 20,
                "filename": "PATIENT_20_PATCH_001.png",
                "source_row_index": 1,
            }
        ]
    )

    output_path = write_split_hdf5(
        split_df=split_df,
        source_hdf5_path=source_path,
        output_path=tmp_path / "TRAIN.h5",
        normalizer=None,
        normalization_method="NOT_NORMALIZED",
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["labels"][:].tolist() == [1]
        assert handle["patient_ids"][:].tolist() == [20]
        assert handle["filenames"][:].tolist() == [b"PATIENT_20_PATCH_001.png"]

    verify_split_hdf5_integrity(output_path, split_df)
