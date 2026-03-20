from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.io import (
    move_file,
    process_and_write_image,
    process_and_write_split_files,
    verify_split_integrity,
)


def test_process_and_write_split_files_moves_files_in_not_normalized_mode(tmp_path: Path) -> None:
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
