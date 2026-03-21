from pathlib import Path

import cv2
import h5py
import numpy as np

from helpers.packaging.discovery import SampleRecord
from helpers.packaging.writer import write_split_hdf5


def _write_rgb_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    cv2.imwrite(str(path), image)


def _write_mask_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:8] = 255
    cv2.imwrite(str(path), mask)


def test_write_split_hdf5_writes_minimal_training_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_a.png"
    mask_path = tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_a.png"
    _write_rgb_png(image_path)
    _write_mask_png(mask_path)

    output_path = tmp_path / "TRAIN.h5"
    write_split_hdf5(
        output_path,
        [
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                filename="PATIENT_1_a.png",
                label=1,
                patient_id=1,
            )
        ],
        img_size=16,
        overwrite=True,
    )

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {"images", "masks", "labels", "patient_ids", "filenames"}
        assert list(handle.attrs.keys()) == ["source_signature"]
        assert handle["images"].shape == (1, 16, 16, 3)
        assert handle["masks"].shape == (1, 16, 16)
        assert handle["labels"].dtype == np.dtype("uint8")
        assert handle["patient_ids"].shape == (1,)
        assert handle["filenames"][0].decode("utf-8") == "PATIENT_1_a.png"
        assert set(np.unique(handle["masks"][0]).tolist()) <= {0, 1}


def test_write_split_hdf5_reuses_existing_output_when_signature_matches(tmp_path: Path) -> None:
    image_path = tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_a.png"
    mask_path = tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_a.png"
    _write_rgb_png(image_path)
    _write_mask_png(mask_path)

    output_path = tmp_path / "TRAIN.h5"
    write_split_hdf5(
        output_path,
        [
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                filename="PATIENT_1_a.png",
                label=1,
                patient_id=1,
            )
        ],
        img_size=16,
        overwrite=True,
    )

    write_split_hdf5(
        output_path,
        [
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                filename="PATIENT_1_a.png",
                label=1,
                patient_id=1,
            )
        ],
        img_size=16,
        overwrite=False,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle["labels"][0] == 1


def test_write_split_hdf5_rejects_existing_output_when_signature_mismatches(tmp_path: Path) -> None:
    output_path = tmp_path / "TRAIN.h5"
    with h5py.File(output_path, "w") as handle:
        handle.attrs["source_signature"] = "stale"
        handle.create_dataset("labels", data=np.array([9], dtype=np.uint8))

    image_path = tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_a.png"
    mask_path = tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_a.png"
    _write_rgb_png(image_path)
    _write_mask_png(mask_path)

    try:
        write_split_hdf5(
            output_path,
            [
                SampleRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    filename="PATIENT_1_a.png",
                    label=1,
                    patient_id=1,
                )
            ],
            img_size=16,
            overwrite=False,
        )
    except ValueError as error:
        assert "does not match the current split inputs" in str(error)
    else:
        raise AssertionError("Expected stale packaging output to be rejected.")
