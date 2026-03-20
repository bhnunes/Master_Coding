from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt

from helpers.packaging.discovery import SampleRecord


def _load_image(image_path: Path, img_size: int) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
    )


def _load_mask(mask_path: Path, img_size: int) -> npt.NDArray[np.uint8]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask file: {mask_path}")
    resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return cast(npt.NDArray[np.uint8], np.where(resized > 0, 1, 0).astype(np.uint8))


def write_split_hdf5(
    output_path: Path,
    samples: list[SampleRecord],
    *,
    img_size: int,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    if not samples:
        raise ValueError(f"Cannot create '{output_path.name}' without any samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        images = handle.create_dataset(
            "images",
            shape=(len(samples), img_size, img_size, 3),
            dtype="uint8",
        )
        masks = handle.create_dataset(
            "masks",
            shape=(len(samples), img_size, img_size),
            dtype="uint8",
        )
        labels = handle.create_dataset("labels", shape=(len(samples),), dtype="uint8")
        patient_ids = handle.create_dataset("patient_ids", shape=(len(samples),), dtype="int32")
        filenames = handle.create_dataset("filenames", shape=(len(samples),), dtype=str_dtype)

        for index, sample in enumerate(samples):
            images[index] = _load_image(sample.image_path, img_size)
            masks[index] = _load_mask(sample.mask_path, img_size)
            labels[index] = sample.label
            patient_ids[index] = sample.patient_id
            filenames[index] = sample.filename

    return output_path
