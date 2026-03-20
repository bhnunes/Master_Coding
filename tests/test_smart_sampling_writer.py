from pathlib import Path

import h5py
import numpy as np

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.writer import write_filtered_hdf5


def _write_source_hdf5(path: Path, *, filename_dataset_name: str = "filenames") -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((3, 4, 4), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 1, 2], dtype=np.int32))
        handle.create_dataset("labels", data=np.array([0, 1, 1], dtype=np.uint8))
        handle.create_dataset(
            filename_dataset_name,
            data=np.array(
                [b"PATIENT_1_a.png", b"PATIENT_1_b.png", b"PATIENT_2_a.png"],
                dtype="S32",
            ),
        )


def test_write_filtered_hdf5_writes_plural_filenames_for_legacy_input(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path, filename_dataset_name="filename")

    config = SmartSamplerConfig(
        source_h5_path=source_path,
        output_dir=output_dir,
        output_filename="TRAIN_FILTERED.h5",
        local_work_dir=None,
        stage_input_locally=False,
        write_sidecars=True,
        overwrite_output=True,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        input_size=224,
        batch_size=8,
        device="cpu",
        n_start=8,
        n_max=8,
        growth_factor=2.0,
        stability_threshold=0.85,
        stability_repeats=2,
        max_steps=2,
        intersection_ratio_threshold=0.2,
        k_min=20,
        k_max=80,
        m_max=2,
        selection_strategy="uniform",
        seed=42,
        num_workers=0,
    )

    result = write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))

    with h5py.File(result, "r") as handle:
        assert set(handle.keys()) == {"filenames", "images", "labels", "masks", "patient_ids"}
        assert handle["filenames"][0].decode("utf-8") == "PATIENT_1_b.png"
        assert handle["patient_ids"][:].tolist() == [1, 2]
