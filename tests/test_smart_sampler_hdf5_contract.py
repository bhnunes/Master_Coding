from pathlib import Path

import h5py
import numpy as np

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.writer import write_filtered_hdf5


def test_write_filtered_hdf5_preserves_plural_filenames_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 2], dtype=np.int32))
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_1_a.png", b"PATIENT_2_b.png"], dtype="S32"),
        )

    config = SmartSamplerConfig(
        source_h5_path=source_path,
        output_dir=output_dir,
        output_filename="TRAIN_FILTERED.h5",
        local_work_dir=None,
        stage_input_locally=False,
        stage_outputs_locally=False,
        clean_local_work_dir=True,
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

    write_filtered_hdf5(config, np.array([1], dtype=np.int64))

    with h5py.File(output_dir / "TRAIN_FILTERED.h5", "r") as handle:
        assert "filenames" in handle
        assert "filename" not in handle
        assert handle["filenames"][0].decode("utf-8") == "PATIENT_2_b.png"
