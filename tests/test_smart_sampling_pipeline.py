from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.pipeline import run_smart_sampling_pipeline
from helpers.training import data as training_data
from helpers.training.data import HybridProstateDataset


def _write_training_hdf5(path: Path) -> None:
    images = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    masks = np.zeros((6, 4, 4), dtype=np.uint8)
    masks[1] = 1
    masks[4] = 1
    labels = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
    patient_ids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    filenames = np.array(
        [
            b"PATIENT_1_a.png",
            b"PATIENT_1_b.png",
            b"PATIENT_1_c.png",
            b"PATIENT_2_a.png",
            b"PATIENT_2_b.png",
            b"PATIENT_2_c.png",
        ],
        dtype="S32",
    )

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("patient_ids", data=patient_ids)
        handle.create_dataset("filenames", data=filenames)


class _DummyEmbeddingExtractor:
    def __init__(self, config: SmartSamplerConfig) -> None:
        self.config = config

    def get_embeddings(
        self, _h5_path: str, indices: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        values = indices.astype(np.float32).reshape(-1, 1)
        return np.concatenate([values, values + 0.5], axis=1)


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(mask),
        }


def test_run_smart_sampling_pipeline_produces_training_compatible_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    _write_training_hdf5(source_path)
    monkeypatch.setattr(
        training_data, "get_transforms", lambda mode, img_size: _IdentityTransform()
    )

    config = SmartSamplerConfig(
        source_h5_path=source_path,
        output_dir=tmp_path / "out",
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

    outputs = run_smart_sampling_pipeline(
        config,
        extractor_factory=_DummyEmbeddingExtractor,
    )

    assert outputs.filtered_h5_path == tmp_path / "out" / "TRAIN_FILTERED.h5"
    assert outputs.selection_csv_path is not None
    assert outputs.stats_csv_path is not None
    assert outputs.run_config_path is not None

    with h5py.File(outputs.filtered_h5_path, "r") as handle:
        assert set(handle.keys()) == {"filenames", "images", "labels", "masks", "patient_ids"}
        assert handle["patient_ids"][:].tolist() == [1, 1, 2, 2]

    dataset = HybridProstateDataset(str(outputs.filtered_h5_path), mode="train")
    assert len(dataset) == 4
