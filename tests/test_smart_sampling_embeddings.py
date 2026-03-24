# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.embeddings import (
    EmbeddingExtractor,
    H5PatchDataset,
    get_preprocessing_transforms,
)


def _build_config(tmp_path: Path) -> SmartSamplerConfig:
    return SmartSamplerConfig(
        source_h5_path=tmp_path / "TRAIN.h5",
        output_dir=tmp_path / "out",
        output_filename="TRAIN_FILTERED.h5",
        local_work_dir=None,
        stage_input_locally=False,
        write_sidecars=True,
        overwrite_output=True,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        input_size=16,
        batch_size=2,
        device="cpu",
        n_start=4,
        n_max=4,
        growth_factor=2.0,
        stability_threshold=0.8,
        stability_repeats=2,
        max_steps=2,
        intersection_ratio_threshold=0.2,
        k_min=2,
        k_max=4,
        m_max=4,
        selection_strategy="uniform",
        seed=42,
        num_workers=0,
    )


def _write_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "images", data=np.arange(2 * 3 * 4 * 4, dtype=np.uint8).reshape(2, 3, 4, 4)
        )


def test_h5_patch_dataset_transposes_chw_images(tmp_path: Path) -> None:
    h5_path = tmp_path / "patches.h5"
    _write_h5(h5_path)
    seen_shapes: list[tuple[int, ...]] = []

    def transform(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        seen_shapes.append(tuple(image.shape))
        return image

    dataset = H5PatchDataset(
        str(h5_path), np.array([0], dtype=np.int64), transform=cast(Any, transform)
    )
    item = dataset[0]

    assert seen_shapes == [(4, 4, 3)]
    assert tuple(item.shape) == (4, 4, 3)


def test_h5_patch_dataset_reuses_single_hdf5_handle_per_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = tmp_path / "patches.h5"
    _write_h5(h5_path)
    open_calls = 0
    original_file = h5py.File

    def counting_file(*args: object, **kwargs: object) -> h5py.File:
        nonlocal open_calls
        open_calls += 1
        return original_file(*args, **kwargs)

    monkeypatch.setattr("helpers.smart_sampling.embeddings.h5py.File", counting_file)

    dataset = H5PatchDataset(
        str(h5_path), np.array([0, 1], dtype=np.int64), transform=cast(Any, lambda image: image)
    )

    _ = dataset[0]
    _ = dataset[1]
    dataset.close()

    assert open_calls == 1


def test_get_preprocessing_transforms_returns_tensor_output() -> None:
    transform = get_preprocessing_transforms(8)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 8, 8)


def test_embedding_extractor_init_uses_encoder_and_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)

    class FakeEncoder:
        def __init__(self) -> None:
            self.device: object | None = None
            self.eval_called = False

        def to(self, device: object) -> FakeEncoder:
            self.device = device
            return self

        def eval(self) -> None:
            self.eval_called = True

    encoder = FakeEncoder()
    fake_smp = SimpleNamespace(Unet=lambda **kwargs: SimpleNamespace(encoder=encoder))
    import sys

    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake_smp)

    extractor = EmbeddingExtractor(config)

    assert cast(Any, extractor.model) is encoder
    assert encoder.device == extractor.device
    assert encoder.eval_called is True


def test_get_embeddings_returns_empty_array_for_empty_indices(tmp_path: Path) -> None:
    extractor = cast(Any, object.__new__(EmbeddingExtractor))
    extractor.config = _build_config(tmp_path)
    extractor.device = torch.device("cpu")
    extractor.model = object()
    extractor.transform = lambda image: image

    embeddings = EmbeddingExtractor.get_embeddings(
        extractor, "/tmp/none.h5", np.array([], dtype=np.int64)
    )

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


def test_get_embeddings_stacks_batches_and_uses_cpu_pin_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = tmp_path / "patches.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))

    class FakeEncoder:
        def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
            batch_size = batch.shape[0]
            features = torch.arange(batch_size * 8, dtype=torch.float32).reshape(
                batch_size, 2, 2, 2
            )
            return [features]

    observed: dict[str, Any] = {}

    def fake_dataloader(_dataset: object, **kwargs: object) -> list[torch.Tensor]:
        observed.update(kwargs)
        return [
            torch.zeros((2, 3, 4, 4), dtype=torch.float32),
            torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        ]

    extractor = cast(Any, object.__new__(EmbeddingExtractor))
    extractor.config = _build_config(tmp_path)
    extractor.device = torch.device("cpu")
    extractor.model = FakeEncoder()
    extractor.transform = lambda image: torch.from_numpy(np.moveaxis(image, -1, 0)).float()

    monkeypatch.setattr("helpers.smart_sampling.embeddings.DataLoader", fake_dataloader)

    embeddings = EmbeddingExtractor.get_embeddings(
        extractor,
        str(h5_path),
        np.array([0, 1, 2], dtype=np.int64),
    )

    assert observed["pin_memory"] is False
    assert embeddings.shape == (3, 2)
    assert embeddings.dtype == np.float32
