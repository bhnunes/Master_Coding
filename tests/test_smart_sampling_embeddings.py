# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.embeddings import EmbeddingExtractor, H5PatchDataset


def _build_config(tmp_path: Path) -> SmartSamplerConfig:
    return SmartSamplerConfig(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "out",
        output_filename="TRAIN_FILTERED.h5",
        local_work_dir=None,
        stage_input_locally=False,
        stage_outputs_locally=False,
        clean_local_work_dir=True,
        write_sidecars=True,
        overwrite_output=True,
        model_name="owkin/phikon-v2",
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
        adaptive_keep_enabled=True,
        keep_min=2,
        keep_step=1,
        keep_improvement_threshold=0.02,
        keep_patience=2,
        m_max=4,
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

    dataset = H5PatchDataset(str(h5_path), np.array([0], dtype=np.int64))
    item = dataset[0]

    assert tuple(item.shape) == (4, 4, 3)
    assert item.dtype == np.uint8


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

    dataset = H5PatchDataset(str(h5_path), np.array([0, 1], dtype=np.int64))

    _ = dataset[0]
    _ = dataset[1]
    dataset.close()

    assert open_calls == 1


def test_embedding_extractor_init_uses_hugging_face_processor_and_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)

    class FakeProcessorFactory:
        def __init__(self) -> None:
            self.requested_model_names: list[str] = []

        def from_pretrained(self, model_name: str) -> object:
            self.requested_model_names.append(model_name)
            return object()

    class FakeModel:
        def __init__(self) -> None:
            self.device_seen: object | None = None
            self.eval_called = False

        def to(self, device: object) -> FakeModel:
            self.device_seen = device
            return self

        def eval(self) -> FakeModel:
            self.eval_called = True
            return self

    class FakeModelFactory:
        def __init__(self, model: FakeModel) -> None:
            self.model = model
            self.requested_model_names: list[str] = []

        def from_pretrained(self, model_name: str) -> FakeModel:
            self.requested_model_names.append(model_name)
            return self.model

    processor_factory = FakeProcessorFactory()
    model = FakeModel()
    model_factory = FakeModelFactory(model)
    monkeypatch.setattr("helpers.smart_sampling.embeddings.AutoImageProcessor", processor_factory)
    monkeypatch.setattr("helpers.smart_sampling.embeddings.AutoModel", model_factory)

    extractor = EmbeddingExtractor(config)

    assert extractor.processor is not None
    assert cast(Any, extractor.model) is model
    assert processor_factory.requested_model_names == [config.model_name]
    assert model_factory.requested_model_names == [config.model_name]
    assert model.device_seen == extractor.device
    assert model.eval_called is True


def test_embedding_extractor_raises_clear_error_when_model_init_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)

    class FakeProcessorFactory:
        def from_pretrained(self, model_name: str) -> object:
            return object()

    class BrokenModelFactory:
        def from_pretrained(self, model_name: str) -> object:
            raise OSError("download failed")

    monkeypatch.setattr(
        "helpers.smart_sampling.embeddings.AutoImageProcessor", FakeProcessorFactory()
    )
    monkeypatch.setattr("helpers.smart_sampling.embeddings.AutoModel", BrokenModelFactory())

    with pytest.raises(RuntimeError, match="Failed to initialize Stage 6 model"):
        EmbeddingExtractor(config)


def test_get_embeddings_returns_empty_array_for_empty_indices(tmp_path: Path) -> None:
    extractor = cast(Any, object.__new__(EmbeddingExtractor))
    extractor.config = _build_config(tmp_path)
    extractor.device = torch.device("cpu")
    extractor.processor = object()
    extractor.model = object()

    embeddings = EmbeddingExtractor.get_embeddings(
        extractor, "/tmp/none.h5", np.array([], dtype=np.int64)
    )

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


def test_get_embeddings_stacks_cls_batches_and_uses_cpu_pin_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = tmp_path / "patches.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))

    class FakeBatch(dict[str, torch.Tensor]):
        def items(self) -> Any:
            return super().items()

    class FakeProcessor:
        def __call__(
            self, *, images: list[npt.NDArray[np.uint8]], return_tensors: str
        ) -> FakeBatch:
            assert return_tensors == "pt"
            return FakeBatch(
                pixel_values=torch.zeros((len(images), 3, 224, 224), dtype=torch.float32)
            )

    class FakeOutput:
        def __init__(self, batch_size: int) -> None:
            self.last_hidden_state = torch.arange(
                batch_size * 3 * 4,
                dtype=torch.float32,
            ).reshape(batch_size, 3, 4)

    class FakeModel:
        def __call__(self, **kwargs: torch.Tensor) -> FakeOutput:
            batch_size = kwargs["pixel_values"].shape[0]
            return FakeOutput(batch_size)

    observed: dict[str, Any] = {}

    def fake_dataloader(_dataset: object, **kwargs: object) -> list[list[npt.NDArray[np.uint8]]]:
        observed.update(kwargs)
        return [
            [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)],
            [np.zeros((4, 4, 3), dtype=np.uint8)],
        ]

    extractor = cast(Any, object.__new__(EmbeddingExtractor))
    extractor.config = _build_config(tmp_path)
    extractor.device = torch.device("cpu")
    extractor.processor = FakeProcessor()
    extractor.model = FakeModel()

    monkeypatch.setattr("helpers.smart_sampling.embeddings.DataLoader", fake_dataloader)

    embeddings = EmbeddingExtractor.get_embeddings(
        extractor,
        str(h5_path),
        np.array([0, 1, 2], dtype=np.int64),
    )

    assert observed["pin_memory"] is False
    assert callable(observed["collate_fn"])
    assert embeddings.shape == (3, 4)
    assert embeddings.dtype == np.float32
