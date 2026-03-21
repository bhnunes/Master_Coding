from __future__ import annotations

import types
from pathlib import Path

import pytest

from helpers.artifact.config import ArtifactDetectionConfig
from helpers.artifact.model_loader import ArtifactModelLoader


class FakeTissueModel:
    def __init__(self) -> None:
        self.loaded_state_dict: object | None = None
        self.device: object | None = None
        self.eval_called = False

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state_dict = state_dict

    def to(self, device: object) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_called = True


def _build_config(tmp_path: Path) -> ArtifactDetectionConfig:
    return ArtifactDetectionConfig(
        images_zip=tmp_path / "slides.zip",
        geojson_output=tmp_path / "geojson",
        database_folder=tmp_path / "db",
        database_path=tmp_path / "db" / "artifact.db",
        temp_root=tmp_path / "temp",
        log_folder=tmp_path / "logs",
        log_file_name="artifact.log",
        device="cpu",
        tissue_detector_model_dir=tmp_path / "models" / "td",
        tissue_detector_model_name="td_model.pth",
        qc_model_dir=tmp_path / "models" / "qc",
        mpp_model=1.5,
    )


def test_model_loader_loads_models_once_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)
    tissue_model = FakeTissueModel()
    torch_load_calls: list[tuple[object, object, bool]] = []

    def fake_torch_load(path: object, map_location: object, weights_only: bool) -> object:
        torch_load_calls.append((path, map_location, weights_only))
        if len(torch_load_calls) == 1:
            return {"weights": 1}
        return {"qc": 2}

    fake_smp = types.SimpleNamespace(
        encoders=types.SimpleNamespace(
            get_preprocessing_fn=lambda encoder, weights: (encoder, weights, "preprocess")
        ),
        UnetPlusPlus=lambda **kwargs: tissue_model,
    )
    fake_torch = types.SimpleNamespace(load=fake_torch_load)

    monkeypatch.setitem(__import__("sys").modules, "segmentation_models_pytorch", fake_smp)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    loader = ArtifactModelLoader(config)

    first = loader.load()
    second = loader.load()

    assert first is second
    assert first.tissue_model is tissue_model
    assert first.preprocessing_fn == (
        config.encoder_model_td,
        config.encoder_weights_td,
        "preprocess",
    )
    assert first.qc_model == {"qc": 2}
    assert tissue_model.loaded_state_dict == {"weights": 1}
    assert tissue_model.device == config.device
    assert tissue_model.eval_called is True
    assert torch_load_calls == [
        (
            str(config.tissue_detector_model_dir / config.tissue_detector_model_name),
            "cpu",
            False,
        ),
        (config.qc_model_dir / config.qc_model_name, config.device, False),
    ]
