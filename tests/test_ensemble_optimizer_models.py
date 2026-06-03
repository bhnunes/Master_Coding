# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from helpers.ensemble_optimizer.metadata import SelectedModelMetadata
from helpers.ensemble_optimizer.models import (
    build_loaded_model_info,
    load_checkpoint_strict_without_aux,
    load_ensemble_models,
    load_single_model,
)

MODEL_LEARNING_RATE = 1e-3
MIN_LOADED_MODEL_COUNT = 2


class FakeModel:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self.loaded_state_dict: dict[str, Any] | None = None
        self.strict: bool | None = None
        self.device: object | None = None
        self.eval_called = False

    def state_dict(self) -> dict[str, int]:
        return {key: 1 for key in self._keys}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool) -> None:
        self.loaded_state_dict = state_dict
        self.strict = strict

    def to(self, device: object) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True


def _selected_model() -> SelectedModelMetadata:
    return SelectedModelMetadata(
        architecture="UNET",
        encoder="resnet34",
        checkpoint_path="/tmp/model.ckpt",
        metadata_filename="model_meta.json",
        sort_metric_value=0.9,
        raw_metadata={
            "best_model_epoch": 7,
            "best_val_auprc_pixel_score": 0.88,
            "validation_monitoring_threshold_pixel_level": 0.42,
            "hyperparameters": {
                "Learning_rate": MODEL_LEARNING_RATE,
                "Batch_Size": 8,
                "Weight_Decay": 1e-4,
                "Optimizer": "AdamW",
                "Seed": 13,
                "Loss_Function": "Dice",
            },
        },
    )


def test_load_checkpoint_strict_without_aux_strips_aux_keys_and_moves_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(["encoder.weight", "decoder.bias"])
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.torch.load",
        lambda path, map_location: {
            "model_state_dict": {
                "encoder.weight": 10,
                "classification_head.weight": 99,
            }
        },
    )

    loaded = load_checkpoint_strict_without_aux(
        cast(Any, model), "/tmp/model.ckpt", device=cast(Any, "cpu")
    )

    assert cast(Any, loaded) is model
    assert model.loaded_state_dict == {"encoder.weight": 10}
    assert model.strict is True
    assert model.device == "cpu"


def test_load_checkpoint_strict_without_aux_aligns_orig_mod_prefixes_and_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(["_orig_mod.encoder.weight"])
    compiled_models: list[object] = []

    def fake_compile(model_obj: object) -> object:
        compiled_models.append(model_obj)
        return model_obj

    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.torch.load",
        lambda path, map_location: {
            "model_state_dict": {"encoder.weight": 10},
            "is_compiled": True,
        },
    )
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.torch.compile",
        fake_compile,
    )

    loaded = load_checkpoint_strict_without_aux(
        cast(Any, model), "/tmp/model.ckpt", device=cast(Any, "cpu")
    )

    assert cast(Any, loaded) is model
    assert model.loaded_state_dict == {"_orig_mod.encoder.weight": 10}
    assert compiled_models == [model]


def test_build_loaded_model_info_returns_expected_metadata_fields() -> None:
    info = build_loaded_model_info(_selected_model())

    assert info["architecture"] == "UNET"
    assert info["encoder"] == "resnet34"
    assert info["checkpoint_path"] == "/tmp/model.ckpt"
    assert info["Learning_rate"] == MODEL_LEARNING_RATE
    assert info["Loss_Function"] == "Dice"


def test_load_single_model_builds_model_loads_checkpoint_and_sets_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(["encoder.weight"])
    selected_model = _selected_model()

    monkeypatch.setattr("helpers.ensemble_optimizer.models.create_model", lambda **kwargs: model)
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.load_checkpoint_strict_without_aux",
        lambda loaded_model, checkpoint_path, device: loaded_model,
    )

    loaded = load_single_model(selected_model, device=cast(Any, "cpu"))

    assert cast(Any, loaded) is model
    assert loaded.arch_name == "UNET"
    assert model.eval_called is True


def test_load_ensemble_models_skips_failures_but_requires_two_models(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    selected_a = _selected_model()
    selected_b = SelectedModelMetadata(
        architecture="FPN",
        encoder="resnet18",
        checkpoint_path="/tmp/b.ckpt",
        metadata_filename="b_meta.json",
        sort_metric_value=0.8,
        raw_metadata={},
    )
    selected_c = SelectedModelMetadata(
        architecture="PAN",
        encoder="resnet50",
        checkpoint_path="/tmp/c.ckpt",
        metadata_filename="c_meta.json",
        sort_metric_value=0.7,
        raw_metadata={},
    )

    def fake_load_single_model(selected_model: SelectedModelMetadata, device: object) -> object:
        del device
        if selected_model.checkpoint_path == "/tmp/b.ckpt":
            raise RuntimeError("broken")
        return {"path": selected_model.checkpoint_path}

    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.load_single_model", fake_load_single_model
    )

    caplog.set_level(logging.INFO)

    models, info = load_ensemble_models(
        [selected_a, selected_b, selected_c], device=cast(Any, "cpu")
    )

    assert len(models) == MIN_LOADED_MODEL_COUNT
    assert [entry["checkpoint_path"] for entry in info] == ["/tmp/model.ckpt", "/tmp/c.ckpt"]
    assert "Failed to load model /tmp/b.ckpt: broken" in caplog.text
    assert "Loaded 2/3 selected models (1 failed)." in caplog.text


def test_load_ensemble_models_raises_when_fewer_than_two_models_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "helpers.ensemble_optimizer.models.load_single_model",
        lambda selected_model, device: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    with pytest.raises(RuntimeError, match="Fewer than 2 models loaded"):
        load_ensemble_models([_selected_model()], device=cast(Any, "cpu"))
