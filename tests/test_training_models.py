import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from helpers.training import models as training_models
from helpers.training.models import create_model, create_optimizer, get_learning_rate
from helpers.training.registry import get_supported_encoders


@pytest.mark.parametrize(
    ("architecture", "expected_lr", "expected_weight_decay"),
    [
        ("SWIN", 3e-4, 1e-4),
        ("segformer", 1e-3, 1e-4),
        ("DPT", 3e-4, 1e-4),
        ("UPERNET", 2e-4, 1e-4),
        ("DEEPLABV3PLUS", 1e-3, 1e-4),
        ("UNET++", 5e-4, 1e-4),
        ("FPN", 3e-4, 1e-4),
        ("MANET", 4e-4, 1e-4),
    ],
)
def test_get_learning_rate_returns_existing_architecture_defaults(
    architecture: str,
    expected_lr: float,
    expected_weight_decay: float,
) -> None:
    learning_rate, weight_decay = get_learning_rate(architecture)

    assert learning_rate == expected_lr
    assert weight_decay == expected_weight_decay


def test_get_learning_rate_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_learning_rate("unknown")


def test_get_learning_rate_reads_from_registry_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"FPN": {"lr": 0.123, "wd": 0.456, "encoders": ["senet154"]}})
    )
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    learning_rate, weight_decay = get_learning_rate("FPN")

    assert learning_rate == 0.123
    assert weight_decay == 0.456


def test_get_learning_rate_rejects_malformed_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"FPN": {"lr": 0.123, "encoders": ["senet154"]}}))
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    with pytest.raises(ValueError, match="must define 'lr', 'wd', and 'encoders'"):
        get_learning_rate("FPN")


def test_get_supported_encoders_returns_approved_research_set() -> None:
    encoders = get_supported_encoders("SEGFORMER")

    assert encoders == ("mit_b5",)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def make_builder(self, name: str) -> Any:
        def _builder(**kwargs: object) -> dict[str, object]:
            self.calls.append((name, kwargs))
            return {"name": name, "kwargs": kwargs}

        return _builder


def test_create_model_uses_expected_builder_and_imagenet_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    fake_smp = SimpleNamespace(
        Unet=recorder.make_builder("Unet"),
        DeepLabV3Plus=recorder.make_builder("DeepLabV3Plus"),
        DPT=recorder.make_builder("DPT"),
        UnetPlusPlus=recorder.make_builder("UnetPlusPlus"),
        FPN=recorder.make_builder("FPN"),
        Segformer=recorder.make_builder("Segformer"),
        MAnet=recorder.make_builder("MAnet"),
        UPerNet=recorder.make_builder("UPerNet"),
    )
    monkeypatch.setattr(training_models, "smp", fake_smp)

    model = cast(dict[str, object], create_model("SEGFORMER", "resnet34", validation=False))

    assert model["name"] == "Segformer"
    assert recorder.calls[0][1]["encoder_name"] == "resnet34"
    assert recorder.calls[0][1]["encoder_weights"] == "imagenet"
    assert recorder.calls[0][1]["classes"] == 2
    assert recorder.calls[0][1]["activation"] is None


def test_create_model_disables_encoder_weights_for_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    fake_smp = SimpleNamespace(
        Unet=recorder.make_builder("Unet"),
        DeepLabV3Plus=recorder.make_builder("DeepLabV3Plus"),
        DPT=recorder.make_builder("DPT"),
        UnetPlusPlus=recorder.make_builder("UnetPlusPlus"),
        FPN=recorder.make_builder("FPN"),
        Segformer=recorder.make_builder("Segformer"),
        MAnet=recorder.make_builder("MAnet"),
        UPerNet=recorder.make_builder("UPerNet"),
    )
    monkeypatch.setattr(training_models, "smp", fake_smp)

    create_model("DPT", "mit_b0", validation=True)

    assert recorder.calls[0][1]["encoder_weights"] is None
    assert recorder.calls[0][1]["decoder_readout"] == "ignore"


def test_create_optimizer_supports_adamw() -> None:
    model = torch.nn.Linear(2, 1)

    optimizer = create_optimizer(model, "AdamW", 1e-3, 1e-4)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_create_optimizer_supports_schedulefree(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyScheduleFree:
        def __init__(self, params: Any, lr: float, weight_decay: float) -> None:
            captured["params"] = list(params)
            captured["lr"] = lr
            captured["weight_decay"] = weight_decay

    fake_schedulefree = type("FakeScheduleFree", (), {"AdamWScheduleFree": DummyScheduleFree})()
    monkeypatch.setattr(training_models, "schedulefree", fake_schedulefree)
    model = torch.nn.Linear(2, 1)

    optimizer = create_optimizer(model, "AdamWScheduleFree", 2e-3, 3e-4)

    assert isinstance(optimizer, DummyScheduleFree)
    assert captured["lr"] == 2e-3
    assert captured["weight_decay"] == 3e-4
    params = captured["params"]

    assert isinstance(params, list)
    assert len(params) == 2


def test_create_optimizer_rejects_unknown_name() -> None:
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer(model, "SGD", 1e-3, 1e-4)
