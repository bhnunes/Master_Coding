from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.training.registry import (
    DEFAULT_MODEL_REGISTRY_PATH,
    get_model_registry_path,
    load_training_model_registry,
    validate_architecture_encoder_pair,
)


def test_get_model_registry_path_returns_default_when_override_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINING_MODEL_REGISTRY_PATH", raising=False)

    assert get_model_registry_path() == DEFAULT_MODEL_REGISTRY_PATH


def test_get_model_registry_path_uses_trimmed_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "custom.json"
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", f"  {registry_path}  ")

    assert get_model_registry_path() == registry_path


def test_load_training_model_registry_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(tmp_path / "missing.json"))

    with pytest.raises(ValueError, match="not found"):
        load_training_model_registry()


def test_load_training_model_registry_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "invalid.json"
    registry_path.write_text("{not-json}", encoding="utf-8")
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    with pytest.raises(ValueError, match="invalid JSON"):
        load_training_model_registry()


def test_load_training_model_registry_rejects_non_object_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "invalid.json"
    registry_path.write_text(json.dumps(["FPN"]), encoding="utf-8")
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_training_model_registry()


def test_load_training_model_registry_rejects_non_string_encoders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "invalid.json"
    registry_path.write_text(
        json.dumps({"FPN": {"lr": 0.1, "wd": 0.01, "encoders": ["resnet34", 5]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    with pytest.raises(ValueError, match="list of strings"):
        load_training_model_registry()


def test_load_training_model_registry_normalizes_keys_and_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "valid.json"
    registry_path.write_text(
        json.dumps({"fpn": {"lr": 0.123, "wd": 0.456, "encoders": ["resnet34"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    registry = load_training_model_registry()

    assert registry == {"FPN": {"lr": 0.123, "wd": 0.456, "encoders": ("resnet34",)}}


def test_validate_architecture_encoder_pair_rejects_unapproved_encoder() -> None:
    with pytest.raises(ValueError, match="TRAINING_ENCODER must be one of"):
        validate_architecture_encoder_pair("SEGFORMER", "resnet34")


def test_validate_architecture_encoder_pair_accepts_approved_encoder() -> None:
    validate_architecture_encoder_pair("SEGFORMER", "mit_b5")
