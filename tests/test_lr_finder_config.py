from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.lr_finder.config import load_lr_finder_config


def test_load_lr_finder_config_builds_model_plan_from_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "fpn": {"lr": 3e-4, "wd": 1e-4, "encoders": ["senet154"]},
                "segformer": {"lr": 1e-3, "wd": 1e-4, "encoders": ["mit_b5", "mit_b4"]},
            }
        ),
        encoding="utf-8",
    )
    hdf5_dir = tmp_path / "h5"
    hdf5_dir.mkdir()
    output_dir = tmp_path / "reports"
    local_dir = tmp_path / "local"

    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))
    config = load_lr_finder_config(
        {
            "LR_FINDER_HDF5_DRIVE_DIR": str(hdf5_dir),
            "LR_FINDER_OUTPUT_DIR": str(output_dir),
            "LR_FINDER_LOCAL_DATA_DIR": str(local_dir),
            "LR_FINDER_ARCHITECTURES": " SEGFORMER , FPN ",
            "LR_FINDER_NUM_LHS_SAMPLES": "4",
            "LR_FINDER_NUM_REPEATS": "2",
        }
    )

    assert config.hdf5_drive_dir == hdf5_dir
    assert config.output_dir == output_dir
    assert config.local_data_dir == local_dir
    assert config.num_lhs_samples == 4
    assert config.num_repeats == 2
    assert config.log_path == Path("logs/lr_finder.log")
    assert config.execution_mode == "PAPER"
    assert [(plan.architecture, plan.encoder) for plan in config.model_plans] == [
        ("FPN", "senet154"),
        ("SEGFORMER", "mit_b5"),
        ("SEGFORMER", "mit_b4"),
    ]


def test_load_lr_finder_config_rejects_unknown_architecture_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"FPN": {"lr": 3e-4, "wd": 1e-4, "encoders": ["senet154"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))

    with pytest.raises(ValueError, match="Unknown LR Finder architectures"):
        load_lr_finder_config(
            {
                "LR_FINDER_HDF5_DRIVE_DIR": str(tmp_path / "h5"),
                "LR_FINDER_OUTPUT_DIR": str(tmp_path / "reports"),
                "LR_FINDER_ARCHITECTURES": "FPN,UNKNOWN",
            }
        )
