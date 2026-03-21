from __future__ import annotations

from pathlib import Path

from helpers.ensemble_inference.config import load_ensemble_inference_config


def test_load_ensemble_inference_config_uses_defaults(tmp_path: Path) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
    }

    config = load_ensemble_inference_config(env)

    assert config.recipe_path == tmp_path / "recipe.json"
    assert config.hdf5_drive_dir == tmp_path / "dataset"
    assert config.output_dir is None
    assert config.stage_input_locally is True
    assert config.batch_size == 32
    assert config.visualization_samples == 5
    assert config.export_latex is True
    assert config.log_path == Path("logs/ensemble_inference.log")
