from __future__ import annotations

from pathlib import Path

from helpers.ensemble_inference.config import load_ensemble_inference_config

DEFAULT_BATCH_SIZE = 32
DEFAULT_VISUALIZATION_SAMPLES = 5


def test_load_ensemble_inference_config_uses_defaults(tmp_path: Path) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
    }

    config = load_ensemble_inference_config(env)

    assert config.recipe_path == tmp_path / "recipe.json"
    assert config.master_manifest_path == tmp_path / "master_manifest.sqlite"
    assert config.output_dir is None
    assert config.stage_input_locally is True
    assert config.batch_size == DEFAULT_BATCH_SIZE
    assert config.visualization_samples == DEFAULT_VISUALIZATION_SAMPLES
    assert config.export_latex is True
    assert config.log_path == Path("logs/ensemble_inference.log")
