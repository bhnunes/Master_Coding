from __future__ import annotations

from pathlib import Path

from helpers.ensemble_inference.config import load_ensemble_inference_config

DEFAULT_BATCH_SIZE = 32
DEFAULT_VISUALIZATION_SAMPLES = 5
CUSTOM_PATCH_AREA_FRACTION_THRESHOLD = 0.125


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
    assert config.export_visualizations is False
    assert config.export_latex is True
    assert config.patch_positive_area_fraction_threshold == 0.0
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"
    assert config.log_path == Path("logs/ensemble_inference.log")


def test_load_ensemble_inference_config_reads_shared_runtime_normalization_method(
    tmp_path: Path,
) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
        "RUNTIME_NORMALIZATION_METHOD": "reinhard",
    }

    config = load_ensemble_inference_config(env)

    assert config.runtime_normalization_method == "REINHARD"


def test_load_ensemble_inference_config_reads_runtime_vahadane_backend(
    tmp_path: Path,
) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
        "RUNTIME_VAHADANE_BACKEND": "torch_staintools_exact",
    }

    config = load_ensemble_inference_config(env)

    assert config.runtime_vahadane_backend == "torch_staintools_exact"


def test_load_ensemble_inference_config_reads_visualization_export_flag(
    tmp_path: Path,
) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
        "ENSEMBLE_INFER_EXPORT_VISUALIZATIONS": "True",
    }

    config = load_ensemble_inference_config(env)

    assert config.export_visualizations is True


def test_load_ensemble_inference_config_reads_patch_area_fraction_threshold(
    tmp_path: Path,
) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
        "ENSEMBLE_INFER_PATCH_POSITIVE_AREA_FRACTION_THRESHOLD": str(
            CUSTOM_PATCH_AREA_FRACTION_THRESHOLD
        ),
    }

    config = load_ensemble_inference_config(env)

    assert config.patch_positive_area_fraction_threshold == CUSTOM_PATCH_AREA_FRACTION_THRESHOLD


def test_load_ensemble_inference_config_rejects_invalid_patch_area_threshold(
    tmp_path: Path,
) -> None:
    env = {
        "ENSEMBLE_INFER_RECIPE_PATH": str(tmp_path / "recipe.json"),
        "ENSEMBLE_INFER_MASTER_MANIFEST_PATH": str(tmp_path / "master_manifest.sqlite"),
        "ENSEMBLE_INFER_PATCH_POSITIVE_AREA_FRACTION_THRESHOLD": "1.5",
    }

    try:
        load_ensemble_inference_config(env)
    except ValueError as error:
        assert "ENSEMBLE_INFER_PATCH_POSITIVE_AREA_FRACTION_THRESHOLD" in str(error)
    else:
        raise AssertionError("expected invalid patch-area threshold to fail")
