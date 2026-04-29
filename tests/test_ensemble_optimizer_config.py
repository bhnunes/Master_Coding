import json
from pathlib import Path

import pytest

from helpers.ensemble_optimizer.config import load_ensemble_optimizer_config

ENSEMBLE_SEED = 17
ENSEMBLE_BATCH_SIZE = 12
ENSEMBLE_WORKERS = 3
VALIDATION_CALIBRATION_FRACTION = 0.25
VALIDATION_HOLDOUT_FRACTION = 0.3
SEMANTIC_TRIALS = 11
SPATIAL_TRIALS = 13


@pytest.fixture(autouse=True)
def _training_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "training_model_registry.json"
    payload = {
        architecture: {
            "lr": 1e-4,
            "wd": 1e-4,
            "encoders": [encoder],
            "loss": {"alpha_bce": 0.1, "beta_dice_bg": 0.2, "gamma_dice_fg": 0.3},
        }
        for architecture, encoder in {
            "SWIN": "enc-swin",
            "DPT": "enc-dpt",
            "SEGFORMER": "enc-segformer",
            "UPERNET": "enc-upernet",
            "DEEPLABV3PLUS": "enc-deeplab",
            "UNET++": "enc-unetpp",
            "FPN": "enc-fpn",
            "MANET": "enc-manet",
        }.items()
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TRAINING_MODEL_REGISTRY_PATH", str(registry_path))


def test_load_ensemble_optimizer_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "ENSEMBLE_OPT_OUTPUT_DIR": str(tmp_path / "reports"),
            "ENSEMBLE_OPT_LOCAL_DATA_DIR": str(tmp_path / "cache"),
            "ENSEMBLE_OPT_PRED_CACHE_DIR": str(tmp_path / "pred_cache"),
            "ENSEMBLE_OPT_SEED": "17",
            "ENSEMBLE_OPT_BATCH_SIZE": "12",
            "ENSEMBLE_OPT_WORKERS": "3",
            "ENSEMBLE_OPT_SORT_METRIC": "best_validation_DICE",
            "ENSEMBLE_OPT_STAGE_INPUT_LOCALLY": "false",
            "ENSEMBLE_OPT_OVERWRITE_OUTPUT": "false",
            "ENSEMBLE_OPT_VAL_CALIBRATION_FRAC": "0.25",
            "ENSEMBLE_OPT_VAL_HOLDOUT_FRAC": "0.3",
            "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES": "swin, segformer",
            "ENSEMBLE_OPT_SPATIAL_ARCHITECTURES": "fpn, manet",
            "ENSEMBLE_OPT_NUM_TRIALS_SEMANTIC": "11",
            "ENSEMBLE_OPT_NUM_TRIALS_SPATIAL": "13",
        }
    )

    assert config.master_manifest_path == tmp_path / "dataset" / "master_manifest.sqlite"
    assert config.metadata_dir == tmp_path / "metadata"
    assert config.output_dir == tmp_path / "reports"
    assert config.local_data_dir == tmp_path / "cache"
    assert config.pred_cache_dir == tmp_path / "pred_cache"
    assert config.seed == ENSEMBLE_SEED
    assert config.batch_size == ENSEMBLE_BATCH_SIZE
    assert config.workers == ENSEMBLE_WORKERS
    assert config.sort_metric == "best_validation_DICE"
    assert config.stage_input_locally is False
    assert config.overwrite_output is False
    assert config.val_calibration_frac == VALIDATION_CALIBRATION_FRACTION
    assert config.val_holdout_frac == VALIDATION_HOLDOUT_FRACTION
    assert config.semantic_architectures == ("SWIN", "SEGFORMER")
    assert config.spatial_architectures == ("FPN", "MANET")
    assert config.num_trials_semantic == SEMANTIC_TRIALS
    assert config.num_trials_spatial == SPATIAL_TRIALS
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"
    assert config.log_path == Path("logs/ensemble_optimizer.log")


def test_load_ensemble_optimizer_config_uses_portable_defaults(tmp_path: Path) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
        }
    )

    assert config.output_dir == Path("reports/ensemble_optimizer")
    assert config.local_data_dir == Path("temp/ensemble_optimizer")
    assert config.pred_cache_dir == Path("temp/ensemble_optimizer_cache")
    assert config.semantic_architectures == ("SWIN", "DPT", "SEGFORMER", "UPERNET")
    assert config.spatial_architectures == ("DEEPLABV3PLUS", "UNET++", "FPN", "MANET")
    assert config.sort_metric == "best_val_auprc_pixel_score"
    assert config.val_calibration_frac == VALIDATION_CALIBRATION_FRACTION
    assert config.spatial_patient_policy == "all"
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"


def test_load_ensemble_optimizer_config_reads_shared_runtime_normalization_method(
    tmp_path: Path,
) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "RUNTIME_NORMALIZATION_METHOD": "vahadane",
        }
    )

    assert config.runtime_normalization_method == "VAHADANE"


def test_load_ensemble_optimizer_config_reads_runtime_vahadane_backend(
    tmp_path: Path,
) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "RUNTIME_VAHADANE_BACKEND": "torch_staintools_exact",
        }
    )

    assert config.runtime_vahadane_backend == "torch_staintools_exact"


def test_load_ensemble_optimizer_config_allows_explicit_positive_only_policy(
    tmp_path: Path,
) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "ENSEMBLE_OPT_SPATIAL_PATIENT_POLICY": "positive_only",
        }
    )

    assert config.spatial_patient_policy == "positive_only"


def test_load_ensemble_optimizer_config_rejects_unknown_group_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown ensemble optimizer architectures"):
        load_ensemble_optimizer_config(
            {
                "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
                "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES": "swin, madeupnet",
            }
        )


def test_load_ensemble_optimizer_config_requires_dataset_path() -> None:
    with pytest.raises(ValueError, match="ENSEMBLE_OPT_MASTER_MANIFEST_PATH"):
        load_ensemble_optimizer_config({})
