import json
from pathlib import Path

import pytest

from helpers.ensemble_optimizer.config import load_ensemble_optimizer_config

ENSEMBLE_SEED = 17
ENSEMBLE_BATCH_SIZE = 12
ENSEMBLE_WORKERS = 3
DEFAULT_OPTIMIZATION_CACHE_MAX_BYTES = 8589934592
EXPLICIT_OPTIMIZATION_CACHE_MAX_BYTES = 123456
LOCAL_SHARD_CACHE_MAX_BYTES = 987654
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
            "ENSEMBLE_OPT_DECISION_THRESHOLD_MIN": "0.55",
            "ENSEMBLE_OPT_DECISION_THRESHOLD_MAX": "0.85",
            "ENSEMBLE_OPT_DECISION_THRESHOLD_STEP": "0.05",
            "ENSEMBLE_OPT_POS_DICE_DROP_TOLERANCE": "0.03",
            "ENSEMBLE_OPT_MIN_COMPONENT_AREA_PX_CANDIDATES": "0,16,16,64",
            "ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES": "1,3,5",
            "ENSEMBLE_OPT_OPTIMIZATION_CACHE_MAX_BYTES": str(EXPLICIT_OPTIMIZATION_CACHE_MAX_BYTES),
            "ENSEMBLE_OPT_LOCAL_SHARD_CACHE_MAX_BYTES": str(LOCAL_SHARD_CACHE_MAX_BYTES),
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
    assert config.decision_threshold_min == pytest.approx(0.55)
    assert config.decision_threshold_max == pytest.approx(0.85)
    assert config.decision_threshold_step == pytest.approx(0.05)
    assert config.pos_dice_drop_tolerance == pytest.approx(0.03)
    assert config.min_component_area_px_candidates == (0, 16, 64)
    assert config.min_patient_positive_patches_candidates == (1, 3, 5)
    assert config.optimization_cache_max_bytes == EXPLICIT_OPTIMIZATION_CACHE_MAX_BYTES
    assert config.local_shard_cache_max_bytes == LOCAL_SHARD_CACHE_MAX_BYTES
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
    assert config.decision_threshold_min == pytest.approx(0.50)
    assert config.decision_threshold_max == pytest.approx(0.95)
    assert config.decision_threshold_step == pytest.approx(0.01)
    assert config.pos_dice_drop_tolerance == pytest.approx(0.02)
    assert config.min_component_area_px_candidates == (0, 16, 32, 64, 128, 256)
    assert config.min_patient_positive_patches_candidates == (1, 2, 3, 5, 10)
    assert config.optimization_cache_max_bytes == DEFAULT_OPTIMIZATION_CACHE_MAX_BYTES
    assert config.local_shard_cache_max_bytes == 0
    assert config.runtime_normalization_method == "NOT_NORMALIZED"
    assert config.runtime_vahadane_backend == "fixed_source"


def test_load_ensemble_optimizer_config_allows_disabling_optimization_cache(
    tmp_path: Path,
) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                tmp_path / "dataset" / "master_manifest.sqlite"
            ),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "ENSEMBLE_OPT_OPTIMIZATION_CACHE_MAX_BYTES": "0",
        }
    )

    assert config.optimization_cache_max_bytes == 0


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


def test_load_ensemble_optimizer_config_requires_unfiltered_rule6_baseline(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must include 0"):
        load_ensemble_optimizer_config(
            {
                "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
                "ENSEMBLE_OPT_MIN_COMPONENT_AREA_PX_CANDIDATES": "16,64",
            }
        )

    with pytest.raises(ValueError, match="must include 1"):
        load_ensemble_optimizer_config(
            {
                "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
                "ENSEMBLE_OPT_MIN_PATIENT_POSITIVE_PATCHES_CANDIDATES": "2,3",
            }
        )


def test_load_ensemble_optimizer_config_rejects_semantic_spatial_overlap(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="overlapping architectures: FPN"):
        load_ensemble_optimizer_config(
            {
                "ENSEMBLE_OPT_MASTER_MANIFEST_PATH": str(
                    tmp_path / "dataset" / "master_manifest.sqlite"
                ),
                "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
                "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES": "swin, fpn",
                "ENSEMBLE_OPT_SPATIAL_ARCHITECTURES": "fpn, manet",
            }
        )


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
