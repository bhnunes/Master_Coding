from pathlib import Path

import pytest

from helpers.ensemble_optimizer.config import load_ensemble_optimizer_config


def test_load_ensemble_optimizer_config_reads_expected_environment(tmp_path: Path) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "ENSEMBLE_OPT_OUTPUT_DIR": str(tmp_path / "reports"),
            "ENSEMBLE_OPT_LOCAL_DATA_DIR": str(tmp_path / "cache"),
            "ENSEMBLE_OPT_PRED_CACHE_DIR": str(tmp_path / "pred_cache"),
            "ENSEMBLE_OPT_SEED": "17",
            "ENSEMBLE_OPT_BATCH_SIZE": "12",
            "ENSEMBLE_OPT_WORKERS": "3",
            "ENSEMBLE_OPT_TOP_MODELS": "5",
            "ENSEMBLE_OPT_SORT_METRIC": "best_validation_DICE",
            "ENSEMBLE_OPT_STAGE_INPUT_LOCALLY": "false",
            "ENSEMBLE_OPT_OVERWRITE_OUTPUT": "false",
            "ENSEMBLE_OPT_VAL_HOLDOUT_FRAC": "0.3",
            "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES": "swin, segformer",
            "ENSEMBLE_OPT_SPATIAL_ARCHITECTURES": "fpn, manet",
            "ENSEMBLE_OPT_NUM_TRIALS_SEMANTIC": "11",
            "ENSEMBLE_OPT_NUM_TRIALS_SPATIAL": "13",
        }
    )

    assert config.hdf5_drive_dir == tmp_path / "dataset"
    assert config.metadata_dir == tmp_path / "metadata"
    assert config.output_dir == tmp_path / "reports"
    assert config.local_data_dir == tmp_path / "cache"
    assert config.pred_cache_dir == tmp_path / "pred_cache"
    assert config.seed == 17
    assert config.batch_size == 12
    assert config.workers == 3
    assert config.top_models == 5
    assert config.sort_metric == "best_validation_DICE"
    assert config.stage_input_locally is False
    assert config.overwrite_output is False
    assert config.val_holdout_frac == 0.3
    assert config.semantic_architectures == ("SWIN", "SEGFORMER")
    assert config.spatial_architectures == ("FPN", "MANET")
    assert config.num_trials_semantic == 11
    assert config.num_trials_spatial == 13
    assert config.log_path == Path("logs/ensemble_optimizer.log")


def test_load_ensemble_optimizer_config_uses_portable_defaults(tmp_path: Path) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
        }
    )

    assert config.output_dir == Path("reports/ensemble_optimizer")
    assert config.local_data_dir == Path("temp/ensemble_optimizer")
    assert config.pred_cache_dir == Path("temp/ensemble_optimizer_cache")
    assert config.semantic_architectures == ("SWIN", "DPT", "SEGFORMER", "UPERNET")
    assert config.spatial_architectures == ("DEEPLABV3PLUS", "UNET++", "FPN", "MANET")
    assert config.sort_metric == "best_val_auprc_pixel_score"
    assert config.spatial_patient_policy == "all"


def test_load_ensemble_optimizer_config_allows_explicit_positive_only_policy(
    tmp_path: Path,
) -> None:
    config = load_ensemble_optimizer_config(
        {
            "ENSEMBLE_OPT_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
            "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
            "ENSEMBLE_OPT_SPATIAL_PATIENT_POLICY": "positive_only",
        }
    )

    assert config.spatial_patient_policy == "positive_only"


def test_load_ensemble_optimizer_config_rejects_unknown_group_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown ensemble optimizer architectures"):
        load_ensemble_optimizer_config(
            {
                "ENSEMBLE_OPT_HDF5_DRIVE_DIR": str(tmp_path / "dataset"),
                "ENSEMBLE_OPT_METADATA_DIR": str(tmp_path / "metadata"),
                "ENSEMBLE_OPT_SEMANTIC_ARCHITECTURES": "swin, madeupnet",
            }
        )


def test_load_ensemble_optimizer_config_requires_dataset_path() -> None:
    with pytest.raises(ValueError, match="ENSEMBLE_OPT_HDF5_DRIVE_DIR"):
        load_ensemble_optimizer_config({})
