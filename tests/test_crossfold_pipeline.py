from pathlib import Path
from typing import Any, cast

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.config import CrossfoldConfig, ObjectiveConfig, SplitConstraints
from helpers.crossfold.pipeline import run_crossfold_pipeline


def test_run_crossfold_pipeline_executes_stage_flow(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    dataset = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": "/src/image.png",
                "mask_path": "/src/mask.png",
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
            }
        ]
    )
    split_data = {
        "train_df": dataset,
        "val_df": pd.DataFrame(columns=dataset.columns),
        "test_df": pd.DataFrame(columns=dataset.columns),
        "constraints": {"random_state": None, "global_cancer_ratio": 1.0},
        "train_patients": [1],
        "val_patients": [],
        "test_patients": [],
        "split_seed": 11,
        "split_attempt": 1,
        "objective_score": 0.0,
        "verification": {
            "TRAIN": {
                "expected_cancer_samples": 1.0,
                "expected_non_cancer_samples": 0.0,
                "chi2_stat": 0.0,
            },
            "VALIDATION": {
                "expected_cancer_samples": 0.0,
                "expected_non_cancer_samples": 0.0,
                "chi2_stat": 0.0,
            },
            "TEST": {
                "expected_cancer_samples": 0.0,
                "expected_non_cancer_samples": 0.0,
                "chi2_stat": 0.0,
            },
        },
    }
    manifest_df = pd.DataFrame(
        [
            {
                "relative_hdf5_path": "TRAIN.h5",
                "hdf5_row_index": 0,
            }
        ]
    )
    calls: list[str] = []
    split_kwargs: dict[str, object] = {}
    provenance_kwargs: dict[str, object] = {}
    write_calls: list[dict[str, object]] = []
    cached_provenance = {"path": str(source_path), "sha256": "source-hash", "attrs": {}}

    def record(name: str, return_value: object | None = None) -> object | None:
        calls.append(name)
        return return_value

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.configure_crossfold_logging",
        lambda log_path: record(f"log:{log_path.name}"),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.load_patch_dataset",
        lambda data_dir: record("load", dataset),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.collect_hdf5_provenance",
        lambda path: cached_provenance,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.create_train_val_test_split_best",
        lambda **kwargs: (split_kwargs.update(kwargs), record("split", split_data))[1],
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.build_hdf5_manifest_from_split_dfs",
        lambda **kwargs: record("manifest", manifest_df),
    )

    def fake_write_split_hdf5(**kwargs: object) -> object:
        write_calls.append(kwargs)
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        return record(f"write:{output_path.name}", output_path)

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_split_hdf5",
        fake_write_split_hdf5,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.verify_split_hdf5_integrity",
        lambda output_path, split_df: record(f"verify:{output_path.name}"),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_manifest_and_log_stats",
        lambda **kwargs: (provenance_kwargs.update(kwargs), record("provenance")),
    )

    summary = run_crossfold_pipeline(
        CrossfoldConfig(
            normalization_method="NOT_NORMALIZED",
            source_hdf5_path=source_path,
            overwrite_output_dir=True,
            random_state=42,
            constraints=SplitConstraints(
                test_patient_count=20,
                validation_patient_count=20,
            ),
            objective=ObjectiveConfig(optuna_trials=25),
            hdf5_compression="NONE",
            copy_batch_size=256,
            calc_checksums=False,
            save_entropy_cache_csv=False,
            log_folder=tmp_path / "logs",
            log_file_name="data_preparation.log",
        )
    )

    assert summary.output_dir == tmp_path / "NOT_NORMALIZED" / "NOT_NORMALIZED_seed_42"
    assert summary.manifest_rows == 1
    split_selection = cast(dict[str, Any], provenance_kwargs["extra"])["split_selection"]
    assert split_selection["method"] == "optuna_greedy_sample_ratio_stratified"
    assert split_selection["tie_break_priority"] == ["TEST", "VALIDATION", "TRAIN"]
    assert split_selection["global_cancer_ratio"] == 1.0
    assert split_selection["optuna_trials"] == 25
    assert split_selection["loss_metric"] == "sum_absolute_split_ratio_delta"
    assert split_selection["final_loss"] == 0.0
    assert len(write_calls) == 1
    assert write_calls[0]["source_hdf5_provenance"] is cached_provenance
    assert provenance_kwargs["source_hdf5_provenance"] is cached_provenance
    assert (
        cast(dict[str, Any], provenance_kwargs["extra"])["verification"]
        == split_data["verification"]
    )
    assert calls == [
        "log:data_preparation.log",
        "load",
        "split",
        "manifest",
        "write:TRAIN.h5",
        "verify:TRAIN.h5",
        "provenance",
    ]
