from pathlib import Path

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
        "constraints": {"random_state": None},
        "train_patients": [1],
        "val_patients": [],
        "test_patients": [],
        "split_seed": 11,
        "split_attempt": 1,
        "objective_score": None,
        "objective_score_split": "TRAIN",
    }
    manifest_df = pd.DataFrame(
        [
            {
                "abs_image_path": str(tmp_path / "TRAIN" / "CANCER" / "PATIENT_1_PATCH_001.png"),
                "abs_mask_path": str(
                    tmp_path / "TRAIN" / "CANCER_MASK" / "PATIENT_1_PATCH_001.png"
                ),
            }
        ]
    )
    calls: list[str] = []

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
        "helpers.crossfold.pipeline.create_train_val_test_split_best",
        lambda **kwargs: record("split", split_data),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.build_hdf5_manifest_from_split_dfs",
        lambda **kwargs: record("manifest", manifest_df),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_split_hdf5",
        lambda **kwargs: record(f"write:{kwargs['output_path'].name}", kwargs["output_path"]),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.verify_split_hdf5_integrity",
        lambda output_path, split_df: record(f"verify:{output_path.name}"),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_manifest_and_log_stats",
        lambda **kwargs: record("provenance"),
    )

    summary = run_crossfold_pipeline(
        CrossfoldConfig(
            normalization_method="NOT_NORMALIZED",
            source_hdf5_path=source_path,
            overwrite_output_dir=True,
            random_state=42,
            allow_destructive_move=False,
            constraints=SplitConstraints(
                min_test_patients=1,
                min_val_patients=1,
                min_train_patients=1,
                max_tries=10,
            ),
            objective=ObjectiveConfig(enable_objective=False),
            calc_checksums=False,
            save_entropy_cache_csv=False,
            log_folder=tmp_path / "logs",
            log_file_name="data_preparation.log",
        )
    )

    assert summary.output_dir == tmp_path / "NOT_NORMALIZED" / "NOT_NORMALIZED_seed_42"
    assert summary.manifest_rows == 1
    assert calls == [
        "log:data_preparation.log",
        "load",
        "split",
        "manifest",
        "write:TRAIN.h5",
        "verify:TRAIN.h5",
        "provenance",
    ]
