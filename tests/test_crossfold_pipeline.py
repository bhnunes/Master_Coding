import sqlite3
from pathlib import Path
from typing import Any, cast

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.config import CrossfoldConfig, ObjectiveConfig, SplitConstraints
from helpers.crossfold.pipeline import _persist_stage5_split_state, run_crossfold_pipeline


def test_run_crossfold_pipeline_executes_stage_flow(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "master_manifest.sqlite"
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
    persist_kwargs: dict[str, object] = {}
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
        "helpers.crossfold.pipeline.collect_source_dataset_provenance",
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
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_manifest_and_log_stats",
        lambda **kwargs: (provenance_kwargs.update(kwargs), record("provenance")),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline._persist_stage5_split_state",
        lambda **kwargs: (persist_kwargs.update(kwargs), record("persist")),
    )

    summary = run_crossfold_pipeline(
        CrossfoldConfig(
            normalization_method="NOT_NORMALIZED",
            source_path=source_path,
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
    assert provenance_kwargs["source_hdf5_provenance"] is cached_provenance
    assert persist_kwargs["master_manifest_path"] == source_path
    assert persist_kwargs["normalization_method"] == "NOT_NORMALIZED"
    assert (
        cast(dict[str, Any], provenance_kwargs["extra"])["verification"]
        == split_data["verification"]
    )
    assert calls == [
        "log:data_preparation.log",
        "load",
        "split",
        "manifest",
        "provenance",
        "persist",
    ]


def test_run_crossfold_pipeline_computes_entropy_only_for_train_split(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "master_manifest.sqlite"
    dataset = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": "/src/train_1.png",
                "mask_path": "/src/train_1_mask.png",
                "label": 1,
                "filename": "PATIENT_1_PATCH_001.png",
            },
            {
                "patient_id": 2,
                "image_path": "/src/val_1.png",
                "mask_path": "/src/val_1_mask.png",
                "label": 0,
                "filename": "PATIENT_2_PATCH_001.png",
            },
            {
                "patient_id": 3,
                "image_path": "/src/test_1.png",
                "mask_path": "/src/test_1_mask.png",
                "label": 1,
                "filename": "PATIENT_3_PATCH_001.png",
            },
        ]
    )
    train_df = dataset.iloc[[0]].reset_index(drop=True)
    val_df = dataset.iloc[[1]].reset_index(drop=True)
    test_df = dataset.iloc[[2]].reset_index(drop=True)
    split_data = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "constraints": {
            "random_state": None,
            "global_cancer_ratio": 2 / 3,
            "test_patient_count": 1,
            "validation_patient_count": 1,
        },
        "train_patients": [1],
        "val_patients": [2],
        "test_patients": [3],
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
                "expected_non_cancer_samples": 1.0,
                "chi2_stat": 0.0,
            },
            "TEST": {
                "expected_cancer_samples": 1.0,
                "expected_non_cancer_samples": 0.0,
                "chi2_stat": 0.0,
            },
        },
    }
    manifest_df = pd.DataFrame([{"relative_hdf5_path": "TRAIN.h5", "hdf5_row_index": 0}])
    entropy_inputs: list[pd.DataFrame] = []
    fit_calls: list[dict[str, object]] = []
    persist_kwargs: dict[str, object] = {}
    cached_provenance = {"path": str(source_path), "sha256": "source-hash", "attrs": {}}

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.configure_crossfold_logging", lambda log_path: None
    )
    monkeypatch.setattr("helpers.crossfold.pipeline.load_patch_dataset", lambda data_dir: dataset)
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.collect_source_dataset_provenance",
        lambda path: cached_provenance,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.create_train_val_test_split_best",
        lambda **kwargs: split_data,
    )

    def fake_compute_all_patch_entropies(**kwargs: object) -> pd.DataFrame:
        entropy_frame = cast(pd.DataFrame, kwargs["df"])
        entropy_inputs.append(entropy_frame.copy())
        return pd.DataFrame(
            {
                "image_path": entropy_frame["image_path"].tolist(),
                "entropy": [0.9] * len(entropy_frame),
            }
        )

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.compute_all_patch_entropies",
        fake_compute_all_patch_entropies,
    )

    def fake_fit_normalizer_on_train_set(
        train_df: pd.DataFrame,
        method_name: str,
        entropy_df: pd.DataFrame | None = None,
    ) -> tuple[object, list[str]]:
        fit_calls.append(
            {
                "train_df": train_df.copy(),
                "method_name": method_name,
                "entropy_df": None if entropy_df is None else entropy_df.copy(),
            }
        )
        return object(), ["/src/train_1.png"]

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.fit_normalizer_on_train_set",
        fake_fit_normalizer_on_train_set,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.save_normalizer_stats",
        lambda normalizer, method_name, output_dir, template_paths: None,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.build_hdf5_manifest_from_split_dfs",
        lambda **kwargs: manifest_df,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.write_manifest_and_log_stats",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline._persist_stage5_split_state",
        lambda **kwargs: persist_kwargs.update(kwargs),
    )

    run_crossfold_pipeline(
        CrossfoldConfig(
            normalization_method="REINHARD",
            source_path=source_path,
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

    assert len(entropy_inputs) == 1
    assert entropy_inputs[0]["image_path"].tolist() == ["/src/train_1.png"]
    assert len(fit_calls) == 1
    assert cast(pd.DataFrame, fit_calls[0]["train_df"])["image_path"].tolist() == [
        "/src/train_1.png"
    ]
    assert cast(pd.DataFrame, fit_calls[0]["entropy_df"])["image_path"].tolist() == [
        "/src/train_1.png"
    ]
    split_frames = cast(dict[str, pd.DataFrame], persist_kwargs["split_frames"])
    assert split_frames["TRAIN"]["image_path"].tolist() == ["/src/train_1.png"]
    assert split_frames["VALIDATION"]["image_path"].tolist() == ["/src/val_1.png"]
    assert split_frames["TEST"]["image_path"].tolist() == ["/src/test_1.png"]
    assert persist_kwargs["normalization_method"] == "REINHARD"


def test_persist_stage5_split_state_updates_master_manifest_sqlite(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    output_dir = tmp_path / "NOT_NORMALIZED" / "NOT_NORMALIZED_seed_42"
    output_dir.mkdir(parents=True)
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")

    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                slide_id TEXT,
                source_signature TEXT,
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL,
                source_slide_path TEXT NOT NULL,
                annotation_path TEXT,
                artifacts_geojson_path TEXT,
                stage2_case_record_id INTEGER NOT NULL,
                stage2_processing_signature TEXT,
                stage2_status TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_hdf5_path, source_row_index)
            );
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_name TEXT NOT NULL,
                config_path TEXT,
                config_sha256 TEXT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                code_version TEXT,
                input_summary_json_path TEXT,
                input_summary_sha256 TEXT
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            """
        )
        for row_index, filename in enumerate(("a.png", "b.png")):
            cursor = connection.execute(
                """
                INSERT INTO patches (
                    source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                    source_signature, source_image_path, source_mask_path, source_slide_path,
                    stage2_case_record_id, stage2_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "/patches/patient_1.h5",
                    row_index,
                    filename,
                    1,
                    row_index,
                    "slide_0",
                    "sig",
                    f"/patches/patient_1.h5::images[{row_index}]",
                    f"/patches/patient_1.h5::masks[{row_index}]",
                    "/slides/slide_0.svs",
                    1,
                    "COMPLETED",
                ),
            )
            connection.execute(
                "INSERT INTO patch_stage_state ("
                "patch_id, is_stage4_accepted, last_updated_stage_name"
                ") VALUES (?, 1, ?)",
                ((cursor.lastrowid or 0), "STAGE4_3"),
            )
        connection.commit()

    split_frames = {
        "TRAIN": pd.DataFrame(
            [
                {
                    "patient_id": 1,
                    "label": 0,
                    "filename": "a.png",
                    "source_hdf5_path": "/patches/patient_1.h5",
                    "source_row_index": 0,
                }
            ]
        ),
        "VALIDATION": pd.DataFrame(
            [
                {
                    "patient_id": 1,
                    "label": 1,
                    "filename": "b.png",
                    "source_hdf5_path": "/patches/patient_1.h5",
                    "source_row_index": 1,
                }
            ]
        ),
        "TEST": pd.DataFrame(
            columns=[
                "patient_id",
                "label",
                "filename",
                "source_hdf5_path",
                "source_row_index",
            ]
        ),
    }

    _persist_stage5_split_state(
        master_manifest_path=master_manifest_path,
        split_frames=split_frames,
        normalization_method="NOT_NORMALIZED",
        output_dir=output_dir,
    )

    with sqlite3.connect(master_manifest_path) as connection:
        updated_rows = connection.execute(
            "SELECT split, normalization_method, normalization_artifact_id, "
            "last_updated_stage_name FROM patch_stage_state ORDER BY patch_id ASC"
        ).fetchall()
        run_rows = connection.execute("SELECT stage_name, config_path FROM runs").fetchall()

    assert updated_rows == [
        ("TRAIN", "NOT_NORMALIZED", None, "STAGE5"),
        ("VALIDATION", "NOT_NORMALIZED", None, "STAGE5"),
    ]
    assert run_rows == [("STAGE5", str(output_dir / "run_config.json"))]


def test_persist_stage5_split_state_records_normalization_artifact(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    output_dir = tmp_path / "REINHARD" / "REINHARD_seed_42"
    templates_dir = output_dir / "normalization_templates"
    output_dir.mkdir(parents=True)
    templates_dir.mkdir()
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (output_dir / "normalization_stats.json").write_text('{"method": "REINHARD"}', encoding="utf-8")

    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hdf5_path TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                slide_id TEXT,
                source_signature TEXT,
                source_image_path TEXT NOT NULL,
                source_mask_path TEXT NOT NULL,
                source_slide_path TEXT NOT NULL,
                annotation_path TEXT,
                artifacts_geojson_path TEXT,
                stage2_case_record_id INTEGER NOT NULL,
                stage2_processing_signature TEXT,
                stage2_status TEXT NOT NULL,
                cov_fold REAL NOT NULL DEFAULT 0.0,
                cov_penmarking REAL NOT NULL DEFAULT 0.0,
                cov_oof REAL NOT NULL DEFAULT 0.0,
                cov_darkspot_foreign REAL NOT NULL DEFAULT 0.0,
                cov_edge_airbubble REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_hdf5_path, source_row_index)
            );
            CREATE TABLE patch_stage_state (
                patch_id INTEGER PRIMARY KEY,
                cleaning_decision TEXT,
                contamination_rate REAL,
                split TEXT,
                normalization_method TEXT,
                normalization_artifact_id INTEGER,
                sampling_decision TEXT,
                is_stage4_accepted INTEGER,
                is_stage7_selected INTEGER,
                last_updated_stage_name TEXT,
                last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_name TEXT NOT NULL,
                config_path TEXT,
                config_sha256 TEXT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                code_version TEXT,
                input_summary_json_path TEXT,
                input_summary_sha256 TEXT
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            """
        )
        cursor = connection.execute(
            """
            INSERT INTO patches (
                source_hdf5_path, source_row_index, filename, patient_id, label, slide_id,
                source_signature, source_image_path, source_mask_path, source_slide_path,
                stage2_case_record_id, stage2_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "/patches/patient_1.h5",
                0,
                "a.png",
                1,
                1,
                "slide_0",
                "sig",
                "/patches/patient_1.h5::images[0]",
                "/patches/patient_1.h5::masks[0]",
                "/slides/slide_0.svs",
                1,
                "COMPLETED",
            ),
        )
        connection.execute(
            "INSERT INTO patch_stage_state ("
            "patch_id, is_stage4_accepted, last_updated_stage_name"
            ") VALUES (?, 1, ?)",
            ((cursor.lastrowid or 0), "STAGE4_3"),
        )
        connection.commit()

    _persist_stage5_split_state(
        master_manifest_path=master_manifest_path,
        split_frames={
            "TRAIN": pd.DataFrame(
                [
                    {
                        "patient_id": 1,
                        "label": 1,
                        "filename": "a.png",
                        "source_hdf5_path": "/patches/patient_1.h5",
                        "source_row_index": 0,
                    }
                ]
            ),
            "VALIDATION": pd.DataFrame(
                columns=[
                    "patient_id",
                    "label",
                    "filename",
                    "source_hdf5_path",
                    "source_row_index",
                ]
            ),
            "TEST": pd.DataFrame(
                columns=[
                    "patient_id",
                    "label",
                    "filename",
                    "source_hdf5_path",
                    "source_row_index",
                ]
            ),
        },
        normalization_method="REINHARD",
        output_dir=output_dir,
    )

    with sqlite3.connect(master_manifest_path) as connection:
        stage_state_rows = connection.execute(
            "SELECT split, normalization_method, normalization_artifact_id "
            "FROM patch_stage_state"
        ).fetchall()
        artifact_rows = connection.execute(
            "SELECT method, state_path, template_path, fit_scope FROM normalization_artifacts"
        ).fetchall()

    assert stage_state_rows == [("TRAIN", "REINHARD", 1)]
    assert artifact_rows == [
        (
            "REINHARD",
            str(output_dir / "normalization_stats.json"),
            str(templates_dir),
            "TRAIN",
        )
    ]
