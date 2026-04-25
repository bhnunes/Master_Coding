import sqlite3
from pathlib import Path
from typing import Any, cast

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold.config import CrossfoldConfig, ObjectiveConfig, SplitConstraints
from helpers.crossfold.pipeline import _persist_stage5_split_state, run_crossfold_pipeline
from helpers.extraction.manifest_paths import build_hdf5_dataset_ref, to_manifest_path_ref

OPTUNA_TRIALS = 25


def test_run_crossfold_pipeline_executes_stage_flow(  # noqa: PLR0915
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
    template_selection_calls: list[dict[str, object]] = []
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
        "helpers.crossfold.pipeline.compute_all_patch_entropies",
        lambda **kwargs: record(
            "entropy",
            pd.DataFrame({"image_path": dataset["image_path"].tolist(), "entropy": [0.9]}),
        ),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.select_template_rows_from_entropy",
        lambda train_df, entropy_df: record(
            "template_rows",
            train_df.assign(entropy=entropy_df["entropy"].iloc[0]),
        ),
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.build_aggregate_target_from_template_rows",
        lambda selected_rows: record("aggregate_target", object()),
    )

    def fake_save_template_selection_artifacts(
        output_dir: Path, **kwargs: object
    ) -> dict[str, int]:
        template_selection_calls.append({"output_dir": output_dir, **kwargs})
        return cast(dict[str, int], record("template_artifacts", {"template_count": 1}))

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.save_template_selection_artifacts",
        fake_save_template_selection_artifacts,
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

    assert summary.output_dir == tmp_path / "STAGE4_SPLITS" / "split_seed_42"
    assert summary.manifest_rows == 1
    provenance_config = cast(Any, provenance_kwargs["config"])
    split_selection = cast(dict[str, Any], provenance_config.extra)["split_selection"]
    assert split_selection["method"] == "optuna_greedy_sample_ratio_stratified"
    assert split_selection["tie_break_priority"] == ["TEST", "VALIDATION", "TRAIN"]
    assert split_selection["global_cancer_ratio"] == 1.0
    assert split_selection["optuna_trials"] == OPTUNA_TRIALS
    assert split_selection["loss_metric"] == "sum_absolute_split_ratio_delta"
    assert split_selection["final_loss"] == 0.0
    assert provenance_config.source_hdf5_provenance is cached_provenance
    assert persist_kwargs["master_manifest_path"] == source_path
    assert cast(dict[str, Any], provenance_config.extra)["template_selection"] == {
        "template_count": 1
    }
    assert cast(dict[str, Any], provenance_config.extra)["normalization_artifacts"] == []
    assert len(template_selection_calls) == 1
    assert calls.count("split") == 1
    assert calls.count("entropy") == 1
    assert calls.count("template_rows") == 1
    assert (
        cast(dict[str, Any], provenance_config.extra)["verification"]
        == split_data["verification"]
    )
    assert calls == [
        "log:data_preparation.log",
        "load",
        "split",
        "manifest",
        "entropy",
        "template_rows",
        "aggregate_target",
        "template_artifacts",
        "provenance",
        "persist",
    ]


def test_run_crossfold_pipeline_computes_entropy_once_for_train_split(
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
    template_rows_inputs: list[dict[str, object]] = []
    template_artifact_calls: list[dict[str, object]] = []
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
        source_df = cast(pd.DataFrame, kwargs["df"])
        entropy_inputs.append(source_df.copy())
        return pd.DataFrame(
            {
                "image_path": source_df["image_path"].tolist(),
                "entropy": [0.9] * len(source_df),
            }
        )

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.compute_all_patch_entropies",
        fake_compute_all_patch_entropies,
    )

    def fake_select_template_rows_from_entropy(
        train_df: pd.DataFrame, entropy_df: pd.DataFrame
    ) -> pd.DataFrame:
        template_rows_inputs.append(
            {
                "train_df": train_df.copy(),
                "entropy_df": entropy_df.copy(),
            }
        )
        return train_df.assign(entropy=entropy_df["entropy"].to_numpy())

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.select_template_rows_from_entropy",
        fake_select_template_rows_from_entropy,
    )
    monkeypatch.setattr(
        "helpers.crossfold.pipeline.build_aggregate_target_from_template_rows",
        lambda selected_rows: selected_rows,
    )

    def fake_save_template_artifacts(output_dir: Path, **kwargs: object) -> dict[str, int]:
        template_artifact_calls.append({"output_dir": output_dir, **kwargs})
        return {"template_count": len(cast(pd.DataFrame, kwargs["selected_rows"]))}

    monkeypatch.setattr(
        "helpers.crossfold.pipeline.save_template_selection_artifacts",
        fake_save_template_artifacts,
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
    assert len(template_rows_inputs) == 1
    assert cast(pd.DataFrame, template_rows_inputs[0]["train_df"])["image_path"].tolist() == [
        "/src/train_1.png"
    ]
    assert cast(pd.DataFrame, template_rows_inputs[0]["entropy_df"])["image_path"].tolist() == [
        "/src/train_1.png"
    ]
    assert len(template_artifact_calls) == 1
    split_frames = cast(dict[str, pd.DataFrame], persist_kwargs["split_frames"])
    assert split_frames["TRAIN"]["image_path"].tolist() == ["/src/train_1.png"]
    assert split_frames["VALIDATION"]["image_path"].tolist() == ["/src/val_1.png"]
    assert split_frames["TEST"]["image_path"].tolist() == ["/src/test_1.png"]


def test_persist_stage5_split_state_updates_master_manifest_sqlite(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    output_dir = tmp_path / "STAGE4_SPLITS" / "split_seed_42"
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
                    to_manifest_path_ref(
                        tmp_path / "patches" / "patient_1.h5",
                        manifest_path=master_manifest_path,
                    ),
                    row_index,
                    filename,
                    1,
                    row_index,
                    "slide_0",
                    "sig",
                    build_hdf5_dataset_ref("images", row_index),
                    build_hdf5_dataset_ref("masks", row_index),
                    "/slides/slide_0.svs",
                    1,
                    "COMPLETED",
                ),
            )
            connection.execute(
                "INSERT INTO patch_stage_state ("
                "patch_id, is_stage4_accepted, last_updated_stage_name"
                ") VALUES (?, 1, ?)",
                ((cursor.lastrowid or 0), "STAGE3_3"),
            )
        connection.commit()

    split_frames = {
        "TRAIN": pd.DataFrame(
            [
                {
                    "patient_id": 1,
                    "label": 0,
                    "filename": "a.png",
                    "source_hdf5_path": str(tmp_path / "patches" / "patient_1.h5"),
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
                    "source_hdf5_path": str(tmp_path / "patches" / "patient_1.h5"),
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
        output_dir=output_dir,
        normalization_artifacts=[],
    )

    with sqlite3.connect(master_manifest_path) as connection:
        updated_rows = connection.execute(
            "SELECT split, stage4_split_bundle_id, normalization_method, "
            "normalization_artifact_id, "
            "last_updated_stage_name FROM patch_stage_state ORDER BY patch_id ASC"
        ).fetchall()
        run_rows = connection.execute("SELECT stage_name, config_path FROM runs").fetchall()
        split_bundle_rows = connection.execute(
            "SELECT run_id, output_dir_path FROM stage4_split_bundles"
        ).fetchall()

    assert updated_rows == [
        ("TRAIN", 1, None, None, "STAGE4"),
        ("VALIDATION", 1, None, None, "STAGE4"),
    ]
    assert run_rows == [
        (
            "STAGE4",
            to_manifest_path_ref(
                output_dir / "run_config.json",
                manifest_path=master_manifest_path,
            ),
        )
    ]
    assert split_bundle_rows == [
        (1, to_manifest_path_ref(output_dir, manifest_path=master_manifest_path))
    ]


def test_persist_stage5_split_state_records_all_normalization_artifacts(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    output_dir = tmp_path / "STAGE4_SPLITS" / "split_seed_42"
    output_dir.mkdir(parents=True)
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")
    template_dir = output_dir / "template_selection"
    template_dir.mkdir()
    artifact_methods = ("REINHARD", "RUIFROK", "MACENKO", "VAHADANE")
    normalization_artifacts = []
    for method in artifact_methods:
        state_path = (
            output_dir
            / "runtime_normalization_artifacts"
            / method.lower()
            / "normalization_stats.json"
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(f'{{"method": "{method}"}}', encoding="utf-8")
        normalization_artifacts.append(
            {
                "method": method,
                "state_path": state_path,
                "template_path": template_dir,
                "fit_scope": "TRAIN",
            }
        )

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
                    to_manifest_path_ref(
                        tmp_path / "patches" / "patient_1.h5",
                        manifest_path=master_manifest_path,
                    ),
                    0,
                    "a.png",
                1,
                1,
                "slide_0",
                "sig",
                build_hdf5_dataset_ref("images", 0),
                build_hdf5_dataset_ref("masks", 0),
                "/slides/slide_0.svs",
                1,
                "COMPLETED",
            ),
        )
        connection.execute(
            "INSERT INTO patch_stage_state ("
            "patch_id, is_stage4_accepted, last_updated_stage_name"
            ") VALUES (?, 1, ?)",
            ((cursor.lastrowid or 0), "STAGE3_3"),
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
                            "source_hdf5_path": str(tmp_path / "patches" / "patient_1.h5"),
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
        output_dir=output_dir,
        normalization_artifacts=normalization_artifacts,
    )

    with sqlite3.connect(master_manifest_path) as connection:
        stage_state_rows = connection.execute(
            "SELECT split, stage4_split_bundle_id, normalization_method, normalization_artifact_id "
            "FROM patch_stage_state"
        ).fetchall()
        artifact_rows = connection.execute(
            "SELECT method, state_path, template_path, fit_scope FROM normalization_artifacts"
        ).fetchall()
        split_bundle_rows = connection.execute(
            "SELECT run_id, output_dir_path FROM stage4_split_bundles"
        ).fetchall()
        bundle_artifact_rows = connection.execute(
            "SELECT stage4_split_bundle_id, normalization_artifact_id "
            "FROM stage4_split_bundle_artifacts"
        ).fetchall()

    assert stage_state_rows == [("TRAIN", 1, None, None)]
    assert artifact_rows == [
        (
            method,
            to_manifest_path_ref(
                output_dir
                / "runtime_normalization_artifacts"
                / method.lower()
                / "normalization_stats.json",
                manifest_path=master_manifest_path,
            ),
            to_manifest_path_ref(template_dir, manifest_path=master_manifest_path),
            "TRAIN",
        )
        for method in artifact_methods
    ]
    assert split_bundle_rows == [
        (1, to_manifest_path_ref(output_dir, manifest_path=master_manifest_path))
    ]
    assert bundle_artifact_rows == [(1, 1), (1, 2), (1, 3), (1, 4)]
