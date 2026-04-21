from __future__ import annotations

import json
from pathlib import Path

from helpers.lr_finder.config import BCEDiceSearchSpace, LRFinderConfig, ModelPlan
from helpers.lr_finder.pipeline import run_lr_finder_pipeline
from helpers.lr_finder.reporting import RunRecord
from helpers.lr_finder.runner import ScreeningOutputs

PIPELINE_BATCH_SIZE = 8
LHS_SAMPLE_COUNT = 3
END_LR = 0.1
NUM_ITER = 100
NUM_REPEATS = 2
WEIGHT_DECAY = 1e-4
START_LR = 1e-8
COMPLETED_TRIALS = 2
VALID_RECORD_COUNT = 1
FAILED_TRIALS = 0
ARCHITECTURE_TRIAL_STATS = {
    "FPN": {
        "valid_records": VALID_RECORD_COUNT,
        "completed_trials": COMPLETED_TRIALS,
        "failed_trials": FAILED_TRIALS,
    }
}


def test_run_lr_finder_pipeline_writes_config_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    config = LRFinderConfig(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=output_dir,
        local_data_dir=tmp_path / "local",
        stage_input_locally=False,
        overwrite_output=True,
        smart_sampling=True,
        execution_mode="FAST_DEV",
        amp_precision="fp16",
        seed=24,
        batch_size=PIPELINE_BATCH_SIZE,
        workers=1,
        use_subset=False,
        subset_ratio=1.0,
        num_lhs_samples=LHS_SAMPLE_COUNT,
        end_lr=END_LR,
        num_iter=NUM_ITER,
        num_repeats=NUM_REPEATS,
        optimizer_weight_decay=WEIGHT_DECAY,
        optimizer_start_lr=START_LR,
        pdf_name="report.pdf",
        hf_token=None,
        search_space=BCEDiceSearchSpace(),
        model_plans=[ModelPlan(architecture="FPN", encoder="senet154")],
    )

    def fake_screening_runner(config: LRFinderConfig) -> ScreeningOutputs:
        plot_path = config.output_dir / "FPN" / "001_curve.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"png")
        summary_path = config.output_dir / "SUMMARY_ALL.csv"
        summary_path.write_text("architecture,median_min_loss\nFPN,0.4\n", encoding="utf-8")
        lhs_path = config.output_dir / "LHS_SAMPLES.json"
        lhs_path.write_text("[]", encoding="utf-8")
        return ScreeningOutputs(
            records=[
                RunRecord(
                    architecture="FPN",
                    encoder="senet154",
                    alpha=0.1,
                    beta=0.2,
                    gamma=0.3,
                    median_min_loss=0.4,
                    plot_path=plot_path,
                    csv_path=summary_path,
                )
            ],
            lhs_samples_path=lhs_path,
            summary_all_path=summary_path,
            architecture_summary_paths={"FPN": config.output_dir / "FPN" / "summary.csv"},
            architecture_trial_stats={"FPN": ARCHITECTURE_TRIAL_STATS["FPN"]},
            completed_trials=COMPLETED_TRIALS,
            failed_trials=FAILED_TRIALS,
            source_split_name="TRAIN_FILTERED_shards",
            training_dataset_provenance={"manifest_path": "train_manifest.parquet"},
            validation_dataset_provenance={"manifest_path": "validation_manifest.parquet"},
        )

    def fake_report_builder(
        records: list[RunRecord],
        out_dir: Path,
        meta: dict[str, str],
        pdf_name: str,
        latex_runner: object | None = None,
    ) -> tuple[Path, Path]:
        del records, meta, latex_runner
        tex_path = out_dir / "report.tex"
        pdf_path = out_dir / pdf_name
        tex_path.write_text("report", encoding="utf-8")
        pdf_path.write_bytes(b"pdf")
        return tex_path, pdf_path

    outputs = run_lr_finder_pipeline(
        config,
        screening_runner=fake_screening_runner,
        report_builder=fake_report_builder,
    )

    assert outputs.pdf_path == output_dir / "report.pdf"
    assert outputs.tex_path == output_dir / "report.tex"
    assert outputs.valid_records == VALID_RECORD_COUNT
    assert outputs.completed_trials == COMPLETED_TRIALS
    assert outputs.failed_trials == FAILED_TRIALS
    assert outputs.architecture_trial_stats == ARCHITECTURE_TRIAL_STATS
    run_config = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert run_config["pdf_name"] == "report.pdf"
    assert run_config["valid_records"] == VALID_RECORD_COUNT
    assert run_config["completed_trials"] == COMPLETED_TRIALS
    assert run_config["failed_trials"] == FAILED_TRIALS
    assert run_config["architecture_trial_stats"] == ARCHITECTURE_TRIAL_STATS
    assert run_config["source_split_name"] == "TRAIN_FILTERED_shards"
    assert run_config["training_dataset_provenance"] == {"manifest_path": "train_manifest.parquet"}
    assert run_config["validation_dataset_provenance"] == {
        "manifest_path": "validation_manifest.parquet"
    }
    assert "runtime_environment" in run_config
    assert "git_commit" in run_config["runtime_environment"]
