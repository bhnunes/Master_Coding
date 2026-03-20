from __future__ import annotations

import json
from pathlib import Path

from helpers.lr_finder.config import BCEDiceSearchSpace, LRFinderConfig, ModelPlan
from helpers.lr_finder.pipeline import run_lr_finder_pipeline
from helpers.lr_finder.reporting import RunRecord
from helpers.lr_finder.runner import ScreeningOutputs


def test_run_lr_finder_pipeline_writes_config_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    config = LRFinderConfig(
        hdf5_drive_dir=tmp_path / "h5",
        output_dir=output_dir,
        local_data_dir=tmp_path / "local",
        stage_input_locally=False,
        overwrite_output=True,
        smart_sampling=True,
        execution_mode="FAST_DEV",
        amp_precision="fp16",
        seed=24,
        batch_size=8,
        workers=1,
        use_subset=False,
        subset_ratio=1.0,
        num_lhs_samples=3,
        end_lr=0.1,
        num_iter=100,
        num_repeats=2,
        optimizer_weight_decay=1e-4,
        optimizer_start_lr=1e-8,
        pdf_name="report.pdf",
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
            completed_trials=2,
            failed_trials=0,
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
    run_config = json.loads(outputs.run_config_path.read_text(encoding="utf-8"))
    assert run_config["pdf_name"] == "report.pdf"
    assert run_config["completed_trials"] == 2
    assert run_config["failed_trials"] == 0
