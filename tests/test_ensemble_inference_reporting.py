from __future__ import annotations

from pathlib import Path

import pytest

from helpers.ensemble_inference.reporting import (
    export_results_to_csv,
    write_ensemble_report_latex,
    write_ensemble_report_markdown,
)


def _sample_recipe() -> dict[str, object]:
    return {
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.33, "scale": 4},
        "decision_config": {"threshold": 0.57},
        "model_registry": [
            {
                "architecture": "SWIN",
                "encoder": "enc-a",
                "checkpoint_path": "/tmp/a.pth",
                "stream_role": "semantic",
                "weight": 1.0,
            }
        ],
    }


def _sample_metrics() -> dict[str, object]:
    ci = {"dice": [0.8, 0.9]}
    point_estimate = {"dice": 0.85}
    return {
        "micro_averaged_metrics": {"point_estimate": point_estimate, "ci": ci},
        "macro_averaged_metrics": {"point_estimate": point_estimate, "ci": ci},
        "auc": 0.91,
        "confusion_matrix": {"tp": 8, "fp": 2, "fn": 1, "tn": 9},
        "bootstrap": {"ran": True, "n_patients": 24, "n_bootstrap_samples": 10000, "seed": 24},
        "macro_dice_rule6_split": {
            "dice_pos_only": {"point_estimate": 0.88, "ci": [0.82, 0.9]},
            "neg_clean_rate": {"point_estimate": 0.75, "ci": [0.6, 0.8]},
            "n_pos_patients": 10,
            "n_neg_patients": 14,
        },
    }


def test_export_results_to_csv_writes_report(tmp_path: Path) -> None:
    csv_path = export_results_to_csv(
        ensemble_recipe=_sample_recipe(),
        metrics_results=_sample_metrics(),
        output_dir=tmp_path,
        timestamp="2026-03-20_12_00_00",
        recipe_path=tmp_path / "recipe.json",
        dataset_dir=tmp_path / "dataset",
        seed=24,
        batch_size=8,
    )

    assert csv_path.name == "FINAL_EVALUATION_REPORT_2026-03-20_12_00_00.csv"
    assert "Ensemble Composition" in csv_path.read_text(encoding="utf-8")
    assert "Decision Threshold" in csv_path.read_text(encoding="utf-8")


def test_write_ensemble_report_latex_builds_tex_and_pdf(tmp_path: Path) -> None:
    cm_path = tmp_path / "cm.png"
    cm_path.write_bytes(b"png")

    def fake_runner(command: list[str], cwd: Path) -> None:
        del command
        (cwd / "FINAL_ENSEMBLE_REPORT_2026-03-20_12_00_00.pdf").write_bytes(b"pdf")

    tex_path, pdf_path = write_ensemble_report_latex(
        ensemble_recipe=_sample_recipe(),
        ensemble_metrics=_sample_metrics(),
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        cm_png_path=cm_path,
        output_dir=tmp_path,
        timestamp="2026-03-20_12_00_00",
        latex_runner=fake_runner,
    )

    assert tex_path.exists()
    assert pdf_path.exists()
    assert "Ensemble Final Evaluation Report" in tex_path.read_text(encoding="utf-8")
    assert "Decision Threshold" in tex_path.read_text(encoding="utf-8")


def test_write_ensemble_report_markdown_writes_report_and_sanitized_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ARTIFACT_DEVICE=cuda\n"
        "WINDOW_SIZE=224\n"
        "ARTIFACT_IMAGES_ZIP=./data/slides.zip\n"
        "HF_TOKEN=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    markdown_path = write_ensemble_report_markdown(
        ensemble_recipe=_sample_recipe(),
        ensemble_metrics=_sample_metrics(),
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        output_dir=tmp_path,
        timestamp="2026-03-20_12_00_00",
    )

    contents = markdown_path.read_text(encoding="utf-8")

    assert markdown_path.exists()
    assert "# Ensemble Final Evaluation Report" in contents
    assert "## Final Metrics (Micro / Pixel-Level)" in contents
    assert "## Environment Variables Used For This Project" in contents
    assert "| ARTIFACT_DEVICE | cuda |" in contents
    assert "| WINDOW_SIZE | 224 |" in contents
    assert "Pred NoCancer" in contents
    assert "![" not in contents
    assert "ARTIFACT_IMAGES_ZIP" not in contents
    assert "HF_TOKEN" not in contents
