from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers.ensemble_inference.reporting import (
    CsvReportConfig,
    LatexReportConfig,
    MarkdownReportConfig,
    export_results_to_csv,
    save_confusion_matrix_png,
    write_ensemble_report_latex,
    write_ensemble_report_markdown,
)
from helpers.runtime_platform import COLAB_INLINE_MATPLOTLIB_BACKEND, HEADLESS_MATPLOTLIB_BACKEND


def _sample_recipe() -> dict[str, object]:
    return {
        "recipe_schema_version": 2,
        "calibration_objective": "balanced_rule6",
        "ensemble_strategy": "two_stream_spatial_gating",
        "roi_config": {"threshold": 0.33, "scale": 4},
        "decision_config": {"threshold": 0.57},
        "postprocessing_config": {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": 16,
            "min_patient_positive_patches": 3,
            "min_patient_positive_area_fraction": 1e-4,
            "min_component_area_fraction_patch": 0.001,
        },
        "validation_calibration_summary": {"objective": "balanced_rule6"},
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
        "auc_source": "raw_probabilities_before_hard_postprocessing",
        "confusion_matrix": {"tp": 8, "fp": 2, "fn": 1, "tn": 9},
        "bootstrap": {"ran": True, "n_patients": 24, "n_bootstrap_samples": 10000, "seed": 24},
        "macro_dice_rule6_split": {
            "dice_pos_only": {"point_estimate": 0.88, "ci": [0.82, 0.9]},
            "neg_clean_rate": {"point_estimate": 0.75, "ci": [0.6, 0.8]},
            "n_pos_patients": 10,
            "n_neg_patients": 14,
        },
        "diagset_patch_level_metrics": {
            "method": "diagset_patch_recognition_from_segmentation_masks",
            "primary_comparison": "pre_patient_suppression",
            "prediction_rule": {
                "mask_source": (
                    "component_filtered_binary_segmentation_mask_before_patient_suppression"
                ),
                "positive_area_fraction_threshold": 0.0,
                "positive_comparator": ">",
            },
            "ground_truth_rule": "stage2_manifest_label_after_extraction_overlap_rule",
            "pre_patient_suppression": {
                "point_estimate": {
                    "accuracy": 0.9458,
                    "avacc": 0.9470,
                    "sensitivity": 0.93,
                    "specificity": 0.964,
                },
                "confusion_matrix": {"tp": 93, "fp": 4, "fn": 7, "tn": 96},
                "support": {
                    "total_patches": 200,
                    "positive_patches": 100,
                    "negative_patches": 100,
                },
            },
            "post_patient_suppression": {
                "point_estimate": {
                    "accuracy": 0.94,
                    "avacc": 0.94,
                    "sensitivity": 0.92,
                    "specificity": 0.96,
                },
                "confusion_matrix": {"tp": 92, "fp": 4, "fn": 8, "tn": 96},
                "support": {
                    "total_patches": 200,
                    "positive_patches": 100,
                    "negative_patches": 100,
                },
            },
        },
        "patient_diagnostics": [
            {
                "patient_id": "p1",
                "gt_positive_area": 9,
                "pred_positive_area": 8,
                "pre_suppression_pred_positive_area": 8,
                "tp_area": 8,
                "fp_area": 0,
                "fn_area": 1,
                "tn_area": 9,
                "dice": 0.9412,
                "precision": 1.0,
                "recall": 0.8889,
                "positive_patch_count": 2,
                "largest_component_area": 8,
                "suppressed": False,
            }
        ],
    }


def test_export_results_to_csv_writes_report(tmp_path: Path) -> None:
    csv_path = export_results_to_csv(
        _sample_recipe(),
        _sample_metrics(),
        CsvReportConfig(
            output_dir=tmp_path,
            timestamp="2026-03-20_12_00_00",
            recipe_path=tmp_path / "recipe.json",
            dataset_dir=tmp_path / "dataset",
            seed=24,
            batch_size=8,
        ),
    )

    assert csv_path.name == "FINAL_EVALUATION_REPORT_2026-03-20_12_00_00.csv"
    assert "Ensemble Composition" in csv_path.read_text(encoding="utf-8")
    assert "Decision Threshold" in csv_path.read_text(encoding="utf-8")
    assert "Min Component Area" in csv_path.read_text(encoding="utf-8")
    assert "Patient Diagnostics" in csv_path.read_text(encoding="utf-8")
    assert "largest_component_area" in csv_path.read_text(encoding="utf-8")


def test_save_confusion_matrix_png_replaces_colab_inline_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", COLAB_INLINE_MATPLOTLIB_BACKEND)

    png_path = save_confusion_matrix_png(8, 2, 1, 9, tmp_path / "cm.png")

    assert png_path.exists()
    assert os.environ["MPLBACKEND"] == HEADLESS_MATPLOTLIB_BACKEND


def test_write_ensemble_report_latex_builds_tex_and_pdf(tmp_path: Path) -> None:
    cm_path = tmp_path / "cm.png"
    cm_path.write_bytes(b"png")

    def fake_runner(command: list[str], cwd: Path) -> None:
        del command
        (cwd / "FINAL_ENSEMBLE_REPORT_2026-03-20_12_00_00.pdf").write_bytes(b"pdf")

    tex_path, pdf_path = write_ensemble_report_latex(
        _sample_recipe(),
        _sample_metrics(),
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        config=LatexReportConfig(
            cm_png_path=cm_path,
            output_dir=tmp_path,
            timestamp="2026-03-20_12_00_00",
            latex_runner=fake_runner,
        ),
    )

    assert tex_path.exists()
    assert pdf_path.exists()
    assert "Ensemble Final Evaluation Report" in tex_path.read_text(encoding="utf-8")
    assert "Decision Threshold" in tex_path.read_text(encoding="utf-8")
    assert "Minimum Component Area" in tex_path.read_text(encoding="utf-8")
    assert "DIAGSET Patch-Level Recognition" in tex_path.read_text(encoding="utf-8")
    assert "AvAcc" in tex_path.read_text(encoding="utf-8")


def test_write_ensemble_report_markdown_writes_report_and_sanitized_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ARTIFACT_DEVICE=cuda\n"
        "WINDOW_SIZE=224\n"
        "TRAINING_ARCHITECTURE=UNET\n"
        "TRAINING_ENCODER=resnet34\n"
        "ARTIFACT_IMAGES_ZIP=./data/slides.zip\n"
        "HF_TOKEN=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    markdown_path = write_ensemble_report_markdown(
        _sample_recipe(),
        _sample_metrics(),
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        config=MarkdownReportConfig(output_dir=tmp_path, timestamp="2026-03-20_12_00_00"),
    )

    contents = markdown_path.read_text(encoding="utf-8")

    assert markdown_path.exists()
    assert "# Ensemble Final Evaluation Report" in contents
    assert "## Final Metrics (Micro / Pixel-Level)" in contents
    assert "## DIAGSET Patch-Level Recognition" in contents
    assert "| AvAcc | 0.9470 |" in contents
    assert "Before Patient-Level Suppression" in contents
    assert "Minimum Component Area" in contents
    assert "## Environment Variables Used For This Project" in contents
    assert "| ARTIFACT_DEVICE | cuda |" in contents
    assert "| WINDOW_SIZE | 224 |" in contents
    assert "Pred NoCancer" in contents
    assert "![" not in contents
    assert "ARTIFACT_IMAGES_ZIP" not in contents
    assert "HF_TOKEN" not in contents
    assert "TRAINING_ARCHITECTURE" not in contents
    assert "TRAINING_ENCODER" not in contents
