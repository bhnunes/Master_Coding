from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_confusion_matrix_png(tp: int, fp: int, fn: int, tn: int, out_path_png: Path) -> Path:
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Pred NoCancer", "Pred Cancer"],
        yticklabels=["True NoCancer", "True Cancer"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Ensemble Confusion Matrix (Row-Normalized %)")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path_png


def render_scientific_analysis_report(metrics_results: dict[str, Any]) -> str:
    micro = metrics_results.get("micro_averaged_metrics", {})
    macro = metrics_results.get("macro_averaged_metrics", {})
    auc = metrics_results.get("auc")
    split = metrics_results.get("macro_dice_rule6_split", {})
    if not micro or not macro or auc is None:
        return "Metrics object is missing required data."

    macro_pe = macro.get("point_estimate", {})
    dice_pos = split.get("dice_pos_only", {})
    neg_clean = split.get("neg_clean_rate", {})
    return "\n".join(
        [
            (
                "============================== Scientific Performance Analysis "
                "=============================="
            ),
            f"Macro Dice: {macro_pe.get('dice', float('nan')):.4f}",
            f"Macro FNR: {macro_pe.get('fnr', float('nan')):.4f}",
            f"Macro FPR: {macro_pe.get('fpr', float('nan')):.4f}",
            f"Pixel-level AUC: {float(auc):.4f}",
            f"Rule-6 Dice (positive GT only): {dice_pos.get('point_estimate', float('nan')):.4f}",
            f"Rule-6 Negative clean rate: {neg_clean.get('point_estimate', float('nan')):.4f}",
        ]
    )


def export_results_to_csv(
    *,
    ensemble_recipe: dict[str, Any],
    metrics_results: dict[str, Any],
    output_dir: Path,
    timestamp: str,
    recipe_path: Path,
    dataset_dir: Path,
    seed: int,
    batch_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"FINAL_EVALUATION_REPORT_{timestamp}.csv"
    config_data = {
        "Parameter": [
            "Experiment Datetime",
            "Ensemble Recipe Path",
            "Dataset ZIP Directory",
            "Evaluation Set",
            "Random Seed",
            "Batch Size",
            "Optimal Ensemble Threshold",
        ],
        "Value": [
            timestamp,
            str(recipe_path),
            str(dataset_dir),
            "TEST",
            seed,
            batch_size,
            ensemble_recipe.get("roi_config", {}).get("threshold", "FAILED"),
        ],
    }
    config_df = pd.DataFrame(config_data)

    comp_models = ensemble_recipe.get("model_registry", [])
    composition_df = pd.DataFrame(
        {
            "Model Index": [f"Model {index + 1}" for index in range(len(comp_models))],
            "Architecture": [model.get("architecture") for model in comp_models],
            "Encoder": [model.get("encoder") for model in comp_models],
            "Ensemble Weight": [model.get("weight") for model in comp_models],
            "Source Checkpoint": [
                Path(str(model.get("checkpoint_path", ""))).name for model in comp_models
            ],
            "Stream Role": [model.get("stream_role", "") for model in comp_models],
        }
    )

    micro = metrics_results.get("micro_averaged_metrics", {})
    macro = metrics_results.get("macro_averaged_metrics", {})
    auc = metrics_results.get("auc")
    metric_keys = sorted(micro.get("point_estimate", {}).keys())
    rows: list[dict[str, Any]] = []
    for metric_type, metrics_dict in (
        ("Micro-Average (Pixel Level)", micro),
        ("Macro-Average (Patient Level)", macro),
    ):
        for key in metric_keys:
            ci = metrics_dict.get("ci", {}).get(key, [np.nan, np.nan])
            rows.append(
                {
                    "Metric Type": metric_type,
                    "Metric": key.title(),
                    "Point Estimate": metrics_dict.get("point_estimate", {}).get(key),
                    "CI Lower (2.5%)": ci[0],
                    "CI Upper (97.5%)": ci[1],
                }
            )
    rows.append(
        {
            "Metric Type": "Micro-Average (Pixel Level)",
            "Metric": "AUC",
            "Point Estimate": auc,
            "CI Lower (2.5%)": np.nan,
            "CI Upper (97.5%)": np.nan,
        }
    )
    metrics_df = pd.DataFrame(rows)

    with filepath.open("w", encoding="utf-8", newline="") as handle:
        handle.write("--- Experiment Configuration ---\n")
        config_df.to_csv(handle, index=False)
        handle.write("\n--- Ensemble Composition ---\n")
        composition_df.to_csv(handle, index=False)
        handle.write("\n--- Final Performance Metrics ---\n")
        metrics_df.to_csv(handle, index=False)
    return filepath


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("^", "\\^{}")
        .replace("~", "\\~{}")
    )


def _fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if np.isnan(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):.4f}"


def _fmt_ci(ci_pair: Any, ran_bootstrap: bool) -> str:
    if not ran_bootstrap or ci_pair is None:
        return "NA"
    low, high = ci_pair
    if np.isnan(low) or np.isnan(high):
        return "NA"
    return f"[{low:.4f}, {high:.4f}]"


def _run_pdflatex(command: list[str], cwd: Path) -> None:
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to build the ensemble inference PDF report.")
    subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def write_ensemble_report_latex(
    *,
    ensemble_recipe: dict[str, Any],
    ensemble_metrics: dict[str, Any],
    train_mean: list[float],
    train_std: list[float],
    cm_png_path: Path,
    output_dir: Path,
    timestamp: str,
    report_name_prefix: str = "FINAL_ENSEMBLE_REPORT",
    latex_runner: Callable[[list[str], Path], Any] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{report_name_prefix}_{timestamp}.tex"
    pdf_path = output_dir / f"{report_name_prefix}_{timestamp}.pdf"
    bootstrap = ensemble_metrics.get("bootstrap", {})
    ran_bootstrap = bool(bootstrap.get("ran", False))
    micro = ensemble_metrics.get("micro_averaged_metrics", {})
    macro = ensemble_metrics.get("macro_averaged_metrics", {})
    auc = ensemble_metrics.get("auc")
    rule6 = ensemble_metrics.get("macro_dice_rule6_split", {})
    dice_pos = rule6.get("dice_pos_only", {})
    neg_clean = rule6.get("neg_clean_rate", {})
    metric_order = ["dice", "iou", "tpr", "tnr", "precision", "accuracy", "fpr", "fnr"]
    comp_models = ensemble_recipe.get("model_registry", [])
    threshold = ensemble_recipe.get("roi_config", {}).get("threshold")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\title{Ensemble Final Evaluation Report}",
        r"\author{Automated Pipeline}",
        rf"\date{{{_tex_escape(timestamp)}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Experiment Summary}",
        r"\begin{itemize}",
        (
            rf"\item Ensemble Strategy: "
            f"{_tex_escape(str(ensemble_recipe.get('ensemble_strategy', 'NA')))}"
        ),
        rf"\item Threshold: {_fmt_float(threshold)}",
        rf"\item AUC: {_fmt_float(auc)}",
        r"\end{itemize}",
        r"\section*{Normalization Statistics}",
        rf"Mean: [{train_mean[0]:.6f}, {train_mean[1]:.6f}, {train_mean[2]:.6f}]\\",
        rf"Std: [{train_std[0]:.6f}, {train_std[1]:.6f}, {train_std[2]:.6f}]",
        r"\section*{Ensemble Composition}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{r l l r l}",
        r"\toprule",
        r"\# & Architecture & Encoder & Weight & Role \\",
        r"\midrule",
    ]
    for index, model in enumerate(comp_models, start=1):
        architecture = _tex_escape(str(model.get("architecture", "NA")))
        encoder = _tex_escape(str(model.get("encoder", "NA")))
        weight = _fmt_float(model.get("weight"))
        role = _tex_escape(str(model.get("stream_role", "NA")))
        lines.append(rf"{index} & {architecture} & {encoder} & {weight} & {role} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    for title, metrics_block in (
        ("Final Metrics (Micro / Pixel-Level)", micro),
        ("Final Metrics (Macro / Patient-Level)", macro),
    ):
        lines.extend(
            [
                rf"\section*{{{title}}}",
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{l r l}",
                r"\toprule",
                r"Metric & Point Estimate & 95\% CI \\",
                r"\midrule",
            ]
        )
        for metric in metric_order:
            point_estimate = _fmt_float(metrics_block.get("point_estimate", {}).get(metric))
            confidence_interval = _fmt_ci(metrics_block.get("ci", {}).get(metric), ran_bootstrap)
            lines.append(
                rf"{_tex_escape(metric.upper())} & {point_estimate} & {confidence_interval} \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    lines.extend(
        [
            r"\section*{Rule-6 Dice Reporting}",
            (
                rf"Dice (Positive GT only): {_fmt_float(dice_pos.get('point_estimate'))} "
                f"{_fmt_ci(dice_pos.get('ci'), ran_bootstrap)}\\"
            ),
            (
                rf"Negative Clean Rate: {_fmt_float(neg_clean.get('point_estimate'))} "
                f"{_fmt_ci(neg_clean.get('ci'), ran_bootstrap)}\\"
            ),
            rf"Patients with lesions: {_tex_escape(str(rule6.get('n_pos_patients', 0)))}\\",
            rf"Patients without lesions: {_tex_escape(str(rule6.get('n_neg_patients', 0)))}",
        ]
    )

    cm_target = output_dir / cm_png_path.name
    if cm_png_path.resolve() != cm_target.resolve():
        shutil.copy2(cm_png_path, cm_target)
    lines.extend(
        [
            r"\section*{Confusion Matrix}",
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.75\linewidth]{{{_tex_escape(cm_target.name)}}}",
            r"\end{figure}",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    runner = latex_runner or _run_pdflatex
    command = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    runner(command, output_dir)
    built_pdf = output_dir / f"{tex_path.stem}.pdf"
    if built_pdf.exists() and built_pdf != pdf_path:
        built_pdf.replace(pdf_path)
    if not pdf_path.exists():
        raise RuntimeError(f"Expected PDF report was not created: {pdf_path}")
    return tex_path, pdf_path
