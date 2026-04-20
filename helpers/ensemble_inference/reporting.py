from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import dotenv_values, find_dotenv

_METRIC_ORDER = ("dice", "iou", "tpr", "tnr", "precision", "accuracy", "fpr", "fnr")
_SENSITIVE_ENV_TERMS = ("PASSWORD", "TOKEN", "SECRET", "PRIVATE", "CREDENTIAL")
_EXCLUDED_REPORT_ENV_VARS = {"TRAINING_ARCHITECTURE", "TRAINING_ENCODER"}
_PATHLIKE_SUFFIXES = (
    ".zip",
    ".db",
    ".sqlite",
    ".json",
    ".csv",
    ".h5",
    ".hdf5",
    ".pth",
    ".pt",
    ".ckpt",
    ".log",
    ".svs",
    ".xml",
    ".png",
    ".pdf",
    ".tex",
    ".md",
    ".parquet",
)


@dataclass(frozen=True)
class ReportMetricRow:
    name: str
    point_estimate: str
    confidence_interval: str


@dataclass(frozen=True)
class ReportMetricsSection:
    title: str
    rows: tuple[ReportMetricRow, ...]


@dataclass(frozen=True)
class ReportCompositionRow:
    index: int
    architecture: str
    encoder: str
    weight: str
    role: str


@dataclass(frozen=True)
class EnsembleReportContent:
    timestamp: str
    ensemble_strategy: str
    roi_threshold: str
    decision_threshold: str
    auc: str
    normalization_mean: str
    normalization_std: str
    composition_rows: tuple[ReportCompositionRow, ...]
    metric_sections: tuple[ReportMetricsSection, ...]
    rule6_dice: str
    rule6_dice_ci: str
    rule6_neg_clean_rate: str
    rule6_neg_clean_rate_ci: str
    n_pos_patients: int
    n_neg_patients: int
    confusion_rows: tuple[tuple[str, int, int], ...]


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
                "Master Manifest Path",
                "Evaluation Set",
                "Random Seed",
                "Batch Size",
            "ROI Gate Threshold",
            "Decision Threshold",
        ],
        "Value": [
            timestamp,
            str(recipe_path),
            str(dataset_dir),
            "TEST",
            seed,
            batch_size,
            ensemble_recipe.get("roi_config", {}).get("threshold", "FAILED"),
            ensemble_recipe.get("decision_config", {}).get("threshold", "FAILED"),
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


def _is_pathlike_value(value: str) -> bool:
    stripped = value.strip()
    if stripped == "":
        return False
    lowered = stripped.lower()
    if lowered.endswith(_PATHLIKE_SUFFIXES):
        return True
    if stripped.startswith(("./", "../", "~/", "/")):
        return True
    if "\\" in stripped or "/" in stripped:
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", stripped))


def _should_include_env_var(key: str, value: str | None) -> bool:
    if value is None:
        return False
    normalized_key = key.upper()
    if normalized_key in _EXCLUDED_REPORT_ENV_VARS:
        return False
    if any(term in normalized_key for term in _SENSITIVE_ENV_TERMS):
        return False
    if normalized_key.endswith("_KEY") or normalized_key == "KEY":
        return False
    if any(term in normalized_key for term in ("PATH", "DIR", "FOLDER")):
        return False
    return not _is_pathlike_value(value)


def _load_project_env_values(env_path: Path | None = None) -> tuple[tuple[str, str], ...]:
    resolved_env_path = env_path
    if resolved_env_path is None:
        discovered = find_dotenv(filename=".env", usecwd=True)
        resolved_env_path = Path(discovered) if discovered else None
    if resolved_env_path is None or not resolved_env_path.exists():
        return tuple()

    parsed = dotenv_values(resolved_env_path)
    return _sanitize_project_env_values(parsed.items())


def _sanitize_project_env_values(
    items: Any,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, value)
        for key, value in items
        if key is not None and _should_include_env_var(key, value)
    )


def _build_metrics_rows(
    metrics_block: dict[str, Any],
    *,
    ran_bootstrap: bool,
) -> tuple[ReportMetricRow, ...]:
    return tuple(
        ReportMetricRow(
            name=metric.upper(),
            point_estimate=_fmt_float(metrics_block.get("point_estimate", {}).get(metric)),
            confidence_interval=_fmt_ci(metrics_block.get("ci", {}).get(metric), ran_bootstrap),
        )
        for metric in _METRIC_ORDER
    )


def _build_report_content(
    *,
    ensemble_recipe: dict[str, Any],
    ensemble_metrics: dict[str, Any],
    train_mean: list[float],
    train_std: list[float],
    timestamp: str,
) -> EnsembleReportContent:
    bootstrap = ensemble_metrics.get("bootstrap", {})
    ran_bootstrap = bool(bootstrap.get("ran", False))
    micro = ensemble_metrics.get("micro_averaged_metrics", {})
    macro = ensemble_metrics.get("macro_averaged_metrics", {})
    rule6 = ensemble_metrics.get("macro_dice_rule6_split", {})
    dice_pos = rule6.get("dice_pos_only", {})
    neg_clean = rule6.get("neg_clean_rate", {})
    comp_models = ensemble_recipe.get("model_registry", [])
    confusion = ensemble_metrics.get("confusion_matrix", {})

    return EnsembleReportContent(
        timestamp=timestamp,
        ensemble_strategy=str(ensemble_recipe.get("ensemble_strategy", "NA")),
        roi_threshold=_fmt_float(ensemble_recipe.get("roi_config", {}).get("threshold")),
        decision_threshold=_fmt_float(ensemble_recipe.get("decision_config", {}).get("threshold")),
        auc=_fmt_float(ensemble_metrics.get("auc")),
        normalization_mean=f"[{train_mean[0]:.6f}, {train_mean[1]:.6f}, {train_mean[2]:.6f}]",
        normalization_std=f"[{train_std[0]:.6f}, {train_std[1]:.6f}, {train_std[2]:.6f}]",
        composition_rows=tuple(
            ReportCompositionRow(
                index=index,
                architecture=str(model.get("architecture", "NA")),
                encoder=str(model.get("encoder", "NA")),
                weight=_fmt_float(model.get("weight")),
                role=str(model.get("stream_role", "NA")),
            )
            for index, model in enumerate(comp_models, start=1)
        ),
        metric_sections=(
            ReportMetricsSection(
                title="Final Metrics (Micro / Pixel-Level)",
                rows=_build_metrics_rows(micro, ran_bootstrap=ran_bootstrap),
            ),
            ReportMetricsSection(
                title="Final Metrics (Macro / Patient-Level)",
                rows=_build_metrics_rows(macro, ran_bootstrap=ran_bootstrap),
            ),
        ),
        rule6_dice=_fmt_float(dice_pos.get("point_estimate")),
        rule6_dice_ci=_fmt_ci(dice_pos.get("ci"), ran_bootstrap),
        rule6_neg_clean_rate=_fmt_float(neg_clean.get("point_estimate")),
        rule6_neg_clean_rate_ci=_fmt_ci(neg_clean.get("ci"), ran_bootstrap),
        n_pos_patients=int(rule6.get("n_pos_patients", 0)),
        n_neg_patients=int(rule6.get("n_neg_patients", 0)),
        confusion_rows=(
            ("True NoCancer", int(confusion.get("tn", 0)), int(confusion.get("fp", 0))),
            ("True Cancer", int(confusion.get("fn", 0)), int(confusion.get("tp", 0))),
        ),
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
    report_content = _build_report_content(
        ensemble_recipe=ensemble_recipe,
        ensemble_metrics=ensemble_metrics,
        train_mean=train_mean,
        train_std=train_std,
        timestamp=timestamp,
    )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\title{Ensemble Final Evaluation Report}",
        r"\author{Automated Pipeline}",
        rf"\date{{{_tex_escape(report_content.timestamp)}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Experiment Summary}",
        r"\begin{itemize}",
        rf"\item Ensemble Strategy: {_tex_escape(report_content.ensemble_strategy)}",
        rf"\item ROI Gate Threshold: {report_content.roi_threshold}",
        rf"\item Decision Threshold: {report_content.decision_threshold}",
        rf"\item AUC: {report_content.auc}",
        r"\end{itemize}",
        r"\section*{Normalization Statistics}",
        rf"Mean: {report_content.normalization_mean}\\",
        rf"Std: {report_content.normalization_std}",
        r"\section*{Ensemble Composition}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{r l l r l}",
        r"\toprule",
        r"\# & Architecture & Encoder & Weight & Role \\",
        r"\midrule",
    ]
    for composition_row in report_content.composition_rows:
        lines.append(
            rf"{composition_row.index} & {_tex_escape(composition_row.architecture)} "
            rf"& {_tex_escape(composition_row.encoder)} & {composition_row.weight} "
            rf"& {_tex_escape(composition_row.role)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    for section in report_content.metric_sections:
        lines.extend(
            [
                rf"\section*{{{section.title}}}",
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{l r l}",
                r"\toprule",
                r"Metric & Point Estimate & 95\% CI \\",
                r"\midrule",
            ]
        )
        for metric_row in section.rows:
            lines.append(
                rf"{_tex_escape(metric_row.name)} & {metric_row.point_estimate} "
                rf"& {metric_row.confidence_interval} \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    lines.extend(
        [
            r"\section*{Rule-6 Dice Reporting}",
            (
                rf"Dice (Positive GT only): {report_content.rule6_dice} "
                f"{report_content.rule6_dice_ci}\\"
            ),
            (
                rf"Negative Clean Rate: {report_content.rule6_neg_clean_rate} "
                f"{report_content.rule6_neg_clean_rate_ci}\\"
            ),
            rf"Patients with lesions: {_tex_escape(str(report_content.n_pos_patients))}\\",
            rf"Patients without lesions: {_tex_escape(str(report_content.n_neg_patients))}",
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


def write_ensemble_report_markdown(
    *,
    ensemble_recipe: dict[str, Any],
    ensemble_metrics: dict[str, Any],
    train_mean: list[float],
    train_std: list[float],
    output_dir: Path,
    timestamp: str,
    project_env_vars: tuple[tuple[str, str], ...] | None = None,
    report_name_prefix: str = "FINAL_ENSEMBLE_REPORT",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{report_name_prefix}_{timestamp}.md"
    report_content = _build_report_content(
        ensemble_recipe=ensemble_recipe,
        ensemble_metrics=ensemble_metrics,
        train_mean=train_mean,
        train_std=train_std,
        timestamp=timestamp,
    )
    env_rows = (
        _sanitize_project_env_values(project_env_vars)
        if project_env_vars is not None
        else _load_project_env_values()
    )

    lines = [
        "# Ensemble Final Evaluation Report",
        "",
        f"Date: {report_content.timestamp}",
        "",
        "## Experiment Summary",
        f"- Ensemble Strategy: {report_content.ensemble_strategy}",
        f"- ROI Gate Threshold: {report_content.roi_threshold}",
        f"- Decision Threshold: {report_content.decision_threshold}",
        f"- AUC: {report_content.auc}",
        "",
        "## Normalization Statistics",
        f"Mean: {report_content.normalization_mean}",
        f"Std: {report_content.normalization_std}",
        "",
        "## Ensemble Composition",
        "| # | Architecture | Encoder | Weight | Role |",
        "| --- | --- | --- | --- | --- |",
    ]
    for composition_row in report_content.composition_rows:
        lines.append(
            "| "
            f"{composition_row.index} | {composition_row.architecture} | "
            f"{composition_row.encoder} | {composition_row.weight} | {composition_row.role} |"
        )

    for section in report_content.metric_sections:
        lines.extend(
            [
                "",
                f"## {section.title}",
                "| Metric | Point Estimate | 95% CI |",
                "| --- | --- | --- |",
            ]
        )
        for metric_row in section.rows:
            lines.append(
                f"| {metric_row.name} | {metric_row.point_estimate} | "
                f"{metric_row.confidence_interval} |"
            )

    lines.extend(
        [
            "",
            "## Rule-6 Dice Reporting",
            f"Dice (Positive GT only): {report_content.rule6_dice} {report_content.rule6_dice_ci}",
            (
                f"Negative Clean Rate: {report_content.rule6_neg_clean_rate} "
                f"{report_content.rule6_neg_clean_rate_ci}"
            ),
            f"Patients with lesions: {report_content.n_pos_patients}",
            f"Patients without lesions: {report_content.n_neg_patients}",
            "",
            "## Confusion Matrix",
            "| True Label | Pred NoCancer | Pred Cancer |",
            "| --- | --- | --- |",
        ]
    )
    for true_label, pred_no_cancer, pred_cancer in report_content.confusion_rows:
        lines.append(f"| {true_label} | {pred_no_cancer} | {pred_cancer} |")

    lines.extend(["", "## Environment Variables Used For This Project"])
    if env_rows:
        lines.extend(["| Variable | Value |", "| --- | --- |"])
        for key, value in env_rows:
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("No sanitized project .env variables were available.")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
