from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DECIMAL_FORMAT_LOWER_BOUND = 1e-2
DECIMAL_FORMAT_UPPER_BOUND = 1e3


@dataclass(frozen=True)
class RunRecord:
    architecture: str
    encoder: str
    alpha: float
    beta: float
    gamma: float
    median_min_loss: float
    plot_path: Path
    csv_path: Path
    effective_batch_size: int | None = None


def format_metric(value: float) -> str:
    if value != value or value in {float("inf"), float("-inf")}:  # noqa: PLR0124
        return "NA"
    if DECIMAL_FORMAT_LOWER_BOUND <= value < DECIMAL_FORMAT_UPPER_BOUND:
        return f"{value:.4f}"
    return f"{value:.2e}"


def latex_escape(value: str) -> str:
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


def _run_pdflatex(command: list[str], cwd: Path) -> None:
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to build the LR Finder PDF report.")
    subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def build_latex_report(
    records: list[RunRecord],
    out_dir: Path,
    meta: dict[str, str],
    pdf_name: str = "report.pdf",
    *,
    latex_runner: Callable[[list[str], Path], Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_stem = Path(pdf_name).stem
    tex_path = out_dir / f"{report_stem}.tex"
    pdf_path = out_dir / pdf_name
    by_arch: dict[str, list[RunRecord]] = {}
    for record in records:
        by_arch.setdefault(record.architecture, []).append(record)

    tex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1.0cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\begin{document}",
        r"\title{LR Finder Screening}",
        rf"\date{{{latex_escape(meta.get('timestamp', ''))}}}",
        r"\maketitle",
    ]
    for architecture, arch_records in sorted(by_arch.items()):
        tex_lines.append(rf"\section{{{latex_escape(architecture)}}}")
        tex_lines.append(rf"\noindent\textbf{{Encoder:}} {latex_escape(arch_records[0].encoder)}\\")
        tex_lines.append(r"\medskip")
        tex_lines.append(r"\begin{longtable}{lll|r}")
        tex_lines.append(r"\caption{Configs sorted by minimum loss (best first)}\\")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"$\alpha$ & $\beta$ & $\gamma$ & \textbf{Min Loss} \\")
        tex_lines.append(r"\midrule")
        tex_lines.append(r"\endfirsthead")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"$\alpha$ & $\beta$ & $\gamma$ & \textbf{Min Loss} \\")
        tex_lines.append(r"\midrule")
        tex_lines.append(r"\endhead")
        sorted_records = sorted(arch_records, key=lambda item: item.median_min_loss)
        for index, record in enumerate(sorted_records):
            row_prefix = r"\textbf{" if index == 0 else ""
            row_suffix = "}" if index == 0 else ""
            tex_lines.append(
                rf"{row_prefix}{record.alpha:.4f}{row_suffix} & "
                rf"{row_prefix}{record.beta:.4f}{row_suffix} & "
                rf"{row_prefix}{record.gamma:.4f}{row_suffix} & "
                rf"{row_prefix}{format_metric(record.median_min_loss)}{row_suffix} \\"
            )
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{longtable}")
        tex_lines.append(r"\subsection*{Curves}")
        for index, record in enumerate(sorted_records):
            tex_lines.append(r"\begin{figure}[H]")
            tex_lines.append(r"\centering")
            tex_lines.append(
                rf"\includegraphics[width=0.75\linewidth]{{{latex_escape(str(record.plot_path.relative_to(out_dir)))}}}"
            )
            caption = (
                f"Rank {index + 1}: BCE(a={record.alpha:.4f}, b={record.beta:.4f}, "
                f"g={record.gamma:.4f}). Min Loss={format_metric(record.median_min_loss)}."
            )
            tex_lines.append(rf"\caption{{{latex_escape(caption)}}}")
            tex_lines.append(r"\end{figure}")
        tex_lines.append(r"\clearpage")
    tex_lines.append(r"\end{document}")
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    runner = latex_runner or _run_pdflatex
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    runner(command, out_dir)
    runner(command, out_dir)
    if not pdf_path.exists():
        raise RuntimeError(f"Expected PDF report was not created: {pdf_path}")
    return tex_path, pdf_path
