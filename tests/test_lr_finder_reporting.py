from __future__ import annotations

from pathlib import Path

from helpers.lr_finder.reporting import RunRecord, build_latex_report

LATEX_RUNNER_CALLS = 2


def test_build_latex_report_writes_tex_and_pdf(tmp_path: Path) -> None:
    plot_path = tmp_path / "FPN" / "curve.png"
    plot_path.parent.mkdir(parents=True)
    plot_path.write_bytes(b"png")
    records = [
        RunRecord(
            architecture="FPN",
            encoder="senet154",
            alpha=0.125,
            beta=0.2,
            gamma=0.3,
            median_min_loss=0.4567,
            plot_path=plot_path,
            csv_path=tmp_path / "summary.csv",
        )
    ]

    calls: list[list[str]] = []

    def fake_runner(command: list[str], cwd: Path) -> None:
        calls.append(command)
        (cwd / "screening.pdf").write_bytes(b"pdf")

    tex_path, pdf_path = build_latex_report(
        records,
        tmp_path,
        meta={"timestamp": "2026-03-20 12:00:00"},
        pdf_name="screening.pdf",
        latex_runner=fake_runner,
    )

    assert tex_path == tmp_path / "screening.tex"
    assert pdf_path == tmp_path / "screening.pdf"
    assert pdf_path.exists()
    assert len(calls) == LATEX_RUNNER_CALLS
    tex_content = tex_path.read_text(encoding="utf-8")
    assert "\\section{FPN}" in tex_content
    assert "senet154" in tex_content
    assert "0.4567" in tex_content
