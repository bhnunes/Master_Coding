from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from helpers.systematic_review import (
    ReviewRecord,
    build_accepted_studies,
    build_assets,
    latex_escape,
    load_protocol_data,
    load_review_records,
    make_bib_key,
    summarize_screening,
    thesis_verdict,
    validate_accepted_markdowns,
)

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "systematic_review" / "database" / "automatic_review.sqlite3"
PAPERS_DIR = ROOT / "systematic_review" / "PAPERS"
PROTOCOL = ROOT / "systematic_review" / "protocol" / "protocol.md"
TOTAL_RECORDS = 1004
ACCEPTED_RECORDS = 61
REJECTED_RECORDS = 943
FAILED_RECORDS = 2
RESEARCH_QUESTION_COUNT = 4
SEARCH_QUERY_COUNT = 3
IEEE_COUNTS = (6, 0, 6)
PUBMED_COUNTS = (204, 8, 196)
SPRINGER_COUNTS = (794, 53, 741)


def test_database_counts_and_source_breakdown() -> None:
    records = load_review_records(DATABASE)
    counts = {count.origin: count for count in summarize_screening(records)}

    assert counts["TOTAL"].total == TOTAL_RECORDS
    assert counts["TOTAL"].accepted == ACCEPTED_RECORDS
    assert counts["TOTAL"].rejected == REJECTED_RECORDS
    assert counts["TOTAL"].raw_failed == FAILED_RECORDS
    assert (counts["IEEE"].total, counts["IEEE"].accepted, counts["IEEE"].rejected) == IEEE_COUNTS
    assert (counts["PUBMED"].total, counts["PUBMED"].accepted, counts["PUBMED"].rejected) == (
        PUBMED_COUNTS
    )
    assert (counts["SPRINGER"].total, counts["SPRINGER"].accepted, counts["SPRINGER"].rejected) == (
        SPRINGER_COUNTS
    )


def test_failed_records_are_thesis_rejected_but_raw_status_is_visible() -> None:
    records = load_review_records(DATABASE)
    failed_records = [record for record in records if record.process_status == "FAILED"]

    assert len(failed_records) == FAILED_RECORDS
    assert all(thesis_verdict(record) == "REJECTED" for record in failed_records)
    assert {record.final_verdict for record in failed_records} == {None}


def test_accepted_rows_match_local_markdown_files() -> None:
    records = load_review_records(DATABASE)
    validate_accepted_markdowns(records, PAPERS_DIR)

    studies = build_accepted_studies(records, PAPERS_DIR)

    assert len(studies) == ACCEPTED_RECORDS
    assert all((PAPERS_DIR / study.markdown_file).exists() for study in studies)
    assert len({study.markdown_file for study in studies}) == ACCEPTED_RECORDS


def test_duplicate_accepted_basenames_fail_validation(tmp_path: Path) -> None:
    record = _record("same.pdf", "ACCEPTED", "VERDICT_GENERATED")
    duplicate = _record("same.pdf", "ACCEPTED", "VERDICT_GENERATED")
    (tmp_path / "same.md").write_text("## Example accepted paper", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate accepted Markdown basenames"):
        validate_accepted_markdowns([record, duplicate], tmp_path)


def test_deterministic_citation_keys_and_latex_escaping() -> None:
    assert make_bib_key("10.1038/s41467-024-54769-y", "fallback.md") == (
        "sysrev_10_1038_s41467_024_54769_y"
    )
    assert latex_escape(r"A&B_50% #1") == r"A\&B\_50\% \#1"


def test_raw_database_still_has_two_failed_rows() -> None:
    with sqlite3.connect(DATABASE) as connection:
        failed_count = connection.execute(
            "SELECT COUNT(*) FROM review_files WHERE process_status = 'FAILED'"
        ).fetchone()[0]

    assert failed_count == FAILED_RECORDS


def test_protocol_queries_and_criteria_are_loaded_from_protocol_file() -> None:
    protocol = load_protocol_data(PROTOCOL)

    assert len(protocol.research_questions) == RESEARCH_QUESTION_COUNT
    assert len(protocol.queries) == SEARCH_QUERY_COUNT
    assert any("Whole Slide Imaging" in query.query for query in protocol.queries)
    assert any("public repository" in criterion for criterion in protocol.inclusion_criteria)
    assert any("Pre-Extracted Patch" in criterion for criterion in protocol.exclusion_criteria)


def test_generated_chapter_tables_and_step5_matrix_are_thesis_ready(tmp_path: Path) -> None:
    outputs = build_assets(
        database_path=DATABASE,
        papers_dir=PAPERS_DIR,
        thesis_dir=tmp_path / "thesis",
        review_generated_dir=tmp_path / "review_generated",
        protocol_path=PROTOCOL,
    )

    matrix_text = outputs.accepted_matrix_tex.read_text(encoding="utf-8")
    chapter_summary = outputs.protocol_summary_tex.read_text(encoding="utf-8")
    stale_appendix = tmp_path / "thesis" / "generated" / "systematic_review_protocol_appendix.tex"

    assert not stale_appendix.exists()
    assert "Como fluxos recentes de WSI tratam artefatos" in chapter_summary
    assert "Aplicação de aprendizagem profunda" in chapter_summary
    assert "Consultas exatas aplicadas" in chapter_summary
    assert '"Whole Slide Imaging"[MeSH]' in chapter_summary
    assert '"2025/12/31"[Date - Publication]' in chapter_summary
    assert '"Document Title":histopathology' in chapter_summary
    assert '"stain normalization" OR "reproducibility"' in chapter_summary
    assert "Base & Consulta exata" in chapter_summary
    assert "Domain: Application" not in chapter_summary
    assert "How do recent" not in chapter_summary
    assert "FINAL VERDICT" not in chapter_summary
    for stale_phrase in (
        "Publicação em inglês",
        "filtro de acesso aberto",
        "inglês; acesso aberto; artigo; 2020--2026",
        "Base e filtros",
    ):
        assert stale_phrase not in chapter_summary
    for heading in (
        "Referência / ano",
        "Artefatos e QC",
        "Normalização",
        "Particionamento",
        "Armazenamento e governança",
        "Ensemble ou pós-processamento",
        "Código",
    ):
        assert heading in matrix_text
    assert "automatic_review.sqlite3" not in chapter_summary
    assert "accepted_studies.csv" not in chapter_summary


def _record(file_name: str, verdict: str | None, status: str) -> ReviewRecord:
    return ReviewRecord(
        file_name=file_name,
        file_path=f"/tmp/{file_name}",
        origin="PUBMED",
        subject="cancer",
        process_status=status,
        markdown_path=f"/tmp/{Path(file_name).with_suffix('.md').name}",
        final_verdict=verdict,
        comments=None,
    )
