from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"
SUMMARY_ORDER = ("IEEE", "PUBMED", "SPRINGER")
MIN_TITLE_HEADING_LENGTH = 18
GOVERNANCE_TERMS = ("hdf5", "zarr", "sqlite", "parquet", "manifest", "nextflow")
PROTOCOL_QUERY_MARKERS = {
    "A. PubMed / MEDLINE": "PubMed / MEDLINE",
    "B. IEEE Xplore": "IEEE Xplore",
    "C. SpringerLink": "SpringerLink",
}
PROTOCOL_QUERY_TABLE_LABELS = {
    "PubMed / MEDLINE": "PubMed / MEDLINE",
    "IEEE Xplore": "IEEE Xplore",
    "SpringerLink": "SpringerLink",
}
PORTUGUESE_RESEARCH_QUESTIONS = (
    (
        "RQ1",
        (
            "Como fluxos recentes de WSI tratam artefatos de lâmina, contaminação por "
            "fundo e necrose antes da extração de amostras para aprendizagem profunda?"
        ),
    ),
    (
        "RQ2",
        (
            "Como a literatura aborda o particionamento por paciente e o risco de "
            "vazamento de dados durante pré-processamentos aplicados ao conjunto, como "
            "normalização de coloração?"
        ),
    ),
    (
        "RQ3",
        (
            "Em que medida os fluxos contemporâneos implementam contratos explícitos de "
            "dados, orquestração rastreável e proveniência automatizada?"
        ),
    ),
    (
        "RQ4",
        (
            "Quais estratégias são usadas para agregação de múltiplos modelos em "
            "histopatologia e como a receita final é otimizada em validação?"
        ),
    ),
)
PORTUGUESE_INCLUSION_CRITERIA = (
    "Aplicação de aprendizagem profunda a histopatologia de lâmina inteira.",
    "Foco explícito em detecção, segmentação ou classificação de câncer, neoplasias ou tumores.",
    (
        "Descrição de pelo menos um componente operacional do fluxo: controle de qualidade "
        "em lâmina, normalização de coloração, particionamento por paciente, serialização "
        "de dados ou agregação de modelos."
    ),
    "Disponibilização de repositório público com a implementação do fluxo.",
)
PORTUGUESE_EXCLUSION_CRITERIA = (
    "Estudos puramente clínicos ou centrados em doenças não neoplásicas.",
    (
        "Uso exclusivo de bases de amostras pré-extraídas que eliminam os desafios de "
        "processamento de WSI."
    ),
    (
        "Particionamento por amostra ou por lâmina, ou normalização global antes da "
        "separação treino-validação-teste."
    ),
    (
        "Trabalhos focados apenas em arquitetura neural, sem documentação do fluxo de "
        "extração, proveniência ou orquestração; ausência de código público verificável."
    ),
)


@dataclass(frozen=True)
class ReviewRecord:
    """One raw record from the automatic-review SQLite database."""

    file_name: str
    file_path: str
    origin: str
    subject: str
    process_status: str
    markdown_path: str | None
    final_verdict: str | None
    comments: str | None


@dataclass(frozen=True)
class ScreeningCount:
    """Thesis-facing screening count for one source."""

    origin: str
    total: int
    accepted: int
    rejected: int
    raw_failed: int


@dataclass(frozen=True)
class ProtocolQuery:
    """One reproducible search query from the systematic-review protocol."""

    database: str
    query: str


@dataclass(frozen=True)
class ProtocolData:
    """Structured protocol content used in the thesis chapter."""

    research_questions: tuple[str, ...]
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    queries: tuple[ProtocolQuery, ...]


@dataclass(frozen=True)
class AcceptedStudy:
    """Extracted thesis matrix row for an accepted paper."""

    study_id: str
    bib_key: str
    doi: str
    year: str
    title: str
    source: str
    cancer_domain: str
    primary_task: str
    artifact_qc_strategy: str
    normalization_policy: str
    partitioning_policy: str
    storage_governance: str
    ensemble_postprocessing: str
    code_url: str
    rq_tags: str
    chapter_summary: str
    markdown_file: str
    raw_process_status: str


@dataclass(frozen=True)
class BuildOutputs:
    """Paths produced by the systematic-review asset builder."""

    protocol_summary_tex: Path
    screening_tables_tex: Path
    findings_summary_tex: Path
    accepted_matrix_tex: Path
    screening_plot_png: Path
    accepted_studies_csv: Path
    bibliography_bib: Path


def thesis_verdict(record: ReviewRecord) -> str:
    """Return the thesis-facing verdict, with failed automation counted as rejected."""

    return ACCEPTED if record.final_verdict == ACCEPTED else REJECTED


def load_review_records(database_path: Path) -> list[ReviewRecord]:
    """Load review records from the automatic-review SQLite database."""

    query = """
        SELECT file_name, file_path, origin, subject, process_status,
               markdown_path, final_verdict, comments
        FROM review_files
        ORDER BY origin, file_name
    """
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(query).fetchall()

    return [
        ReviewRecord(
            file_name=str(row[0]),
            file_path=str(row[1]),
            origin=str(row[2]),
            subject=str(row[3]),
            process_status=str(row[4]),
            markdown_path=None if row[5] is None else str(row[5]),
            final_verdict=None if row[6] is None else str(row[6]),
            comments=None if row[7] is None else str(row[7]),
        )
        for row in rows
    ]


def summarize_screening(records: list[ReviewRecord]) -> list[ScreeningCount]:
    """Summarize records by source using thesis-facing verdict semantics."""

    by_origin: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in records:
        by_origin[record.origin].append(record)

    counts = [_count_origin(origin, by_origin[origin]) for origin in sorted(by_origin)]
    return [*counts, _count_origin("TOTAL", records)]


def validate_accepted_markdowns(records: list[ReviewRecord], papers_dir: Path) -> None:
    """Validate accepted records against local Markdown files."""

    accepted = [record for record in records if thesis_verdict(record) == ACCEPTED]
    names = [markdown_name_for_record(record) for record in accepted]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate accepted Markdown basenames: {', '.join(duplicates)}")

    missing = [name for name in names if not (papers_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing accepted Markdown files: {', '.join(sorted(missing))}")


def build_accepted_studies(records: list[ReviewRecord], papers_dir: Path) -> list[AcceptedStudy]:
    """Build one extraction-matrix row per accepted paper."""

    validate_accepted_markdowns(records, papers_dir)
    accepted = [record for record in records if thesis_verdict(record) == ACCEPTED]
    return [
        _extract_study(record, papers_dir / markdown_name_for_record(record))
        for record in sorted(accepted, key=lambda item: (item.origin, item.file_name))
    ]


def build_assets(
    database_path: Path,
    papers_dir: Path,
    thesis_dir: Path,
    review_generated_dir: Path,
    protocol_path: Path | None = None,
) -> BuildOutputs:
    """Generate CSV, LaTeX, BibTeX, and plot assets for the thesis chapter."""

    if protocol_path is None:
        protocol_path = database_path.parents[1] / "protocol" / "protocol.md"

    records = load_review_records(database_path)
    counts = summarize_screening(records)
    studies = build_accepted_studies(records, papers_dir)
    protocol = load_protocol_data(protocol_path)

    thesis_generated_dir = thesis_dir / "generated"
    figures_dir = thesis_dir / "figuras"
    thesis_generated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    review_generated_dir.mkdir(parents=True, exist_ok=True)

    outputs = BuildOutputs(
        protocol_summary_tex=thesis_generated_dir / "systematic_review_protocol_summary.tex",
        screening_tables_tex=thesis_generated_dir / "systematic_review_screening_tables.tex",
        findings_summary_tex=thesis_generated_dir / "systematic_review_findings_summary.tex",
        accepted_matrix_tex=thesis_generated_dir / "systematic_review_accepted_matrix.tex",
        screening_plot_png=figures_dir / "systematic_review_screening.png",
        accepted_studies_csv=review_generated_dir / "accepted_studies.csv",
        bibliography_bib=thesis_dir / "bibliografia_revisao_sistematica.bib",
    )
    stale_protocol_appendix = thesis_generated_dir / "systematic_review_protocol_appendix.tex"
    if stale_protocol_appendix.exists():
        stale_protocol_appendix.unlink()

    write_protocol_summary(protocol, outputs.protocol_summary_tex)
    write_screening_tables(counts, outputs.screening_tables_tex)
    write_findings_summary(studies, outputs.findings_summary_tex)
    write_accepted_matrix(studies, outputs.accepted_matrix_tex)
    write_accepted_studies_csv(studies, outputs.accepted_studies_csv)
    write_bibliography(studies, outputs.bibliography_bib)
    write_screening_plot(counts, outputs.screening_plot_png)
    return outputs


def load_protocol_data(protocol_path: Path) -> ProtocolData:
    """Load the protocol elements needed for the chapter-facing summary."""

    lines = protocol_path.read_text(encoding="utf-8").splitlines()
    return ProtocolData(
        research_questions=tuple(line.strip() for line in lines if line.startswith("RQ")),
        inclusion_criteria=tuple(
            _protocol_block(lines, "Inclusion Criteria:", "Exclusion Criteria:")
        ),
        exclusion_criteria=tuple(
            _protocol_block(lines, "Exclusion Criteria:", "3. Databases and Exact Search Queries")
        ),
        queries=tuple(_protocol_queries(lines)),
    )


def markdown_name_for_record(record: ReviewRecord) -> str:
    """Return the local Markdown filename for one review record."""

    if record.markdown_path:
        return Path(record.markdown_path).name
    return f"{Path(record.file_name).stem}.md"


def make_bib_key(doi: str, fallback_name: str) -> str:
    """Make a deterministic BibLaTeX key for one accepted study."""

    source = doi if doi else Path(fallback_name).stem
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower()
    return f"sysrev_{cleaned}"


def latex_escape(value: str) -> str:
    """Escape text for safe use in regular LaTeX text mode."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_screening_tables(counts: list[ScreeningCount], output_path: Path) -> None:
    """Write thesis-facing screening tables."""

    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Resultado da triagem por base consultada.}",
        r"\label{tab:revisao_sistematica_triagem}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Base & Registros & Aceitos & Rejeitados & Falhas agregadas \\",
        r"\midrule",
    ]
    for count in counts:
        lines.append(_screening_table_row(count))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol_summary(protocol: ProtocolData, output_path: Path) -> None:
    """Write compact, chapter-facing protocol tables."""

    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Perguntas de pesquisa da revisão sistemática.}",
        r"\label{tab:revisao_sistematica_perguntas}",
        r"\small",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.0cm}"
        r">{\raggedright\arraybackslash}p{11.1cm}}",
        r"\toprule",
        r"Código & Pergunta \\",
        r"\midrule",
    ]
    for code, question_text in PORTUGUESE_RESEARCH_QUESTIONS:
        lines.append(f"{latex_escape(code)} & {latex_escape(question_text)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}", ""])

    lines.extend(
        [
            r"\begin{table}[!htb]",
            r"\centering",
            r"\caption{Critérios de elegibilidade aplicados na revisão.}",
            r"\label{tab:revisao_sistematica_criterios}",
            r"\scriptsize",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{6.35cm}"
            r">{\raggedright\arraybackslash}p{6.35cm}}",
            r"\toprule",
            r"Inclusão & Exclusão \\",
            r"\midrule",
        ]
    )
    max_rows = max(len(PORTUGUESE_INCLUSION_CRITERIA), len(PORTUGUESE_EXCLUSION_CRITERIA))
    for index in range(max_rows):
        inclusion = (
            PORTUGUESE_INCLUSION_CRITERIA[index]
            if index < len(PORTUGUESE_INCLUSION_CRITERIA)
            else ""
        )
        exclusion = (
            PORTUGUESE_EXCLUSION_CRITERIA[index]
            if index < len(PORTUGUESE_EXCLUSION_CRITERIA)
            else ""
        )
        lines.append(f"{latex_escape(inclusion)} & {latex_escape(exclusion)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}", ""])

    lines.extend(
        [
            r"\begin{table}[!htb]",
            r"\centering",
            r"\caption{Consultas exatas aplicadas na revisão sistemática.}",
            r"\label{tab:revisao_sistematica_consultas}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.8cm}"
            r">{\raggedright\arraybackslash}p{10.45cm}}",
            r"\toprule",
            r"Base & Consulta exata \\",
            r"\midrule",
        ]
    )
    for query in protocol.queries:
        lines.append(
            f"{latex_escape(PROTOCOL_QUERY_TABLE_LABELS[query.database])} "
            f"& {latex_escape(query.query)} \\\\[0.35em]"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_findings_summary(studies: list[AcceptedStudy], output_path: Path) -> None:
    """Write a compact synthesis table grouped by protocol research question."""

    artifact = Counter(study.artifact_qc_strategy for study in studies)
    normalization = Counter(study.normalization_policy for study in studies)
    partitioning = Counter(study.partitioning_policy for study in studies)
    governance = Counter(study.storage_governance for study in studies)
    ensemble = Counter(study.ensemble_postprocessing for study in studies)

    rows = [
        (
            "RQ1",
            (
                f"{artifact['controle de qualidade ou artefatos explicitado']} estudos explicitam "
                "controle de qualidade ou artefatos; "
                f"{artifact['segmentação de tecido/fundo']} tratam tecido/fundo como etapa "
                "operacional."
            ),
            (
                "QC aparece como componente recorrente, mas com profundidade variável; por isso, "
                "deve ser tratado como contrato verificável do fluxo."
            ),
        ),
        (
            "RQ2",
            (
                f"{partitioning['separação por paciente']} estudos declaram separação por "
                "paciente; "
                f"{sum(normalization.values()) - normalization['não especificada']} mencionam "
                "normalização de cor."
            ),
            (
                "A literatura reconhece particionamento e normalização como fontes de viés, "
                "mas nem sempre explicita a ordem operacional das etapas."
            ),
        ),
        (
            "RQ3",
            (
                f"{len(studies) - governance['governança não detalhada']} estudos apresentam algum "
                "sinal de governança, repositório público, manifesto ou formato persistente."
            ),
            (
                "A reprodutibilidade depende de rastrear lâmina, paciente, amostra e configuração, "
                "não apenas de publicar pesos de modelo."
            ),
        ),
        (
            "RQ4",
            (
                f"{ensemble['ensemble/incerteza discutido']} estudos discutem ensemble ou "
                "incerteza; "
                f"{ensemble['pós-processamento descrito']} enfatizam pós-processamento."
            ),
            (
                "Agregação de modelos é metodologicamente defensável quando a receita é definida "
                "em validação e congelada antes do teste."
            ),
        ),
    ]

    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Síntese dos achados organizados pelas perguntas de pesquisa.}",
        r"\label{tab:revisao_sistematica_sintese}",
        r"\small",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.2cm}"
        r">{\raggedright\arraybackslash}p{5.9cm}"
        r">{\raggedright\arraybackslash}p{6.0cm}}",
        r"\toprule",
        r"RQ & Achado agregado & Implicação para a dissertação \\",
        r"\midrule",
    ]
    for code, finding, implication in rows:
        lines.append(
            f"{latex_escape(code)} & {latex_escape(finding)} & {latex_escape(implication)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_accepted_matrix(studies: list[AcceptedStudy], output_path: Path) -> None:
    """Write the accepted-paper extraction matrix required by protocol step 5."""

    lines = [
        "% Corpus-complete accepted-study bibliography coverage.",
        *[rf"\nocite{{{study.bib_key}}}" for study in studies],
        r"\begin{landscape}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\emergencystretch=2em",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{4.2cm}"
        r">{\raggedright\arraybackslash}p{3.1cm}"
        r">{\raggedright\arraybackslash}p{3.0cm}"
        r">{\raggedright\arraybackslash}p{3.0cm}"
        r">{\raggedright\arraybackslash}p{3.2cm}"
        r">{\raggedright\arraybackslash}p{3.1cm}"
        r">{\raggedright\arraybackslash}p{3.2cm}}",
        r"\caption{Matriz de extração dos estudos aceitos conforme o protocolo da revisão.}",
        r"\label{tab:revisao_sistematica_matriz}\\[0.6em]",
        r"\toprule",
        r"Referência / ano & Artefatos e QC & Normalização & Particionamento & "
        r"Armazenamento e governança & Ensemble ou pós-processamento & Código \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Matriz de extração dos estudos aceitos conforme o protocolo da revisão "
        r"(continuação).}\\[0.6em]",
        r"\toprule",
        r"Referência / ano & Artefatos e QC & Normalização & Particionamento & "
        r"Armazenamento e governança & Ensemble ou pós-processamento & Código \\",
        r"\midrule",
        r"\endhead",
    ]
    lines.extend(_accepted_matrix_row(study) for study in studies)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\renewcommand{\arraystretch}{1.0}",
            r"\endgroup",
            r"\end{landscape}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_accepted_studies_csv(studies: list[AcceptedStudy], output_path: Path) -> None:
    """Write an auditable CSV extraction matrix."""

    fieldnames = list(AcceptedStudy.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for study in studies:
            writer.writerow({field: getattr(study, field) for field in fieldnames})


def write_bibliography(studies: list[AcceptedStudy], output_path: Path) -> None:
    """Write deterministic BibLaTeX entries for the accepted studies."""

    entries = [_bib_entry(study) for study in studies]
    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def write_screening_plot(counts: list[ScreeningCount], output_path: Path) -> None:
    """Write a stacked bar chart for the thesis screening summary."""

    from matplotlib import pyplot as plt

    source_counts = [count for count in counts if count.origin != "TOTAL"]
    labels = [_display_origin(count.origin) for count in source_counts]
    accepted = [count.accepted for count in source_counts]
    rejected = [count.rejected for count in source_counts]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(labels, rejected, label="Rejeitados", color="#9f5f5f")
    ax.bar(labels, accepted, bottom=rejected, label="Aceitos", color="#3a7d75")
    ax.set_ylabel("Registros")
    ax.set_title("Triagem da revisão sistemática por base consultada")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    for index, count in enumerate(source_counts):
        ax.text(index, count.total + 8, str(count.total), ha="center", va="bottom", fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _count_origin(origin: str, records: list[ReviewRecord]) -> ScreeningCount:
    accepted = sum(thesis_verdict(record) == ACCEPTED for record in records)
    failed = sum(record.process_status == "FAILED" for record in records)
    return ScreeningCount(
        origin=origin,
        total=len(records),
        accepted=accepted,
        rejected=len(records) - accepted,
        raw_failed=failed,
    )


def _extract_study(record: ReviewRecord, markdown_path: Path) -> AcceptedStudy:
    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    doi = _extract_doi(text, markdown_path.name)
    title = _extract_title(text, markdown_path.name)
    bib_key = make_bib_key(doi, markdown_path.name)
    evidence = _classify_evidence(text)
    return AcceptedStudy(
        study_id=Path(record.file_name).stem,
        bib_key=bib_key,
        doi=doi,
        year=_extract_year(text, doi),
        title=title,
        source=record.origin,
        cancer_domain=_classify_domain(text),
        primary_task=_classify_task(text),
        artifact_qc_strategy=evidence["artifact_qc_strategy"],
        normalization_policy=evidence["normalization_policy"],
        partitioning_policy=evidence["partitioning_policy"],
        storage_governance=evidence["storage_governance"],
        ensemble_postprocessing=evidence["ensemble_postprocessing"],
        code_url=_extract_code_url(text),
        rq_tags=_rq_tags(evidence),
        chapter_summary=_chapter_summary(text),
        markdown_file=markdown_path.name,
        raw_process_status=record.process_status,
    )


def _extract_doi(text: str, markdown_name: str) -> str:
    stem = Path(markdown_name).stem
    if stem.startswith("10."):
        prefix, _, suffix = stem.partition("_")
        if suffix:
            return _clean_doi(f"{prefix}/{suffix}")

    article_text = _text_before_references(text)
    pattern = r"(?:https?://doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s<>)\]]+)"
    match = re.search(pattern, article_text, re.I)
    if match:
        return _clean_doi(match.group(1))
    return ""


def _clean_doi(value: str) -> str:
    cleaned = re.sub(r"\s+", "", value)
    cleaned = cleaned.replace("\\_", "_").replace("&amp;", "&")
    return cleaned.rstrip(".,;]")


def _extract_title(text: str, fallback_name: str) -> str:
    headings = [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]
    for heading in headings[:20]:
        if _is_probable_title(heading):
            return _clean_text(heading)
    return Path(fallback_name).stem.replace("_", "/")


def _is_probable_title(heading: str) -> bool:
    lowered = heading.lower()
    blocked = {
        "a r t i c l e i n f o",
        "article",
        "abstract",
        "authors",
        "computers in biology and medicine",
        "hhs public access",
        "iscience",
        "nature methods",
        "nature communications",
        "open access",
        "open",
        "oPeN".lower(),
        "research",
        "report",
        "summary",
        "introduction",
        "results",
        "methods",
        "references",
        "article open",
        "highlights",
        "graphical abstract",
        "scientific data 110110 0111101 11011110 011101101",
        "software",
    }
    return (
        len(heading) > MIN_TITLE_HEADING_LENGTH
        and lowered not in blocked
        and not lowered.startswith("figure ")
    )


def _extract_year(text: str, doi: str) -> str:
    doi_match = re.search(r"[-./](20[2-9][0-9])[-./]", doi)
    if doi_match:
        return doi_match.group(1)

    nature_match = re.search(r"s\d+-0?(\d{2})-", doi)
    if nature_match:
        return f"20{nature_match.group(1)}"

    candidates: list[str] = re.findall(r"\b(20[2-9][0-9])\b", _text_before_references(text)[:4000])
    if candidates:
        return candidates[0]
    return "2020--2025"


def _extract_code_url(text: str) -> str:
    search_text = _code_availability_text(text)
    urls = re.findall(
        r"(?:https?://)?(?:www\.)?(?:github|gitlab)\.com/[A-Za-z0-9_.~/%#?=&:+@-]+",
        _normalize_broken_urls(search_text),
        re.I,
    )
    if not urls:
        return "Não localizado na conversão textual"
    url = _clean_text(urls[0].rstrip(".,;]"))
    if not url.startswith("http"):
        return f"https://{url}"
    return url


def _classify_domain(text: str) -> str:
    lowered = text.lower()
    domains = [
        ("pulmão", ("lung", "nsclc", "adenocarcinoma", "squamous")),
        ("mama", ("breast", "her2", "hormone receptor")),
        ("próstata", ("prostate", "gleason")),
        ("gastrointestinal", ("colorectal", "gastric", "barrett", "gastro")),
        ("ginecológico", ("ovarian", "endometrial")),
        ("pan-câncer", ("pan-cancer", "pan cancer", "multiple cancer", "multi-cancer")),
    ]
    for label, keywords in domains:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "oncologia digital"


def _classify_task(text: str) -> str:
    lowered = text.lower()
    if "foundation model" in lowered:
        return "modelo fundacional / representação"
    if "survival" in lowered or "risk stratification" in lowered:
        return "prognóstico / estratificação de risco"
    if "segmentation" in lowered or "segment" in lowered:
        return "segmentação"
    if "classification" in lowered or "diagnosis" in lowered:
        return "classificação diagnóstica"
    if "spatial transcript" in lowered or "visium" in lowered:
        return "integração histologia-ômica"
    return "pipeline de patologia computacional"


def _classify_evidence(text: str) -> dict[str, str]:
    lowered = text.lower()
    return {
        "artifact_qc_strategy": _artifact_qc(lowered),
        "normalization_policy": _normalization_policy(lowered),
        "partitioning_policy": _partitioning_policy(lowered),
        "storage_governance": _storage_governance(lowered),
        "ensemble_postprocessing": _ensemble_policy(lowered),
    }


def _artifact_qc(lowered: str) -> str:
    if "artifact" in lowered or "quality control" in lowered or "qc" in lowered:
        return "controle de qualidade ou artefatos explicitado"
    if "background" in lowered or "tissue segmentation" in lowered:
        return "segmentação de tecido/fundo"
    return "não central"


def _normalization_policy(lowered: str) -> str:
    if "macenko" in lowered or "vahadane" in lowered or "reinhard" in lowered:
        return "normalização de cor descrita"
    if "stain normalization" in lowered or "color normalization" in lowered:
        return "normalização de coloração mencionada"
    return "não especificada"


def _partitioning_policy(lowered: str) -> str:
    if "patient-level" in lowered or "patient level" in lowered:
        return "separação por paciente"
    if "cross-validation" in lowered or "train" in lowered and "test" in lowered:
        return "divisão treino/validação/teste descrita"
    return "não verificável no resumo extraído"


def _storage_governance(lowered: str) -> str:
    found = [term for term in GOVERNANCE_TERMS if term in lowered]
    if found:
        return ", ".join(sorted(set(found)))
    if "github.com" in lowered or "gitlab.com" in lowered:
        return "código público"
    return "governança não detalhada"


def _ensemble_policy(lowered: str) -> str:
    if "ensemble" in lowered or "uncertainty" in lowered:
        return "ensemble/incerteza discutido"
    if "post-processing" in lowered or "postprocessing" in lowered:
        return "pós-processamento descrito"
    return "não central"


def _rq_tags(evidence: dict[str, str]) -> str:
    tags = []
    if evidence["artifact_qc_strategy"] != "não central":
        tags.append("RQ1")
    if (
        evidence["normalization_policy"] != "não especificada"
        or "paciente" in evidence["partitioning_policy"]
    ):
        tags.append("RQ2")
    if evidence["storage_governance"] != "governança não detalhada":
        tags.append("RQ3")
    if evidence["ensemble_postprocessing"] != "não central":
        tags.append("RQ4")
    return ", ".join(tags) if tags else "RQ geral"


def _chapter_summary(text: str) -> str:
    lowered = text.lower()
    if "foundation model" in lowered:
        return "Mostra a consolidação de modelos fundacionais e avaliação em larga escala."
    if "quality control" in lowered or "artifact" in lowered:
        return "Evidencia que QC e artefatos precisam ser tratados como parte do fluxo."
    if "hdf5" in lowered or "zarr" in lowered or "nextflow" in lowered:
        return "Reforça a importância de contratos de dados e execução reprodutível."
    if "ensemble" in lowered or "uncertainty" in lowered:
        return "Relaciona agregação de modelos ou incerteza à segurança da inferência."
    if "segmentation" in lowered:
        return "Contribui com evidência sobre segmentação supervisionada em câncer."
    return "Contribui para o panorama recente de DL aplicado à patologia oncológica."


def _screening_table_row(count: ScreeningCount) -> str:
    origin = _display_origin(count.origin)
    return (
        f"{latex_escape(origin)} & {count.total} & {count.accepted} & "
        f"{count.rejected} & {count.raw_failed} \\\\"
    )


def _accepted_matrix_row(study: AcceptedStudy) -> str:
    study_cell = (
        f"{latex_escape(study.title)} "
        f"({latex_escape(_display_origin(study.source))}, {latex_escape(study.year)})"
    )
    return (
        f"{study_cell} & {latex_escape(study.artifact_qc_strategy)} & "
        f"{latex_escape(study.normalization_policy)} & "
        f"{latex_escape(study.partitioning_policy)} & "
        f"{latex_escape(study.storage_governance)} & "
        f"{latex_escape(study.ensemble_postprocessing)} & "
        f"{_code_availability_cell(study.code_url)} \\\\"
    )


def _bib_entry(study: AcceptedStudy) -> str:
    fields = [
        f"  title = {{{_bib_escape(study.title)}}}",
        f"  year = {{{_bib_escape(study.year)}}}",
        f"  keywords = {{{_bib_escape(study.rq_tags)}}}",
        f"  note = {{{_bib_escape(study.chapter_summary)}}}",
    ]
    if study.doi:
        fields.append(f"  doi = {{{_bib_escape(study.doi)}}}")
    if study.code_url.startswith("http"):
        fields.append(f"  url = {{{_bib_escape(study.code_url)}}}")
    return "@article{" + study.bib_key + ",\n" + ",\n".join(fields) + "\n}"


def _bib_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("&", r"\&")


def _clean_text(value: str) -> str:
    value = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", value)
    cleaned = re.sub(r"<[^>]+>", "", value)
    cleaned = cleaned.replace("&amp;", "&")
    cleaned = cleaned.replace("\\_", "_")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _code_availability_text(text: str) -> str:
    marker = re.search(r"^##\s+(?:Data and code|Code) availability\s*$", text, re.I | re.M)
    if marker is None:
        return _text_before_references(text)
    next_heading = re.search(r"^##\s+", text[marker.end() :], re.M)
    if next_heading is None:
        return text[marker.start() :]
    return text[marker.start() : marker.end() + next_heading.start()]


def _normalize_broken_urls(value: str) -> str:
    normalized = value.replace("\\_", "_")
    normalized = re.sub(r"https?://\s+", "https://", normalized, flags=re.I)
    normalized = re.sub(r"(github|gitlab)\s*\.\s*com", r"\1.com", normalized, flags=re.I)
    normalized = re.sub(r"(?<=github\.com/)\s+", "", normalized, flags=re.I)
    normalized = re.sub(r"(?<=gitlab\.com/)\s+", "", normalized, flags=re.I)
    normalized = re.sub(r"/\s+", "/", normalized)
    return normalized


def _text_before_references(text: str) -> str:
    marker = re.search(r"^##\s+References\s*$", text, re.I | re.M)
    return text if marker is None else text[: marker.start()]


def _display_origin(origin: str) -> str:
    labels = {"IEEE": "IEEE", "PUBMED": "PubMed", "SPRINGER": "Springer", "TOTAL": "Total"}
    return labels.get(origin, origin.title())


def _code_availability_cell(code_url: str) -> str:
    if code_url.startswith("http"):
        return rf"\url{{{_url_escape(code_url)}}}"
    return latex_escape(code_url)


def _url_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("#", r"\#")


def _protocol_block(lines: list[str], start_marker: str, end_marker: str) -> list[str]:
    start = _line_index(lines, start_marker) + 1
    end = _line_index(lines, end_marker)
    block = []
    for line in lines[start:end]:
        stripped = line.strip()
        if stripped:
            block.append(stripped)
    return block


def _protocol_queries(lines: list[str]) -> list[ProtocolQuery]:
    queries: list[ProtocolQuery] = []
    for index, line in enumerate(lines):
        database = PROTOCOL_QUERY_MARKERS.get(line.strip())
        if database is None:
            continue
        query = _next_non_empty_line(lines, index + 1)
        queries.append(ProtocolQuery(database=database, query=query))
    return queries


def _line_index(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return index
    raise ValueError(f"Protocol marker not found: {marker}")


def _next_non_empty_line(lines: list[str], start: int) -> str:
    for line in lines[start:]:
        if line.strip():
            return line.strip()
    raise ValueError("Expected a non-empty protocol line")
