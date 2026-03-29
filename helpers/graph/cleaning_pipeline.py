from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from helpers.graph.contamination import GraphContaminationParameters, calculate_roi_contamination
from helpers.optimization_sampling.sampling import discover_hdf5_image_mask_pairs

ACCEPTED = "accepted"
REJECTED = "rejected"

Scorer = Callable[[Path | str, Path | str, GraphContaminationParameters], float | None]
type ProgressFactory = Callable[[Iterable[str]], Iterable[str]]


@dataclass(frozen=True)
class GraphCleaningSummary:
    """Execution summary for Stage 4.3 graph cleaning."""

    total_images: int
    accepted: int
    rejected: int
    skipped: int
    output_base_dir: Path
    accepted_manifest_path: Path | None = None
    rejected_manifest_path: Path | None = None


@dataclass(frozen=True)
class SourceCandidateRecord:
    filename: str
    image_path: Path | str
    mask_path: Path | str
    patient_id: str | None = None
    slide_id: str | None = None
    source_hdf5_path: str | None = None
    source_row_index: int | None = None


@dataclass(frozen=True)
class CleaningDecisionRecord:
    filename: str
    decision: str
    contamination_rate: float | None
    patient_id: str | None = None
    slide_id: str | None = None
    source_hdf5_path: str | None = None
    source_row_index: int | None = None


def run_graph_cleaning_pipeline(
    *,
    source_hdf5_path: Path,
    output_base_dir: Path,
    graph_params: GraphContaminationParameters,
    tau: float,
    num_workers: int,
    logger: logging.Logger,
    scorer: Scorer = calculate_roi_contamination,
    progress_factory: ProgressFactory | None = None,
) -> GraphCleaningSummary:
    """Run Stage 4.3 filtering and write accepted/rejected manifests."""

    started_at = time.time()
    logger.info("--- Starting Stage 4.3 HDF5-backed filtering process ---")
    logger.info("Using optimal parameters: %s | tau=%.2f", graph_params, tau)
    logger.info("Distributing work across %s CPU cores.", num_workers)

    accepted_manifest_path = output_base_dir / "accepted_manifest.csv"
    rejected_manifest_path = output_base_dir / "rejected_manifest.csv"
    logger.info("Accepted/rejected manifests will be written under: %s", output_base_dir)

    candidates = _list_hdf5_candidates(source_hdf5_path, logger)

    if not candidates:
        return GraphCleaningSummary(
            total_images=0,
            accepted=0,
            rejected=0,
            skipped=0,
            output_base_dir=output_base_dir,
            accepted_manifest_path=accepted_manifest_path,
            rejected_manifest_path=rejected_manifest_path,
        )

    logger.info("Found %s source rows to process.", len(candidates))
    decisions = _process_hdf5_candidates(
        candidates=candidates,
        graph_params=graph_params,
        tau=tau,
        scorer=scorer,
        progress_factory=progress_factory,
    )
    _write_decision_manifest(accepted_manifest_path, decisions, ACCEPTED)
    _write_decision_manifest(rejected_manifest_path, decisions, REJECTED)
    result_counts = Counter(record.decision for record in decisions)
    skipped_total = 0

    logger.info("\n--- Filtering Complete ---")
    logger.info("Total source rows analyzed: %s", len(candidates))
    logger.info("Accepted rows: %s", result_counts[ACCEPTED])
    logger.info("Rejected rows: %s", result_counts[REJECTED])
    logger.info("Skipped rows: %s", skipped_total)
    logger.info("Total execution time: %.2f minutes.", (time.time() - started_at) / 60)
    logger.info("A detailed log has been saved to: %s", _resolve_log_destination(logger))
    return GraphCleaningSummary(
        total_images=len(candidates),
        accepted=result_counts[ACCEPTED],
        rejected=result_counts[REJECTED],
        skipped=skipped_total,
        output_base_dir=output_base_dir,
        accepted_manifest_path=accepted_manifest_path,
        rejected_manifest_path=rejected_manifest_path,
    )


def build_cleaning_message(summary: GraphCleaningSummary) -> str:
    """Build the final operator-facing result message."""

    return (
        "\n\n--- Cleaning Complete ---\n"
        f"Total source rows analyzed: {summary.total_images}\n"
        f"Accepted: {summary.accepted}\n"
        f"Rejected: {summary.rejected}\n"
        f"Skipped: {summary.skipped}\n"
        f"Accepted manifest: {summary.accepted_manifest_path}\n"
        f"Rejected manifest: {summary.rejected_manifest_path}"
    )


def _list_hdf5_candidates(
    source_hdf5_path: Path,
    logger: logging.Logger,
) -> list[SourceCandidateRecord]:
    if not source_hdf5_path.is_file():
        logger.error("The source HDF5 dataset '%s' does not exist.", source_hdf5_path)
        return []
    with h5py.File(source_hdf5_path, "r") as handle:
        filenames = handle["filenames"]
        if len(filenames) == 0:
            logger.error("No rows found in the source HDF5 dataset: '%s'.", source_hdf5_path)
            return []
    pairs = discover_hdf5_image_mask_pairs(source_hdf5_path)
    candidates: list[SourceCandidateRecord] = []
    with h5py.File(source_hdf5_path, "r") as handle:
        patient_ids = handle["patient_ids"]
        filenames = handle["filenames"]
        slide_ids = handle.get("slide_ids")
        for index in range(len(filenames)):
            filename_value = filenames[index]
            filename = (
                filename_value.decode("utf-8")
                if isinstance(filename_value, bytes)
                else str(filename_value)
            )
            stem = Path(filename).stem
            pair = pairs[stem]
            slide_value = slide_ids[index] if slide_ids is not None else None
            candidates.append(
                SourceCandidateRecord(
                    filename=filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    patient_id=str(int(patient_ids[index])),
                    slide_id=(
                        slide_value.decode("utf-8")
                        if isinstance(slide_value, bytes)
                        else str(slide_value)
                    )
                    if slide_value is not None
                    else None,
                    source_hdf5_path=str(source_hdf5_path),
                    source_row_index=index,
                )
            )
    return candidates


def _process_hdf5_candidates(
    *,
    candidates: Sequence[SourceCandidateRecord],
    graph_params: GraphContaminationParameters,
    tau: float,
    scorer: Scorer,
    progress_factory: ProgressFactory | None,
) -> list[CleaningDecisionRecord]:
    iterable: Iterable[SourceCandidateRecord] = candidates
    if progress_factory is not None:
        iterable = cast(
            Iterable[SourceCandidateRecord], progress_factory([c.filename for c in candidates])
        )
        filename_to_candidate = {candidate.filename: candidate for candidate in candidates}
        iterable = (filename_to_candidate[filename] for filename in cast(Iterable[str], iterable))
    decisions: list[CleaningDecisionRecord] = []
    for candidate in iterable:
        contamination_rate = scorer(candidate.image_path, candidate.mask_path, graph_params)
        decision = (
            REJECTED
            if contamination_rate is not None
            and not np.isnan(contamination_rate)
            and contamination_rate > tau
            else ACCEPTED
        )
        decisions.append(
            CleaningDecisionRecord(
                filename=candidate.filename,
                decision=decision,
                contamination_rate=float(contamination_rate)
                if contamination_rate is not None and not np.isnan(contamination_rate)
                else None,
                patient_id=candidate.patient_id,
                slide_id=candidate.slide_id,
                source_hdf5_path=candidate.source_hdf5_path,
                source_row_index=candidate.source_row_index,
            )
        )
    return decisions


def _write_decision_manifest(
    output_path: Path,
    decisions: Sequence[CleaningDecisionRecord],
    decision_name: str,
) -> None:
    rows = [record for record in decisions if record.decision == decision_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "decision",
                "contamination_rate",
                "patient_id",
                "slide_id",
                "source_hdf5_path",
                "source_row_index",
            ],
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "filename": record.filename,
                    "decision": record.decision,
                    "contamination_rate": record.contamination_rate,
                    "patient_id": record.patient_id,
                    "slide_id": record.slide_id,
                    "source_hdf5_path": record.source_hdf5_path,
                    "source_row_index": record.source_row_index,
                }
            )


def _resolve_log_destination(logger: logging.Logger) -> str:
    for handler in logger.handlers:
        filename = getattr(handler, "baseFilename", None)
        if isinstance(filename, str):
            return filename
    return "log file"
