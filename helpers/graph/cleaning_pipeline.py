from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from helpers.extraction.master_manifest import ManifestPatchRecord, MasterManifest
from helpers.graph.contamination import (
    GraphContaminationParameters,
    calculate_roi_contamination,
    calculate_roi_contamination_from_arrays,
    load_graph_source_batch,
)
from helpers.provenance import hash_file_sha256

ACCEPTED = "accepted"
REJECTED = "rejected"

Scorer = Callable[[Path | str, Path | str, GraphContaminationParameters], float | None]
ProgressFactory = Callable[[Iterable[Any]], Iterable[Any]]

_HDF5_SCORING_BATCH_SIZE = 512


@dataclass(frozen=True)
class GraphCleaningSummary:
    """Execution summary for Stage 3.3 graph cleaning."""

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
    source_hdf5_sha256: str | None = None
    source_row_index: int | None = None


@dataclass(frozen=True)
class CleaningDecisionRecord:
    filename: str
    decision: str
    contamination_rate: float | None
    patient_id: str | None = None
    slide_id: str | None = None
    source_hdf5_path: str | None = None
    source_hdf5_sha256: str | None = None
    source_row_index: int | None = None


@dataclass(frozen=True)
class GraphCleaningPipelineConfig:
    master_manifest_path: Path
    output_base_dir: Path
    graph_params: GraphContaminationParameters
    tau: float
    num_workers: int
    logger: logging.Logger
    scorer: Scorer = calculate_roi_contamination
    progress_factory: ProgressFactory | None = None


def run_graph_cleaning_pipeline(config: GraphCleaningPipelineConfig) -> GraphCleaningSummary:
    """Run Stage 3.3 filtering without mutating canonical Stage 2 shard contents."""

    started_at = time.time()
    config.logger.info("--- Starting Stage 3.3 manifest-driven filtering process ---")
    config.logger.info("Using optimal parameters: %s | tau=%.2f", config.graph_params, config.tau)
    config.logger.info("Distributing work across %s CPU cores.", config.num_workers)

    accepted_manifest_path = config.output_base_dir / "accepted_manifest.csv"
    rejected_manifest_path = config.output_base_dir / "rejected_manifest.csv"
    config.logger.info(
        "Accepted/rejected manifests will be written under: %s", config.output_base_dir
    )
    config.logger.info(
        "Canonical Stage 3.3 output is persisted to patch_stage_state in '%s'; "
        "Stage 2 HDF5 shards are treated as read-only pixel sources.",
        config.master_manifest_path,
    )

    candidates = _list_manifest_candidates(config.master_manifest_path, config.logger)

    if not candidates:
        return GraphCleaningSummary(
            total_images=0,
            accepted=0,
            rejected=0,
            skipped=0,
            output_base_dir=config.output_base_dir,
            accepted_manifest_path=accepted_manifest_path,
            rejected_manifest_path=rejected_manifest_path,
        )

    config.logger.info("Found %s source rows to process.", len(candidates))
    decisions = _process_hdf5_candidates(
        candidates=candidates,
        graph_params=config.graph_params,
        tau=config.tau,
        scorer=config.scorer,
        progress_factory=config.progress_factory,
    )
    MasterManifest(config.master_manifest_path).update_stage3_3_cleaning_decisions(
        decisions=[record.__dict__ for record in decisions]
    )
    _write_decision_manifest(accepted_manifest_path, decisions, ACCEPTED)
    _write_decision_manifest(rejected_manifest_path, decisions, REJECTED)
    result_counts = Counter(record.decision for record in decisions)
    skipped_total = 0

    config.logger.info("\n--- Filtering Complete ---")
    config.logger.info("Total source rows analyzed: %s", len(candidates))
    config.logger.info("Accepted rows: %s", result_counts[ACCEPTED])
    config.logger.info("Rejected rows: %s", result_counts[REJECTED])
    config.logger.info("Skipped rows: %s", skipped_total)
    config.logger.info("Total execution time: %.2f minutes.", (time.time() - started_at) / 60)
    config.logger.info(
        "A detailed log has been saved to: %s", _resolve_log_destination(config.logger)
    )
    return GraphCleaningSummary(
        total_images=len(candidates),
        accepted=result_counts[ACCEPTED],
        rejected=result_counts[REJECTED],
        skipped=skipped_total,
        output_base_dir=config.output_base_dir,
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
        "Canonical output: patch_stage_state in master_manifest.sqlite\n"
        "Stage 2 shard files were not rewritten\n"
        f"Accepted manifest: {summary.accepted_manifest_path}\n"
        f"Rejected manifest: {summary.rejected_manifest_path}"
    )


def _list_manifest_candidates(
    master_manifest_path: Path,
    logger: logging.Logger,
) -> list[SourceCandidateRecord]:
    try:
        records = MasterManifest(master_manifest_path).list_stage2_patch_records()
    except FileNotFoundError:
        logger.error("The Stage 2 master manifest '%s' does not exist.", master_manifest_path)
        return []

    candidates: list[SourceCandidateRecord] = []
    if not records:
        logger.warning(
            "No canonical Stage 2 rows found in master manifest: '%s'.",
            master_manifest_path,
        )
        return []

    for record in records:
        if record.label != 1:
            continue
        candidates.append(_candidate_from_manifest_record(record))
    return candidates


def _candidate_from_manifest_record(record: ManifestPatchRecord) -> SourceCandidateRecord:
    source_hdf5_sha256 = record.source_signature or hash_file_sha256(record.source_hdf5_path)
    return SourceCandidateRecord(
        filename=record.filename,
        image_path=record.source_image_path,
        mask_path=record.source_mask_path,
        patient_id=str(record.patient_id),
        slide_id=record.slide_id,
        source_hdf5_path=str(record.source_hdf5_path),
        source_hdf5_sha256=source_hdf5_sha256,
        source_row_index=record.source_row_index,
    )


def _process_hdf5_candidates(
    *,
    candidates: Sequence[SourceCandidateRecord],
    graph_params: GraphContaminationParameters,
    tau: float,
    scorer: Scorer,
    progress_factory: ProgressFactory | None,
) -> list[CleaningDecisionRecord]:
    if scorer is calculate_roi_contamination and _can_batch_hdf5_candidates(candidates):
        return _process_hdf5_candidates_batched(
            candidates=candidates,
            graph_params=graph_params,
            tau=tau,
            progress_factory=progress_factory,
        )

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
        decisions.append(_build_decision_record(candidate, contamination_rate, tau))
    return decisions


def _process_hdf5_candidates_batched(
    *,
    candidates: Sequence[SourceCandidateRecord],
    graph_params: GraphContaminationParameters,
    tau: float,
    progress_factory: ProgressFactory | None,
) -> list[CleaningDecisionRecord]:
    decisions: list[CleaningDecisionRecord] = []
    batches = list(_iter_hdf5_candidate_batches(candidates))
    iterable: Iterable[Sequence[SourceCandidateRecord]] = batches
    if progress_factory is not None:
        iterable = cast(Iterable[Sequence[SourceCandidateRecord]], progress_factory(batches))

    for batch_candidates in iterable:
        active_images, active_masks = _load_hdf5_candidate_batch(batch_candidates)
        for batch_offset, candidate in enumerate(batch_candidates):
            contamination_rate = calculate_roi_contamination_from_arrays(
                active_images[batch_offset],
                active_masks[batch_offset],
                graph_params,
                base_name=candidate.filename,
            )
            decisions.append(_build_decision_record(candidate, contamination_rate, tau))

    return decisions


def _can_batch_hdf5_candidates(candidates: Sequence[SourceCandidateRecord]) -> bool:
    if not candidates:
        return False

    current_path: str | None = None
    expected_row_index: int | None = None
    for candidate in candidates:
        if candidate.source_hdf5_path is None or candidate.source_row_index is None:
            return False
        if candidate.source_hdf5_path != current_path:
            current_path = candidate.source_hdf5_path
            expected_row_index = candidate.source_row_index
        if candidate.source_row_index != expected_row_index:
            return False
        assert expected_row_index is not None
        expected_row_index += 1
    return True


def _iter_hdf5_candidate_batches(
    candidates: Sequence[SourceCandidateRecord],
) -> list[Sequence[SourceCandidateRecord]]:
    batches: list[Sequence[SourceCandidateRecord]] = []
    batch_start = 0
    while batch_start < len(candidates):
        first_candidate = candidates[batch_start]
        assert first_candidate.source_hdf5_path is not None
        batch_end = min(batch_start + _HDF5_SCORING_BATCH_SIZE, len(candidates))
        while (
            batch_end < len(candidates)
            and candidates[batch_end].source_hdf5_path != first_candidate.source_hdf5_path
        ):
            batch_end -= 1
        if batch_end == batch_start:
            batch_end = batch_start + 1
        batches.append(candidates[batch_start:batch_end])
        batch_start = batch_end
    return batches


def _load_hdf5_candidate_batch(
    candidates: Sequence[SourceCandidateRecord],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    first_candidate = candidates[0]
    assert first_candidate.source_hdf5_path is not None
    assert first_candidate.source_row_index is not None
    last_candidate = candidates[-1]
    assert last_candidate.source_row_index is not None

    start_index = first_candidate.source_row_index
    end_index = last_candidate.source_row_index + 1
    return load_graph_source_batch(first_candidate.source_hdf5_path, start_index, end_index)


def _build_decision_record(
    candidate: SourceCandidateRecord,
    contamination_rate: float | None,
    tau: float,
) -> CleaningDecisionRecord:
    decision = (
        REJECTED
        if contamination_rate is not None
        and not np.isnan(contamination_rate)
        and contamination_rate > tau
        else ACCEPTED
    )
    return CleaningDecisionRecord(
        filename=candidate.filename,
        decision=decision,
        contamination_rate=float(contamination_rate)
        if contamination_rate is not None and not np.isnan(contamination_rate)
        else None,
        patient_id=candidate.patient_id,
        slide_id=candidate.slide_id,
        source_hdf5_path=candidate.source_hdf5_path,
        source_hdf5_sha256=candidate.source_hdf5_sha256,
        source_row_index=candidate.source_row_index,
    )


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
                "source_hdf5_sha256",
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
                    "source_hdf5_sha256": record.source_hdf5_sha256,
                    "source_row_index": record.source_row_index,
                }
            )


def _resolve_log_destination(logger: logging.Logger) -> str:
    for handler in logger.handlers:
        filename = getattr(handler, "baseFilename", None)
        if isinstance(filename, str):
            return filename
    return "log file"
