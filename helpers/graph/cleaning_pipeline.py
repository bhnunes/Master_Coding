from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt

from helpers.extraction.master_manifest import MasterManifest
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
    source_hdf5_path: Path
    output_base_dir: Path
    graph_params: GraphContaminationParameters
    tau: float
    num_workers: int
    logger: logging.Logger
    master_manifest_path: Path | None = None
    scorer: Scorer = calculate_roi_contamination
    progress_factory: ProgressFactory | None = None


def run_graph_cleaning_pipeline(config: GraphCleaningPipelineConfig) -> GraphCleaningSummary:
    """Run Stage 4.3 filtering and write accepted/rejected manifests."""

    started_at = time.time()
    config.logger.info("--- Starting Stage 4.3 HDF5-backed filtering process ---")
    config.logger.info("Using optimal parameters: %s | tau=%.2f", config.graph_params, config.tau)
    config.logger.info("Distributing work across %s CPU cores.", config.num_workers)

    accepted_manifest_path = config.output_base_dir / "accepted_manifest.csv"
    rejected_manifest_path = config.output_base_dir / "rejected_manifest.csv"
    config.logger.info(
        "Accepted/rejected manifests will be written under: %s", config.output_base_dir
    )

    candidates = _list_hdf5_candidates(config.source_hdf5_path, config.logger)

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
    if config.master_manifest_path is not None:
        MasterManifest(config.master_manifest_path).update_stage4_cleaning_decisions(
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
        filenames = cast(Any, handle["filenames"])
        if len(filenames) == 0:
            logger.error("No rows found in the source HDF5 dataset: '%s'.", source_hdf5_path)
            return []

        source_signature = handle.attrs.get("source_signature")
        source_hdf5_sha256 = (
            source_signature.decode("utf-8")
            if isinstance(source_signature, bytes)
            else str(source_signature)
            if source_signature is not None
            else hash_file_sha256(source_hdf5_path)
        )
        patient_ids = cast(Any, handle["patient_ids"])
        filename_values = filenames[:]
        patient_id_values = patient_ids[:]
        slide_values = cast(Any, handle["slide_ids"])[:] if "slide_ids" in handle else None

    candidates: list[SourceCandidateRecord] = []
    source_hdf5_path_text = str(source_hdf5_path)
    for index, filename_value in enumerate(filename_values):
        filename = (
            filename_value.decode("utf-8")
            if isinstance(filename_value, bytes)
            else str(filename_value)
        )
        slide_value = slide_values[index] if slide_values is not None else None
        candidates.append(
            SourceCandidateRecord(
                filename=filename,
                image_path=f"{source_hdf5_path_text}::images[{index}]",
                mask_path=f"{source_hdf5_path_text}::masks[{index}]",
                patient_id=str(int(patient_id_values[index])),
                slide_id=(
                    slide_value.decode("utf-8")
                    if isinstance(slide_value, bytes)
                    else str(slide_value)
                )
                if slide_value is not None
                else None,
                source_hdf5_path=source_hdf5_path_text,
                source_hdf5_sha256=source_hdf5_sha256,
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
    iterable: Iterable[int] = range(len(candidates))
    if progress_factory is not None:
        iterable = cast(Iterable[int], progress_factory(list(iterable)))

    decisions: list[CleaningDecisionRecord] = []
    active_batch_start = -1
    active_batch_candidates: Sequence[SourceCandidateRecord] = ()
    active_images: npt.NDArray[np.uint8] | None = None
    active_masks: npt.NDArray[np.uint8] | None = None

    for candidate_index in iterable:
        batch_start = (candidate_index // _HDF5_SCORING_BATCH_SIZE) * _HDF5_SCORING_BATCH_SIZE
        if batch_start != active_batch_start:
            active_batch_start = batch_start
            active_batch_candidates = candidates[
                batch_start : batch_start + _HDF5_SCORING_BATCH_SIZE
            ]
            active_images, active_masks = _load_hdf5_candidate_batch(active_batch_candidates)

        assert active_images is not None
        assert active_masks is not None

        batch_offset = candidate_index - active_batch_start
        candidate = candidates[candidate_index]
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

    source_hdf5_path = candidates[0].source_hdf5_path
    first_row_index = candidates[0].source_row_index
    if source_hdf5_path is None or first_row_index is None:
        return False

    expected_row_index = first_row_index
    for candidate in candidates:
        if candidate.source_hdf5_path != source_hdf5_path:
            return False
        if candidate.source_row_index != expected_row_index:
            return False
        expected_row_index += 1
    return True


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
