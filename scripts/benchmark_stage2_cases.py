from __future__ import annotations

# ruff: noqa: E402
import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from helpers.extraction.config import load_database_manager_config
from helpers.extraction.image_reader_service import (
    SlideProcessingRequest,
    load_slide_runtime_settings,
    run_slide_processing,
)
from helpers.extraction.repository import ExtractionRepository

DEFAULT_CASE_IDS = (8, 4, 6, 7, 2, 5, 1)


@dataclass(frozen=True)
class SlideBenchmarkResult:
    case_id: int
    patient: str
    image_path: str
    annotation_path: str
    status: str
    elapsed_seconds: float
    extraction_seconds: float | None
    post_extraction_seconds: float | None
    cancer_patches_created: int
    not_cancer_patches_created: int
    artifact_patch_records: int
    profile_output_path: str


def _load_stage2_module() -> object:
    module_path = WORKSPACE_ROOT / "2_database_manager.py"
    spec = importlib.util.spec_from_file_location("stage2_database_manager", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Stage 2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_case_ids(raw_value: str | None) -> tuple[int, ...]:
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CASE_IDS
    return tuple(int(token.strip()) for token in raw_value.split(",") if token.strip())


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage 2 patch extraction for explicit database case IDs.",
    )
    parser.add_argument(
        "--case-ids",
        default=",".join(str(case_id) for case_id in DEFAULT_CASE_IDS),
        help="Comma-separated Stage 2 database IDs to benchmark in order.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/stage2_benchmarks",
        help="Directory where per-slide profiles and benchmark summaries are written.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Environment file to load before benchmarking.",
    )
    parser.add_argument(
        "--skip-hdf5-write",
        action="store_true",
        help="Benchmark extraction without writing the Stage 2 HDF5 shard.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    load_dotenv(args.env_file, override=True)
    config = load_database_manager_config(os.environ)
    runtime_settings = load_slide_runtime_settings(os.environ)
    repository = ExtractionRepository(database_path=config.database_path, tag=config.tag)
    stage2_module = cast(Any, _load_stage2_module())

    case_ids = _parse_case_ids(args.case_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    geojson_lookup = (
        stage2_module.GeoJsonLookup.from_directory(config.geojson_path)
        if config.geojson_path is not None
        and (config.use_advanced_artifact_filtering or config.activate_sanity_check_geojson)
        else None
    )

    cases = repository.list_cases_by_ids(case_ids)
    found_ids = {case.record_id for case in cases}
    missing_ids = [case_id for case_id in case_ids if case_id not in found_ids]
    if missing_ids:
        raise ValueError(f"Requested Stage 2 case IDs were not found: {missing_ids}")

    results: list[SlideBenchmarkResult] = []
    for case in cases:
        if case.annotation_path is None:
            raise ValueError(f"Case {case.record_id} is missing an annotation path.")
        artifacts_geojson_path = stage2_module.resolve_artifacts_geojson(
            case,
            config,
            geojson_lookup=geojson_lookup,
        )
        profile_output_path = profile_dir / f"case_{case.record_id}_{case.image_path.stem}.json"
        request = cast(
            SlideProcessingRequest,
            stage2_module.build_slide_request(
                case,
                config,
                runtime_settings,
                artifacts_geojson_path=artifacts_geojson_path,
            ),
        )
        request = replace(
            request,
            profile_output_path=profile_output_path,
            hdf5_output_path=None if args.skip_hdf5_write else request.hdf5_output_path,
        )

        started_at = perf_counter()
        result = run_slide_processing(request)
        elapsed_seconds = perf_counter() - started_at
        extraction_seconds: float | None = None
        if profile_output_path.exists():
            profile_payload = json.loads(profile_output_path.read_text(encoding="utf-8"))
            extraction_seconds = float(profile_payload["total_runtime_seconds"])
        benchmark_result = SlideBenchmarkResult(
            case_id=case.record_id,
            patient=case.patient,
            image_path=str(case.image_path),
            annotation_path=str(case.annotation_path),
            status=result.status,
            elapsed_seconds=elapsed_seconds,
            extraction_seconds=extraction_seconds,
            post_extraction_seconds=(
                max(0.0, elapsed_seconds - extraction_seconds)
                if extraction_seconds is not None
                else None
            ),
            cancer_patches_created=result.cancer_patches_created,
            not_cancer_patches_created=result.not_cancer_patches_created,
            artifact_patch_records=len(result.artifact_patch_records),
            profile_output_path=str(profile_output_path),
        )
        results.append(benchmark_result)
        print(
            json.dumps(
                {
                    "case_id": benchmark_result.case_id,
                    "elapsed_seconds": round(benchmark_result.elapsed_seconds, 3),
                    "extraction_seconds": (
                        round(benchmark_result.extraction_seconds, 3)
                        if benchmark_result.extraction_seconds is not None
                        else None
                    ),
                    "post_extraction_seconds": (
                        round(benchmark_result.post_extraction_seconds, 3)
                        if benchmark_result.post_extraction_seconds is not None
                        else None
                    ),
                    "status": benchmark_result.status,
                    "cancer_patches_created": benchmark_result.cancer_patches_created,
                    "not_cancer_patches_created": benchmark_result.not_cancer_patches_created,
                    "artifact_patch_records": benchmark_result.artifact_patch_records,
                    "profile_output_path": benchmark_result.profile_output_path,
                },
                sort_keys=True,
            )
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "case_ids": list(case_ids),
                "results": [asdict(result) for result in results],
                "total_elapsed_seconds": round(
                    sum(result.elapsed_seconds for result in results), 6
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
