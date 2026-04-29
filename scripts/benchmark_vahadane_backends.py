from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.training.master_manifest_queries import (  # noqa: E402
    CanonicalRowRecord,
    load_lr_finder_training_records,
)
from helpers.training.stain_normalization import build_split_stain_normalizer  # noqa: E402

ImageArray = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    runtime_normalization_method: str
    runtime_vahadane_backend: str = "fixed_source"


@dataclass(frozen=True)
class PatchSample:
    record: CanonicalRowRecord
    image: ImageArray


def _parse_bool(value: str) -> bool:
    candidate = value.strip().lower()
    if candidate in {"1", "true", "t", "yes", "y"}:
        return True
    if candidate in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _default_manifest_path() -> str | None:
    return os.environ.get("LR_FINDER_MASTER_MANIFEST_PATH") or os.environ.get(
        "TRAINING_MASTER_MANIFEST_PATH"
    )


def _sample_records(
    records: list[CanonicalRowRecord],
    *,
    limit: int,
    seed: int,
) -> list[CanonicalRowRecord]:
    if not records:
        raise ValueError("No LR-finder training records were found in master_manifest.sqlite.")
    rng = np.random.default_rng(seed)
    sample_size = min(limit, len(records))
    indices = rng.choice(len(records), size=sample_size, replace=False)
    return [records[int(index)] for index in sorted(indices.tolist())]


def _read_patch_samples(records: list[CanonicalRowRecord]) -> list[PatchSample]:
    handles: dict[Path, h5py.File] = {}
    try:
        samples: list[PatchSample] = []
        for record in records:
            handle = handles.get(record.source_hdf5_path)
            if handle is None:
                handle = h5py.File(record.source_hdf5_path, "r")
                handles[record.source_hdf5_path] = handle
            image = cast(ImageArray, np.asarray(handle["images"][record.source_row_index]))
            samples.append(PatchSample(record=record, image=image))
        return samples
    finally:
        for handle in handles.values():
            handle.close()


def _benchmark_case(
    benchmark_case: BenchmarkCase,
    *,
    master_manifest_path: Path,
    records: list[CanonicalRowRecord],
    samples: list[PatchSample],
    device: torch.device,
    warmup: int,
) -> dict[str, Any]:
    try:
        normalizer = build_split_stain_normalizer(
            master_manifest_path,
            records,
            runtime_normalization_method=benchmark_case.runtime_normalization_method,
            runtime_vahadane_backend=benchmark_case.runtime_vahadane_backend,
            device=device,
        )
    except Exception as error:
        return {
            "name": benchmark_case.name,
            "status": "skipped",
            "error": f"{type(error).__name__}: {error}",
        }
    if normalizer is None:
        return {
            "name": benchmark_case.name,
            "status": "skipped",
            "error": "runtime normalizer was not built",
        }

    for sample in samples[:warmup]:
        normalizer.normalize_image(sample.image, cache_key=sample.record.filename)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for sample in samples:
        normalizer.normalize_image(sample.image, cache_key=sample.record.filename)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    patches_per_second = len(samples) / elapsed_seconds if elapsed_seconds > 0 else float("inf")

    return {
        "name": benchmark_case.name,
        "status": "ok",
        "method": benchmark_case.runtime_normalization_method,
        "runtime_vahadane_backend": benchmark_case.runtime_vahadane_backend,
        "patches": len(samples),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_patch": elapsed_seconds / len(samples),
        "patches_per_second": patches_per_second,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact VAHADANE, fixed-source VAHADANE, and RUIFROK on sampled "
            "Phase 7 training patches."
        )
    )
    parser.add_argument(
        "--master-manifest-path",
        type=Path,
        default=_default_manifest_path(),
        help=(
            "Path to master_manifest.sqlite. Defaults to LR_FINDER_MASTER_MANIFEST_PATH, then "
            "TRAINING_MASTER_MANIFEST_PATH."
        ),
    )
    parser.add_argument("--limit", type=int, default=16, help="Maximum sampled patches to time.")
    parser.add_argument("--seed", type=int, default=24, help="Sampling seed.")
    parser.add_argument(
        "--smart-sampling",
        type=_parse_bool,
        default=True,
        help="Use Stage 6/7 selected training rows when true.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Normalizer device. Use cpu for DataLoader-worker-equivalent timings.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup patches before timing.")
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="Skip the old torch-staintools exact VAHADANE backend.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.master_manifest_path is None:
        raise SystemExit("Provide --master-manifest-path or set LR_FINDER_MASTER_MANIFEST_PATH.")
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1.")

    master_manifest_path = Path(args.master_manifest_path)
    records = load_lr_finder_training_records(
        master_manifest_path,
        smart_sampling=bool(args.smart_sampling),
    )
    sampled_records = _sample_records(records, limit=int(args.limit), seed=int(args.seed))
    samples = _read_patch_samples(sampled_records)
    device = torch.device(str(args.device))

    cases = [
        BenchmarkCase(
            name="vahadane_torch_staintools_exact",
            runtime_normalization_method="VAHADANE",
            runtime_vahadane_backend="torch_staintools_exact",
        ),
        BenchmarkCase(
            name="vahadane_fixed_source",
            runtime_normalization_method="VAHADANE",
            runtime_vahadane_backend="fixed_source",
        ),
        BenchmarkCase(
            name="ruifrok_baseline",
            runtime_normalization_method="RUIFROK",
            runtime_vahadane_backend="fixed_source",
        ),
    ]
    if args.skip_exact:
        cases = [case for case in cases if case.name != "vahadane_torch_staintools_exact"]

    results = [
        _benchmark_case(
            benchmark_case,
            master_manifest_path=master_manifest_path,
            records=sampled_records,
            samples=samples,
            device=device,
            warmup=max(0, int(args.warmup)),
        )
        for benchmark_case in cases
    ]
    ruifrok_seconds = next(
        (
            float(result["seconds_per_patch"])
            for result in results
            if result["name"] == "ruifrok_baseline" and result["status"] == "ok"
        ),
        None,
    )
    for result in results:
        if result["status"] == "ok" and ruifrok_seconds is not None:
            result["relative_to_ruifrok_seconds_per_patch"] = (
                float(result["seconds_per_patch"]) / ruifrok_seconds
            )

    payload = {
        "master_manifest_path": str(master_manifest_path),
        "smart_sampling": bool(args.smart_sampling),
        "sampled_patches": len(samples),
        "device": str(device),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
