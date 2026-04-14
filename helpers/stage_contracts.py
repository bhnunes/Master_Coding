from __future__ import annotations

from pathlib import Path

STAGE5_SINGLETON_SPLIT_FILES: dict[str, str] = {
    "TRAIN": "TRAIN.h5",
    "VALIDATION": "VALIDATION.h5",
    "TEST": "TEST.h5",
}
STAGE5_MANIFEST_FILE_NAME = "manifest.csv"
STAGE5_RUN_CONFIG_FILE_NAME = "run_config.json"
STAGE5_SPLIT_STATS_FILE_NAME = "split_stats.csv"


def stage5_singleton_split_paths(base_dir: Path) -> dict[str, Path]:
    """Return the canonical Stage 5 singleton HDF5 paths for a run directory."""

    return {
        split_name: base_dir / file_name
        for split_name, file_name in STAGE5_SINGLETON_SPLIT_FILES.items()
    }


def stage6_5_patient_shard_dir_name(split_name: str) -> str:
    """Return the canonical Stage 6.5 shard directory name for one split."""

    return f"{split_name}_shards"


def stage6_5_patient_shard_dir_paths(base_dir: Path) -> dict[str, Path]:
    """Return the canonical Stage 6.5 shard directories for a run directory."""

    return {
        split_name: base_dir / stage6_5_patient_shard_dir_name(split_name)
        for split_name in STAGE5_SINGLETON_SPLIT_FILES
    }
