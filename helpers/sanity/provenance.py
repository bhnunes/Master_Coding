from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


def load_manifest(base_dir: Path) -> pd.DataFrame:
    manifest_path = base_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv not found in base_dir: {manifest_path}")
    return pd.read_csv(manifest_path)


def load_run_config(base_dir: Path) -> dict[str, Any] | None:
    run_config_path = base_dir / "run_config.json"
    if not run_config_path.is_file():
        return None
    return cast(dict[str, Any], json.loads(run_config_path.read_text(encoding="utf-8")))


def load_split_stats(base_dir: Path) -> pd.DataFrame | None:
    split_stats_path = base_dir / "split_stats.csv"
    if not split_stats_path.is_file():
        return None
    try:
        split_stats_df = pd.read_csv(split_stats_path)
    except pd.errors.EmptyDataError:
        return None
    if split_stats_df.empty or len(split_stats_df.columns) == 0:
        return None
    return split_stats_df
