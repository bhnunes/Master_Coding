from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.ensemble_optimizer.metadata import load_and_select_models


def test_load_and_select_models_sorts_and_limits_candidates(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    entries = [
        (
            "a_meta.json",
            {
                "architecture": "FPN",
                "encoder": "senet154",
                "checkpoint_path": "a.ckpt",
                "best_val_auprc_pixel_score": 0.61,
            },
        ),
        (
            "b_meta.json",
            {
                "architecture": "SWIN",
                "encoder": "enc",
                "checkpoint_path": "b.ckpt",
                "best_val_auprc_pixel_score": 0.88,
            },
        ),
        (
            "c_meta.json",
            {
                "architecture": "MANET",
                "encoder": "enc",
                "checkpoint_path": "c.ckpt",
                "best_val_auprc_pixel_score": 0.73,
            },
        ),
    ]
    for filename, payload in entries:
        (metadata_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    selected = load_and_select_models(
        metadata_dir=metadata_dir,
        n_top_models=2,
        sort_metric="best_val_auprc_pixel_score",
    )

    assert [item.architecture for item in selected] == ["SWIN", "MANET"]
    assert selected[0].metadata_filename == "b_meta.json"


def test_load_and_select_models_rejects_invalid_sort_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid sort_metric"):
        load_and_select_models(tmp_path, n_top_models=2, sort_metric="dice")
