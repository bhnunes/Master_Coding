from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.ensemble_optimizer.metadata import (
    load_and_select_models,
    load_model_candidates,
    select_top_models,
)


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


def test_select_top_models_can_use_override_scores(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for filename, payload in (
        (
            "a_meta.json",
            {
                "architecture": "FPN",
                "encoder": "enc-a",
                "checkpoint_path": "a.ckpt",
                "best_val_auprc_pixel_score": 0.2,
            },
        ),
        (
            "b_meta.json",
            {
                "architecture": "SWIN",
                "encoder": "enc-b",
                "checkpoint_path": "b.ckpt",
                "best_val_auprc_pixel_score": 0.9,
            },
        ),
    ):
        (metadata_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
    ranked = select_top_models(
        candidates,
        n_top_models=1,
        score_getter=lambda item: {"a_meta.json": 0.95, "b_meta.json": 0.1}[item.metadata_filename],
    )

    assert [item.metadata_filename for item in ranked] == ["a_meta.json"]
