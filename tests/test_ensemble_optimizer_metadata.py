from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from helpers.ensemble_optimizer.metadata import (
    load_model_candidates,
    select_best_candidates_by_architecture,
)


def _write_candidate(
    path: Path,
    *,
    architecture: str = "FPN",
    encoder: str = "enc",
    checkpoint_path: str = "model.ckpt",
    score: float = 0.8,
    compatibility_signature: str = "compat-a",
) -> None:
    path.write_text(
        json.dumps(
            {
                "architecture": architecture,
                "encoder": encoder,
                "checkpoint_path": checkpoint_path,
                "best_val_auprc_pixel_score": score,
                "compatibility_signature": compatibility_signature,
                "provenance": {
                    "dataset": {"sha256": "dataset-sha", "path": "TRAIN.h5"},
                    "validation_dataset": {"sha256": "validation-sha", "path": "VALIDATION.h5"},
                    "split_lineage": {"dataset_sha256": "dataset-sha", "source_signature": None},
                    "packaging_lineage": {
                        "dataset_sha256": "dataset-sha",
                        "source_signature": None,
                    },
                    "normalization_lineage": {"dataset_sha256": "dataset-sha"},
                    "smart_sampling_lineage": {
                        "enabled": False,
                        "selection_signature": None,
                    },
                    "artifact_aware_loss": {
                        "enabled": False,
                        "master_manifest_path": None,
                        "master_manifest_sha256": None,
                    },
                    "validation_lineage": {
                        "dataset_sha256": "validation-sha",
                        "source_signature": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_select_best_candidates_by_architecture_picks_best_requested_models(tmp_path: Path) -> None:
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
        _write_candidate(
            metadata_dir / filename,
            architecture=str(payload["architecture"]),
            encoder=str(payload["encoder"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            score=cast(float, payload["best_val_auprc_pixel_score"]),
        )

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
    selected, skipped = select_best_candidates_by_architecture(
        candidates,
        requested_architectures=("SWIN", "MANET"),
    )

    assert [item.architecture for item in selected] == ["SWIN", "MANET"]
    assert selected[0].metadata_filename == "b_meta.json"
    assert skipped == {}


def test_select_best_candidates_by_architecture_can_use_override_scores(tmp_path: Path) -> None:
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
        _write_candidate(
            metadata_dir / filename,
            architecture=str(payload["architecture"]),
            encoder=str(payload["encoder"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            score=cast(float, payload["best_val_auprc_pixel_score"]),
        )

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
    ranked, skipped = select_best_candidates_by_architecture(
        candidates,
        requested_architectures=("FPN",),
        score_getter=lambda item: {"a_meta.json": 0.95, "b_meta.json": 0.1}[item.metadata_filename],
    )

    assert [item.metadata_filename for item in ranked] == ["a_meta.json"]
    assert skipped == {}


def test_select_best_candidates_by_architecture_skips_missing_requested_models(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", architecture="SWIN", checkpoint_path="a.ckpt")

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
    selected, skipped = select_best_candidates_by_architecture(
        candidates,
        requested_architectures=("SWIN", "FPN"),
    )

    assert [item.architecture for item in selected] == ["SWIN"]
    assert skipped == {"FPN": "no valid candidate metadata found"}


def test_load_model_candidates_rejects_missing_provenance_fields(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "bad_meta.json").write_text(
        json.dumps(
            {
                "architecture": "FPN",
                "encoder": "enc",
                "checkpoint_path": "a.ckpt",
                "best_val_auprc_pixel_score": 0.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fail-closed provenance"):
        load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")


def test_load_model_candidates_rejects_missing_validation_lineage(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    payload = {
        "architecture": "FPN",
        "encoder": "enc",
        "checkpoint_path": "a.ckpt",
        "best_val_auprc_pixel_score": 0.5,
        "compatibility_signature": "compat-a",
        "provenance": {
            "dataset": {"sha256": "dataset-sha", "path": "TRAIN.h5"},
            "validation_dataset": {"sha256": "validation-sha", "path": "VALIDATION.h5"},
            "split_lineage": {"dataset_sha256": "dataset-sha", "source_signature": None},
            "packaging_lineage": {"dataset_sha256": "dataset-sha", "source_signature": None},
            "normalization_lineage": {"dataset_sha256": "dataset-sha"},
            "smart_sampling_lineage": {"enabled": False, "selection_signature": None},
            "artifact_aware_loss": {
                "enabled": False,
                "master_manifest_path": None,
                "master_manifest_sha256": None,
            },
        },
    }
    (metadata_dir / "bad_meta.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_lineage"):
        load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")


def test_load_model_candidates_rejects_incompatible_candidate_groups(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", compatibility_signature="compat-a")
    _write_candidate(
        metadata_dir / "b_meta.json",
        architecture="SWIN",
        checkpoint_path="b.ckpt",
        compatibility_signature="compat-b",
    )

    with pytest.raises(ValueError, match="incompatible provenance groups"):
        load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
