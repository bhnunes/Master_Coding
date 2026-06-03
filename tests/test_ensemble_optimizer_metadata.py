from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.ensemble_optimizer.metadata import (
    load_model_candidates,
    select_unique_candidates_by_architecture,
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


def test_select_unique_candidates_by_architecture_selects_one_per_requested_in_order(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", architecture="FPN", checkpoint_path="a.ckpt")
    _write_candidate(metadata_dir / "b_meta.json", architecture="MANET", checkpoint_path="b.ckpt")
    _write_candidate(metadata_dir / "c_meta.json", architecture="SWIN", checkpoint_path="c.ckpt")
    _write_candidate(metadata_dir / "d_meta.json", architecture="DPT", checkpoint_path="d.ckpt")

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")
    selected = select_unique_candidates_by_architecture(
        candidates,
        requested_architectures=("SWIN", "DPT", "FPN", "MANET"),
    )

    assert [item.architecture for item in selected] == ["SWIN", "DPT", "FPN", "MANET"]
    assert selected[0].metadata_filename == "c_meta.json"


def test_select_unique_candidates_by_architecture_rejects_duplicate_requested_metadata(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", architecture="FPN", checkpoint_path="a.ckpt")
    _write_candidate(metadata_dir / "b_meta.json", architecture="FPN", checkpoint_path="b.ckpt")
    _write_candidate(metadata_dir / "c_meta.json", architecture="SWIN", checkpoint_path="c.ckpt")

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")

    with pytest.raises(ValueError, match="duplicates=.*FPN"):
        select_unique_candidates_by_architecture(
            candidates,
            requested_architectures=("SWIN", "FPN"),
        )


def test_select_unique_candidates_by_architecture_rejects_missing_requested_metadata(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", architecture="SWIN", checkpoint_path="a.ckpt")

    candidates = load_model_candidates(metadata_dir, "best_val_auprc_pixel_score")

    with pytest.raises(ValueError, match="missing=.*FPN"):
        select_unique_candidates_by_architecture(
            candidates,
            requested_architectures=("SWIN", "FPN"),
        )


def test_select_unique_candidates_by_architecture_ignores_unrequested_duplicates(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_candidate(metadata_dir / "a_meta.json", architecture="SWIN", checkpoint_path="a.ckpt")
    _write_candidate(metadata_dir / "b_meta.json", architecture="FPN", checkpoint_path="b.ckpt")
    _write_candidate(
        metadata_dir / "c_meta.json",
        architecture="MANET",
        checkpoint_path="c.ckpt",
        compatibility_signature="compat-b",
    )
    _write_candidate(
        metadata_dir / "d_meta.json",
        architecture="MANET",
        checkpoint_path="d.ckpt",
        compatibility_signature="compat-b",
    )

    candidates = load_model_candidates(
        metadata_dir,
        "best_val_auprc_pixel_score",
        requested_architectures=("SWIN", "FPN"),
    )
    selected = select_unique_candidates_by_architecture(
        candidates,
        requested_architectures=("SWIN", "FPN"),
    )

    assert [item.metadata_filename for item in selected] == ["a_meta.json", "b_meta.json"]


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
