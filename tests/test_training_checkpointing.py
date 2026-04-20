import json
from pathlib import Path

import h5py
import pytest
import torch

from helpers.training.checkpointing import (
    EarlyStopping,
    build_training_compatibility_signature,
    get_previous_metrics,
    load_checkpoint_for_resume,
    save_metadata,
)


def test_early_stopping_saves_first_best_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(checkpoint_path),
    )

    improved = early_stopping(
        0.8,
        model,
        optimizer,
        epoch=1,
        val_loss=0.2,
        val_auprc=0.8,
        val_mcc_star=0.7,
        val_auroc=0.9,
    )

    assert improved is True
    assert checkpoint_path.exists()
    assert early_stopping.best_score == 0.8
    assert early_stopping._current_best_checkpoint_on_disk_path == str(checkpoint_path)


def test_early_stopping_triggers_after_patience_without_improvement(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(checkpoint_path),
    )

    early_stopping(0.8, model, optimizer, 1, 0.2, 0.8, 0.7, 0.9)
    improved = early_stopping(0.8, model, optimizer, 2, 0.2, 0.8, 0.7, 0.9)
    early_stopping(0.8, model, optimizer, 3, 0.2, 0.8, 0.7, 0.9)

    assert improved is False
    assert early_stopping.early_stop is True
    assert early_stopping.counter == 2


def test_load_checkpoint_for_resume_restores_model_optimizer_and_score(tmp_path: Path) -> None:
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": saved_model.state_dict(),
            "optimizer_state_dict": saved_optimizer.state_dict(),
            "best_val_score": 0.91,
        },
        checkpoint_path,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(tmp_path / "best_model.pth"),
    )

    start_epoch = load_checkpoint_for_resume(
        model=model,
        optimizer=optimizer,
        early_stopping=early_stopping,
        checkpoint_path=str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert start_epoch == 4
    assert early_stopping.best_score == 0.91
    assert early_stopping.counter == 0
    assert early_stopping._current_best_checkpoint_on_disk_path == str(checkpoint_path)
    for expected, restored in zip(saved_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(expected, restored)


def test_load_checkpoint_for_resume_rejects_incompatible_provenance(tmp_path: Path) -> None:
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": saved_model.state_dict(),
            "optimizer_state_dict": saved_optimizer.state_dict(),
            "best_val_score": 0.91,
        },
        checkpoint_path,
    )
    meta_path = tmp_path / "resume_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "compatibility_signature": "old-lineage",
                "provenance": {
                    "dataset": {},
                    "validation_dataset": {},
                    "split_lineage": {},
                    "packaging_lineage": {},
                    "normalization_lineage": {},
                    "smart_sampling_lineage": {},
                    "artifact_aware_loss": {},
                    "validation_lineage": {},
                },
            }
        ),
        encoding="utf-8",
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(
        patience=2,
        verbose=False,
        output_best_model_path=str(tmp_path / "best_model.pth"),
    )

    with pytest.raises(ValueError, match="incompatible provenance"):
        load_checkpoint_for_resume(
            model=model,
            optimizer=optimizer,
            early_stopping=early_stopping,
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
            expected_compatibility_signature="current-lineage",
        )


def test_get_previous_metrics_returns_checkpoint_metrics() -> None:
    checkpoint = {
        "val_auprc": 0.8,
        "val_mcc_star": 0.7,
        "val_auroc": 0.9,
        "val_loss": 0.2,
    }

    metrics = get_previous_metrics(checkpoint, None, None, None, None)

    assert metrics == (0.8, 0.7, 0.9, 0.2)


def test_early_stopping_can_clear_missing_initial_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(checkpoint_path))

    early_stopping.set_initial_best_checkpoint_path(str(tmp_path / "missing.pth"))

    assert early_stopping._current_best_checkpoint_on_disk_path is None


def test_early_stopping_saves_original_module_state_dict(tmp_path: Path) -> None:
    compiled_inner = torch.nn.Linear(2, 2)
    wrapper = type("CompiledModel", (torch.nn.Module,), {})()
    wrapper._orig_mod = compiled_inner
    optimizer = torch.optim.SGD(compiled_inner.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best_model.pth"
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(checkpoint_path))

    early_stopping.save_checkpoint(0.2, wrapper, optimizer, 1, 0.8, 0.7, 0.8, 0.9)
    saved = torch.load(checkpoint_path, map_location="cpu")

    assert saved["is_compiled"] is True
    assert saved["model_state_dict"].keys() == compiled_inner.state_dict().keys()


def test_load_checkpoint_for_resume_handles_empty_or_missing_paths(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    assert (
        load_checkpoint_for_resume(model, optimizer, early_stopping, "", torch.device("cpu")) == 0
    )
    assert (
        load_checkpoint_for_resume(
            model,
            optimizer,
            early_stopping,
            str(tmp_path / "missing.pth"),
            torch.device("cpu"),
        )
        == 0
    )


def test_load_checkpoint_for_resume_handles_missing_model_state_dict(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.pth"
    torch.save({"epoch": 2}, checkpoint_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    start_epoch = load_checkpoint_for_resume(
        model, optimizer, early_stopping, str(checkpoint_path), torch.device("cpu")
    )

    assert start_epoch == 0
    assert early_stopping._current_best_checkpoint_on_disk_path is None


def test_load_checkpoint_for_resume_ignores_optimizer_restore_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"bad": "state"},
            "best_val_score": 0.5,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        optimizer, "load_state_dict", lambda state: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    early_stopping = EarlyStopping(verbose=False, output_best_model_path=str(tmp_path / "best.pth"))

    start_epoch = load_checkpoint_for_resume(
        model, optimizer, early_stopping, str(checkpoint_path), torch.device("cpu")
    )

    assert start_epoch == 3
    assert early_stopping.best_score == 0.5


def test_save_metadata_writes_json_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "TRAIN.h5"
    validation_path = tmp_path / "VALIDATION.h5"
    dataset_path.write_bytes(b"dataset-v1")
    validation_path.write_bytes(b"validation-v1")

    save_metadata(
        best_val_score=0.9,
        checkpoint={"epoch": 5},
        encoder="resnet34",
        architecture="UNET++",
        metadata_best_path=str(tmp_path / "best_model.pth"),
        val_loss=0.2,
        val_mcc=0.7,
        val_auroc=0.8,
        metadata_dir=str(tmp_path),
        amp_log={"precision": "fp32"},
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=2,
        seed=7,
        dataset=str(dataset_path),
        validation_dataset=str(validation_path),
        patience=3,
        optimizer_name="AdamW",
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
        run_ohem=False,
    )

    meta_path = tmp_path / "best_model_meta.json"
    assert meta_path.exists()
    contents = meta_path.read_text(encoding="utf-8")
    assert '"best_model_epoch": 5' in contents
    assert '"architecture": "UNET++"' in contents
    assert '"runtime_environment"' in contents


def test_save_metadata_records_reproducibility_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "TRAIN.h5"
    validation_path = tmp_path / "VALIDATION.h5"
    dataset_path.write_bytes(b"dataset-v1")
    validation_path.write_bytes(b"validation-v1")
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"manifest-v1")

    save_metadata(
        best_val_score=0.9,
        checkpoint={"epoch": 5},
        encoder="resnet34",
        architecture="UNET++",
        metadata_best_path=str(tmp_path / "best_model.pth"),
        val_loss=0.2,
        val_mcc=0.7,
        val_auroc=0.8,
        metadata_dir=str(tmp_path),
        amp_log={"precision": "fp32"},
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=2,
        seed=7,
        dataset=str(dataset_path),
        validation_dataset=str(validation_path),
        patience=3,
        optimizer_name="AdamW",
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
        execution_mode="PAPER",
        use_artifact_aware_loss=True,
        master_manifest_path=master_manifest_path,
        resume_checkpoint="resume.pth",
        run_ohem=True,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    payload = json.loads((tmp_path / "best_model_meta.json").read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "PAPER"
    assert payload["use_artifact_aware_loss"] is True
    assert payload["run_ohem"] is True
    assert payload["master_manifest_path"] == str(master_manifest_path)
    assert payload["resume_checkpoint"] == "resume.pth"
    assert "git_commit" in payload["runtime_environment"]


def test_save_metadata_writes_fail_closed_provenance_payload(tmp_path: Path) -> None:
    dataset_path = tmp_path / "TRAIN_FILTERED.h5"
    dataset_path.write_bytes(b"dataset-v1")
    validation_path = tmp_path / "VALIDATION.h5"
    validation_path.write_bytes(b"validation-v1")
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    master_manifest_path.write_bytes(b"manifest-v1")

    save_metadata(
        best_val_score=0.9,
        checkpoint={"epoch": 5},
        encoder="resnet34",
        architecture="UNET++",
        metadata_best_path=str(tmp_path / "best_model.pth"),
        val_loss=0.2,
        val_mcc=0.7,
        val_auroc=0.8,
        metadata_dir=str(tmp_path),
        amp_log={"precision": "fp32"},
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=2,
        seed=7,
        dataset=str(dataset_path),
        validation_dataset=str(validation_path),
        patience=3,
        optimizer_name="AdamW",
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
        execution_mode="PAPER",
        master_manifest_path=master_manifest_path,
        resume_checkpoint="resume.pth",
        run_ohem=True,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    payload = json.loads((tmp_path / "best_model_meta.json").read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert payload["compatibility_signature"]
    assert provenance["dataset"]["sha256"]
    assert provenance["validation_dataset"]["sha256"]
    assert provenance["validation_lineage"]["dataset_sha256"]
    assert provenance["artifact_aware_loss"]["enabled"] is True
    assert provenance["artifact_aware_loss"]["master_manifest_sha256"]
    assert provenance["ohem"]["enabled"] is True


def test_save_metadata_records_stage7_lineage_details_from_filtered_hdf5(tmp_path: Path) -> None:
    dataset_path = tmp_path / "TRAIN_FILTERED.h5"
    validation_path = tmp_path / "VALIDATION.h5"
    with h5py.File(dataset_path, "w") as handle:
        handle.attrs["selection_signature"] = "sig-123"
        handle.attrs["stage7_label_aware"] = True
        handle.attrs["stage7_selector"] = "legacy_adaptive_coverage"
        handle.attrs["stage7_model_name"] = "owkin/phikon-v2"
        handle.attrs["stage7_seed"] = 42
        handle.attrs["stage7_holdout_mode"] = "within_patient_patch_holdout"
        handle.attrs["stage7_protect_positive_labels"] = True
        handle.attrs["stage7_protect_mask_positive"] = True
        handle.attrs["stage7_positive_mask_fraction_threshold"] = 0.0
    with h5py.File(validation_path, "w"):
        pass

    save_metadata(
        best_val_score=0.9,
        checkpoint={"epoch": 5},
        encoder="resnet34",
        architecture="UNET++",
        metadata_best_path=str(tmp_path / "best_model.pth"),
        val_loss=0.2,
        val_mcc=0.7,
        val_auroc=0.8,
        metadata_dir=str(tmp_path),
        amp_log={"precision": "fp32"},
        base_learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        num_epochs=10,
        workers=2,
        seed=7,
        dataset=str(dataset_path),
        validation_dataset=str(validation_path),
        patience=3,
        optimizer_name="AdamW",
        alpha_bce=0.6,
        beta_dice_bg=0.2,
        gamma_dice_fg=0.8,
        run_ohem=False,
    )

    payload = json.loads((tmp_path / "best_model_meta.json").read_text(encoding="utf-8"))
    smart_sampling = payload["provenance"]["smart_sampling_lineage"]
    assert smart_sampling["enabled"] is True
    assert smart_sampling["selection_signature"] == "sig-123"
    assert smart_sampling["label_aware"] is True
    assert smart_sampling["selector"] == "legacy_adaptive_coverage"
    assert smart_sampling["model_name"] == "owkin/phikon-v2"
    assert smart_sampling["holdout_mode"] == "within_patient_patch_holdout"


def test_build_training_compatibility_signature_changes_with_validation_dataset(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "TRAIN.h5"
    validation_a_path = tmp_path / "VALIDATION_A.h5"
    validation_b_path = tmp_path / "VALIDATION_B.h5"
    train_path.write_bytes(b"train-v1")
    validation_a_path.write_bytes(b"validation-a")
    validation_b_path.write_bytes(b"validation-b")

    signature_a = build_training_compatibility_signature(
        dataset=str(train_path),
        validation_dataset=str(validation_a_path),
        master_manifest_path=None,
        run_ohem=False,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )
    signature_b = build_training_compatibility_signature(
        dataset=str(train_path),
        validation_dataset=str(validation_b_path),
        master_manifest_path=None,
        run_ohem=False,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    assert signature_a != signature_b


def test_build_training_compatibility_signature_changes_with_ohem_settings(tmp_path: Path) -> None:
    train_path = tmp_path / "TRAIN.h5"
    validation_path = tmp_path / "VALIDATION.h5"
    train_path.write_bytes(b"train-v1")
    validation_path.write_bytes(b"validation-v1")

    signature_a = build_training_compatibility_signature(
        dataset=str(train_path),
        validation_dataset=str(validation_path),
        master_manifest_path=None,
        run_ohem=False,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )
    signature_b = build_training_compatibility_signature(
        dataset=str(train_path),
        validation_dataset=str(validation_path),
        master_manifest_path=None,
        run_ohem=True,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    assert signature_a != signature_b


def test_build_training_compatibility_signature_accepts_precomputed_provenance() -> None:
    signature = build_training_compatibility_signature(
        dataset={
            "path": "TRAIN_FILTERED_shards/sample_manifest.parquet",
            "sha256": "train-sha",
            "source_signature": "train-sig",
            "selection_signature": "sel-sig",
            "smart_sampling_enabled": True,
            "smart_sampling_metadata": {"stage7_label_aware": True},
        },
        validation_dataset={
            "path": "VALIDATION_shards/sample_manifest.parquet",
            "sha256": "val-sha",
            "source_signature": "val-sig",
            "selection_signature": None,
            "smart_sampling_enabled": False,
            "smart_sampling_metadata": {},
        },
        master_manifest_path=None,
        run_ohem=False,
        ohem_start_epoch=2,
        ohem_ratio=0.25,
        ohem_min_kept=1024,
    )

    assert isinstance(signature, str)
    assert signature
